"""Trajectory-first chooser — drop-in alternative to `chooser.choose`.

Replaces the K-step fast_sim rollout + composite-leaf-value approach
with deterministic trajectory analysis + single-tick combat prediction.

Pipeline per turn:
  1. Iterate proposer's prerank candidates in cheap-Δ order.
  2. For each candidate `(src, tgt, ships, angle, eta, wait_N)`:
     a. `predict_fleet_fate(src, tgt, angle, ships, world)` — drop
        on `sun` / `oob` / `timeout` / `planet` (path-blocked by a
        different planet) / `comet_collision` (predicted-hit comet).
     b. For surviving "target" outcomes that ARE comets, drop if the
        comet's remaining lifetime ≤ ETA (expires before arrival).
     c. `predict_garrison_at(tgt, eta, ledger[tgt.id] + [our arrival])`
        — single-tick combat result.
     d. Score:
          captured (was-enemy → now-us)  → production × time_remaining,
                                            capped at comet life.
          reinforced (already-ours)      → skip (no extra credit;
                                            threat reinforcement is
                                            handled at proposer.propose).
          bounced (still-not-us)         → -ships (waste penalty).
  3. Sort surviving by score desc; greedy non-dogpile dedup by
     (src_id, tgt_id); emit `wait_N==0` winners only (`wait_N>0`
     reserves src+tgt, emits nothing this turn).

No K-step rollout, no leaf value-function approximation, no fast_sim
state cloning. Cost is O(candidates × (trajectory_steps + eta + arrivals)).

PI critique 2026-05-17: "we should be thinking in fleet trajectories";
"sun-deaths should be 0% with proper trajectory analysis". See
`knowledge-base/concepts/trajectory-first-architecture.md`.
"""

from __future__ import annotations

import math
import os
import time

from agents.baseline.chooser import affordable_validate_cap, opp_actions_for_snap
from agents.baseline.value import DEFAULT_GAMMA, select_favor_fn
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy as _me_policy
from lib.opp_model import me_defensive_action as _me_defends_policy
from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime, predict_garrison_at


EPISODE_STEPS_TOTAL: int = 500
WASTE_WEIGHT: float = 0.5
CAPTURE_REWARD_WEIGHT: float = 0.05

# Bug #14 fix attempt — CHEAP MIRROR. NEGATIVE RESULT 2026-05-18 PM.
#
# Premise: at each tick of the leaf rollout, drive ME with
# `lite_greedy_policy` (the same policy used for opp seats) instead
# of standing still after the single injected launch. Pre-fix the
# rollout was asymmetric: opp reacted each tick but WE didn't, so
# every candidate was scored against a worst-case "I make this move
# and then sit on my hands for 25 ticks while opp keeps playing"
# baseline. The asymmetry is documented in the bug catalog at
# `audit/2026-05-18-bug-catalog.md#14`.
#
# Empirical result with `BASELINE_ME_REACTS=1`:
# - The 3 xfail oracles (cleanup / coordinated / solo) did NOT flip
#   to pass — the mirror didn't unlock the expected coordination.
# - The 2 working defense oracles (defense_against_incoming_multi_fleet,
#   defense_wide_gap_multi_wave) REGRESSED to FAIL. The likely cause:
#   `lite_greedy_policy` is too greedy/attack-biased — in the baseline
#   it emits attack launches from the would-be reinforcer planet
#   (e.g. P1 with 200 ships → launches at opp), so the baseline lets
#   the threatened planet fall too. The candidate's reinforce can't
#   look more attractive than a baseline that's already failing in
#   the same way.
#
# PI's caveat was exactly this: "our chooser is meant to be SMARTER
# than lite_greedy. Using lite_greedy as our rollout policy
# UNDER-rates our skill." Under-rates badly enough to break working
# tests. The fix isn't viable as written.
#
# Next steps (deferred): try the "future-capture credit at the leaf"
# alternative (catalog option #3) — don't simulate us in the rollout
# at all; instead at the leaf add bonus credit for OUR in-flight
# captures past the leaf horizon. Captures the "we'll defend"
# intuition via accounting, not simulation, and doesn't depend on
# lite_greedy's tactical quality.
#
# The toggle and `_me_reactive_action` helper are kept so the
# experiment is reproducible. Default OFF — env var
# `BASELINE_ME_REACTS=1` to re-enable.
_ME_REACTS_ENABLED = os.environ.get("BASELINE_ME_REACTS", "0") != "0"


def _me_reactive_action(snap, me: int) -> list:
    """`lite_greedy_policy` driven from ME's observation. Same call
    shape as `opp_actions_for_snap` for non-me seats; isolated here so
    rollout sites stay readable."""
    try:
        return _me_policy(snap.state[me].observation) or []
    except Exception:
        return []


# Bug #14 fix — OPTION 5: PURELY DEFENSIVE policy for ME in the rollout
# (2026-05-18 PM). Supersedes the failed cheap-mirror (option 1 above).
# At each rollout tick, ME runs `lib.opp_model.me_defensive_action`:
# scan inbound enemy fleets, find under-defended owned planets, emit a
# reinforce launch from the nearest viable sister planet. Never attacks.
#
# Rationale vs the cheap-mirror failure: lite_greedy is too attack-biased
# — in the baseline path it emitted attack launches from the would-be
# reinforcer, so the baseline let the threatened planet fall too. A
# purely-defensive policy avoids that pathology because it never
# misallocates the reinforcer's ships to offense. The chooser's own
# attack moves are made on its real next turn; the rollout's job is to
# model opp's reaction (which implies us defending), not us attacking
# again.
#
# Default OFF (env var `BASELINE_ME_DEFENDS=1` to enable). When both
# DEFENDS and REACTS are set, DEFENDS takes precedence (REACTS is the
# deprecated cheap-mirror experiment kept only for reproducibility).
_ME_DEFENDS_ENABLED = os.environ.get("BASELINE_ME_DEFENDS", "0") != "0"


def _me_defensive_action(snap, me: int) -> list:
    """Defensive policy on ME's observation. Same call shape as
    `_me_reactive_action`; isolated here so the three rollout sites
    stay readable."""
    try:
        return _me_defends_policy(snap.state[me].observation, me) or []
    except Exception:
        return []

# How many ticks AFTER fleet arrival to keep simulating before reading
# the leaf. Long enough to see immediate combat aftermath (production
# tick, opp counter-arrivals already in flight); short enough that we
# don't run a full v15-style 40-step rollout. v3 trade-off vs v2:
# v2 used predict_garrison_at (single-tick static math); v3 uses
# fast_sim along the actual trajectory so the leaf reflects opp's
# reactive launches (via lite_greedy_policy each tick).
SETTLE_TURNS: int = 3

# Multi-launch budget (Step A, 2026-05-17 v2): the v1 chooser hard-
# capped at 1 launch per source per turn. v15 routinely emits
# parallel launches (3-5 sources × 1-2 fleets each); the 1/source
# limit was a load-bearing reason v1 lost 0/32 vs v15. v2 tracks
# remaining ships per source and emits multiple launches until the
# source's ships fall below the next candidate's requirement.
#
# MIN_SOURCE_RESERVE = 0: v15-line baseline does NOT hold a reserve;
# proposer's MIN_FLEET_SIZE filter ensures we never emit absurdly
# small launches. Holding even 2 ships back blocks the very common
# early-game case where a 10-ship home wants to send all 10 to a
# 5-ship neutral. Adopt v15's "spend it all if it captures" stance.
MIN_SOURCE_RESERVE: int = 0

# Opp 1-turn lookahead (Step C, 2026-05-17 v2): predict each opp
# source's likely best 1 launch; inject into ledger before scoring.
# Knobs first-pass; tune if A/B is borderline.
OPP_NEAREST_K: int = 4
OPP_SHIP_FRACTION: float = 0.8
OPP_MIN_SHIPS: int = 4

# Wallclock budgeting (2026-05-17 wait_N session): mirror composite
# chooser. v4 with wait_N>0 routinely blew the 1000ms env cap on heavy
# turns (max=2416ms in n=64 A/B vs v15). Composite stays within cap via
# affordable_validate_cap + safe_deadline pre-bail.
#
# N_VALIDATE=200 (vs composite's 60): trajectory v4's per-candidate
# cost is shorter on average (prop_horizon clamps to MIN_HORIZON=25 for
# most candidates; composite's avg horizon is closer to 32). The pre-
# wallclock-fix A/B (no cap) hit 65.6%; the N_VALIDATE=60 cap dropped
# it to 57.8% — confirming candidate breadth matters. Let safe_deadline
# bind the actual budget; N_VALIDATE is just a generous upper bound.
N_VALIDATE: int = 200
RESERVED_OVERHEAD_MS: float = 50.0

# Direction B — joint candidate evaluation (2026-05-18).
# Verified (C)+(E) via scripts/verify_solo_vs_joint.py on live episodes
# of 52754310 (mu=1271.8): solo launches from idle planets capture only
# 21pct of nearest targets (production growth out-paces accumulation);
# joint launches with a neighbor capture 89pct (+68pp lift).
# Opt-in via BASELINE_JOINT=1. Production stays on solo-only path.
JOINT_TOP_K_PER_TARGET: int = 3   # consider top-K solo candidates per target
JOINT_MAX_PAIRS: int = 20         # global cap to bound wallclock


def score_candidate(src, tgt, ships: int, angle: float, eta_hint: int,
                    me: int, world, ledger: dict,
                    ) -> tuple[float, str, int | None]:
    """Score a single candidate launch.

    Returns `(score, status, fate_step)`:
        `status` ∈ {'captured', 'reinforced', 'bounced', 'sun', 'oob',
                    'timeout', 'comet_collision', 'comet_expired',
                    'path_blocked'}
        `fate_step` = the tick of the resolving event (None if dropped pre-flight).
    """
    fate = predict_fleet_fate(src, tgt, angle, ships, world)

    if fate.outcome == "sun":
        return (float("-inf"), "sun", fate.step)
    if fate.outcome == "oob":
        return (float("-inf"), "oob", fate.step)
    if fate.outcome == "timeout":
        return (float("-inf"), "timeout", fate.step)
    if fate.outcome == "planet":
        # Hit a non-target planet first. Could be a comet collision
        # (engine treats comets as planets) — distinguish via comet_ids.
        if fate.hit_planet_id in world.comet_ids:
            return (float("-inf"), "comet_collision", fate.step)
        return (float("-inf"), "path_blocked", fate.step)

    # outcome == "target": fleet reaches the intended planet at fate.step.
    eta = int(fate.step)

    # Comet-expired guard: if the target IS a comet and runs out of
    # path at/before our arrival, the planet won't exist for capture.
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= eta:
            return (float("-inf"), "comet_expired", eta)

    # Sparse single-tick combat prediction. Include our hypothetical
    # arrival in the ledger so resolve_arrivals handles same-tick
    # combat correctly with any other fleets due that tick.
    base_arrivals = list(ledger.get(int(tgt.id), []))
    our_arrival = (eta, int(me), int(ships))
    pred_owner, _pred_garrison = predict_garrison_at(
        tgt, eta, base_arrivals + [our_arrival],
    )

    if pred_owner != me:
        # We didn't end up holding the planet — bounce / under-sized.
        return (-WASTE_WEIGHT * ships, "bounced", eta)

    # We hold the planet at eta. Was it ours before our arrival?
    # If the planet was already me (with no enemy interference), this
    # is reinforcement — no extra credit. Otherwise it's a capture.
    if int(tgt.owner) == me:
        # Check whether anything would flip it away from us between now
        # and eta-1 (in which case our arrival is a recapture).
        pred_owner_without_us, _ = predict_garrison_at(
            tgt, eta, base_arrivals,
        )
        if pred_owner_without_us == me:
            # Still ours without us — pure reinforcement.
            return (0.0, "reinforced", eta)
        # We recaptured a planet that would otherwise have been lost.
        # Credit the recapture like a fresh capture (production × time).
        time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
        held = time_remaining
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is not None:
                held = min(held, max(0, life - eta))
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
                "captured", eta)

    # Fresh capture (planet was not ours).
    time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
    held = time_remaining
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is not None:
            held = min(held, max(0, life - eta))
    return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
            "captured", eta)


def score_candidate_dyn(snap_base, src, tgt, ships: int, angle: float,
                        me: int, num_seats: int, world,
                        settle_turns: int = SETTLE_TURNS,
                        ) -> tuple[float, str, int | None]:
    """v3 dynamic scoring: fast_sim along the trajectory.

    Same admissibility gate as v2 (sun / oob / comet-collision /
    comet-expired-by-arrival are rejected deterministically by
    `predict_fleet_fate`). For surviving candidates, runs `fs_step`
    for `eta + settle_turns` ticks with our action injected at tick 0
    and `lite_greedy_policy` driving every opp seat reactively. Reads
    the target planet's ACTUAL owner from the simulated leaf —
    capturing whatever happens during flight (opp counter-launches,
    other fleets arriving, production accumulation, multi-fleet
    combat resolution).

    This is the convergence of trajectory thinking (deterministic
    admissibility filter, no expensive leaf value function) with
    the K-step rollout's strategic depth (reactive opp via fast_sim
    + lite_greedy). Cost per candidate ≈ (eta + settle) × per-step
    fast_sim cost (~0.5 ms). For eta=10, that's ~6.5 ms — vs v15's
    composite chooser ~20 ms (40-step rollout + 2-5 ms composite leaf).

    Returns `(score, status, eta)`. Statuses: same vocabulary as v2.
    """
    fate = predict_fleet_fate(src, tgt, angle, ships, world)
    if fate.outcome == "sun":
        return (float("-inf"), "sun", fate.step)
    if fate.outcome == "oob":
        return (float("-inf"), "oob", fate.step)
    if fate.outcome == "timeout":
        return (float("-inf"), "timeout", fate.step)
    if fate.outcome == "planet":
        if fate.hit_planet_id in world.comet_ids:
            return (float("-inf"), "comet_collision", fate.step)
        return (float("-inf"), "path_blocked", fate.step)

    eta = int(fate.step)
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= eta:
            return (float("-inf"), "comet_expired", eta)

    # Run fast_sim eta + settle ticks; inject our action at tick 0.
    snap = fs_clone(snap_base)
    horizon = eta + settle_turns
    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t == 0:
            actions[me] = [[int(src.id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)

    # Read target's leaf state from the simulated obs.
    leaf_obs = snap.state[me].observation
    leaf_planets = (
        leaf_obs.get("planets", []) if isinstance(leaf_obs, dict)
        else getattr(leaf_obs, "planets", [])
    )
    target_pid = int(tgt.id)
    leaf_owner: int = -2  # sentinel: target not found (e.g. expired comet)
    for p in leaf_planets:
        if int(p[0]) == target_pid:
            leaf_owner = int(p[1])
            break

    # Score from leaf outcome. The fast_sim leaf reflects opp's
    # reactive launches and production over `eta + settle` ticks, so
    # "owner == me at leaf" is a much stronger signal than the static
    # predict_garrison_at v2 used.
    if leaf_owner == me:
        # Was it ours BEFORE the launch?
        if int(tgt.owner) == me:
            # Already ours, still ours — pure reinforcement (no extra credit).
            return (0.0, "reinforced", eta)
        # Captured (or recaptured a planet that would have fallen).
        time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
        held = time_remaining
        if target_pid in world.comet_ids:
            life = comet_remaining_lifetime(target_pid, world)
            if life is not None:
                held = min(held, max(0, life - eta))
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
                "captured", eta)

    # Leaf shows target NOT ours: either bounce (still enemy/neutral)
    # or the planet vanished (comet expired). Both → waste.
    return (-WASTE_WEIGHT * ships, "bounced", eta)


# ---------------------------------------------------------------------------
# v4 (Direction A, 2026-05-17 PM): favor leaf + idle-baseline Δ-scoring.
# ---------------------------------------------------------------------------
#
# v3 lost 0/32 vs v15 with a BINARY leaf (target.owner == me at leaf?).
# Hypothesis: binary scoring collapses ~bits of strategic info that the
# v15-style continuous `favor` leaf preserves (ship balance + production
# balance × pv_horizon). v4 replaces v3's binary check with v15's
# Δ-from-idle-baseline-favor scoring, keeping v3's eta-bounded
# trajectory rollout (cheaper than v15's fixed K=40).
#
# If v4 reaches v15 parity: information-collapse hypothesis confirmed;
# trajectory chooser was architecturally fine, leaf was the bug.
# If v4 still loses: scoring isn't the binding constraint; pivot to
# joint-action or sequential planning (Directions B/C in concept doc).


def build_trajectory_baseline(snap_base, me: int, num_seats: int,
                              horizon: int, favor_fn, gamma: float,
                              ) -> list[float]:
    """Idle-baseline favor at each tick in [0, horizon]. Used by v4 to
    subtract the do-nothing alternative from each candidate's leaf
    favor (mirrors `chooser.build_idle_baseline`).

    Returns a list of length `horizon + 1`. Cost: `horizon` calls to
    `fs_step` + `(horizon + 1)` calls to `favor_fn`. Runs ONCE per
    chooser invocation, not per candidate.

    Bug #14 fix (2026-05-18): when `BASELINE_ME_REACTS=1`, the
    baseline ALSO has ME play `lite_greedy_policy` reactively each
    tick — same policy used for opp seats. Reason: if only the
    candidate path has ME reactive but the baseline doesn't, the Δ
    captures "value of ME playing at all" rather than "value of THIS
    candidate's specific move." Symmetric framing isolates the
    candidate's marginal contribution.
    """
    snap = fs_clone(snap_base)
    out: list[float] = [
        favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma),
    ]
    for _ in range(horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = opp_actions_for_snap(snap, me, num_seats)
        # Baseline IS asymmetric on purpose (ME idle, opp reactive) —
        # we measure the candidate's marginal value above the worst-
        # case "I do nothing this turn AND on every future turn"
        # outcome. Auto-defense applies in CANDIDATE rollouts only
        # (sites B/C below), where it represents "future me reacting
        # to opp's response to MY move." Adding auto-defense here
        # makes the baseline too capable and zeros the candidate-Δ
        # for defensive launches (auto-defense already handles them),
        # so the chooser refuses to emit real defense.
        if _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)
        out.append(
            favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma),
        )
    return out


def score_candidate_v4(snap_base, src, tgt, ships: int, angle: float,
                       me: int, num_seats: int, world,
                       baseline_favors: list[float],
                       favor_fn, gamma: float,
                       horizon: int,
                       skip_admissibility: bool = False,
                       wait_N: int = 0,
                       ) -> tuple[float, str, int | None]:
    """v4 scoring: same admissibility filter + fast_sim rollout as v3,
    but the leaf is `favor_fn` instead of a binary owner-check, and the
    score is `Δ = leaf_with_action − baseline_favors[horizon]`.

    Returns `(delta, status, eta)`. Statuses match v3 plus "scored"
    (the success case for v4, since "captured/reinforced/bounced" are
    no longer first-class outcomes — favor implicitly encodes them).

    `skip_admissibility=True` bypasses the predict_fleet_fate filter
    (env var TRAJECTORY_SKIP_ADMISSIBILITY=on) to isolate whether the
    filter is false-rejecting valid candidates that composite_a2 lets
    through.

    `wait_N>0` defers action injection to step `wait_N` in the rollout
    (matches composite chooser's `score_action` pattern at
    `agents/baseline/chooser.py:60-73`). Admissibility filter only runs
    for `wait_N==0` (the source planet orbits between now and the wait
    point, so the pre-launch trajectory analysis is stale); for wait>0
    candidates, fast_sim's collision resolution catches real sun/oob/
    comet hits inside the rollout.
    """
    eta = 0
    if not skip_admissibility and int(wait_N) == 0:
        fate = predict_fleet_fate(src, tgt, angle, ships, world)
        if fate.outcome == "sun":
            return (float("-inf"), "sun", fate.step)
        if fate.outcome == "oob":
            return (float("-inf"), "oob", fate.step)
        if fate.outcome == "timeout":
            return (float("-inf"), "timeout", fate.step)
        if fate.outcome == "planet":
            if fate.hit_planet_id in world.comet_ids:
                return (float("-inf"), "comet_collision", fate.step)
            return (float("-inf"), "path_blocked", fate.step)

        eta = int(fate.step)
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is None or life <= eta:
                return (float("-inf"), "comet_expired", eta)

    # Clamp horizon to baseline length (caller pre-sized).
    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    snap = fs_clone(snap_base)
    # Bug #14 option 5: precompute defensive reinforces ONCE from the
    # candidate's tick-0 observation. In real game the chooser emits
    # ALL of this turn's moves (candidate + any reactive defense)
    # simultaneously — we model that by merging them at `wait_N` and
    # NOT re-evaluating defense on every rollout tick. Per-call cost
    # drops from `horizon × per-candidate` to `1 × per-candidate`,
    # which is the difference between bench-WATCH (10 outliers >1s
    # at horizon=25 with the per-tick variant) and bench-PASS.
    me_defense_emits: list = []
    if _ME_DEFENDS_ENABLED:
        me_defense_emits = _me_defensive_action(snap, me)

    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t == int(wait_N):
            # Candidate first (the chooser's primary decision), then
            # defensive emits. If the candidate drains the planet
            # defense wanted as reinforcer, fs_step will cap the
            # defensive launch at remaining ships.
            actions[me] = (
                [[int(src.id), float(angle), int(ships)]]
                + list(me_defense_emits)
            )
        elif _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    delta = leaf - baseline_favors[horizon]
    return (delta, "scored", eta)


def score_candidate_v4_joint(snap_base, launches, me: int, num_seats: int,
                              world,
                              baseline_favors: list[float],
                              favor_fn, gamma: float,
                              horizon: int,
                              skip_admissibility: bool = False,
                              ) -> tuple[float, str]:
    """Direction B: score a JOINT candidate of multiple launches in one
    fast_sim rollout. `launches` is a list of
    `(src, tgt, ships, angle, wait_N)` tuples — all injected at their
    respective wait_N steps in the SAME rollout. Leaf scoring identical
    to v4 solo: Δ = leaf − baseline_favors[horizon].

    Returns `(delta, status)` where status is:
      - 'admissibility_fail' if any wait_N==0 leg fails predict_fleet_fate
      - 'comet_expired' if any leg targets a comet that expires before eta
      - 'scored' otherwise

    For v1 simplicity, all legs typically have wait_N==0 (fire-now joint).
    Multi-wait joints are valid by construction but not enumerated yet
    (see proposer path in `choose_trajectory`).
    """
    # Per-leg admissibility filter (only meaningful for wait_N==0 legs).
    for src, tgt, ships, angle, wait_N in launches:
        if skip_admissibility or int(wait_N) != 0:
            continue
        fate = predict_fleet_fate(src, tgt, angle, ships, world)
        if fate.outcome == "sun":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "oob":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "timeout":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "planet":
            return (float("-inf"), "admissibility_fail")
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is None or life <= int(fate.step):
                return (float("-inf"), "comet_expired")

    # Clamp horizon to baseline length.
    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    # Build the inject schedule keyed by wait_N step.
    inject_at: dict[int, list] = {}
    for src, tgt, ships, angle, wait_N in launches:
        inject_at.setdefault(int(wait_N), []).append(
            [int(src.id), float(angle), int(ships)],
        )

    snap = fs_clone(snap_base)
    # Defensive emits computed once from tick-0 obs, attached to the
    # earliest inject step (typically wait_N=0). Matches the
    # score_candidate_v4 wiring — see comment there.
    me_defense_emits: list = []
    if _ME_DEFENDS_ENABLED:
        me_defense_emits = _me_defensive_action(snap, me)
    earliest_inject_t = min(inject_at.keys()) if inject_at else -1

    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t in inject_at:
            base_actions = list(inject_at[t])
            if t == earliest_inject_t and me_defense_emits:
                base_actions = base_actions + list(me_defense_emits)
            actions[me] = base_actions
        elif _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    return (leaf - baseline_favors[horizon], "scored")


def predict_opp_responses(world, me: int, num_seats: int,
                          ) -> list[tuple[int, int, int, int]]:
    """1-turn opp lookahead: project each enemy source's likely best
    launch into a list of (target_pid, eta, opp_owner, ships) tuples
    that can be merged into our arrival ledger.

    Heuristic per opp source: scan its `OPP_NEAREST_K` nearest non-opp
    targets; pick the first one whose straight-line trajectory is
    admissible (no sun, no oob, no comet collision). Project a fleet
    of `OPP_SHIP_FRACTION × src.ships` ships.

    Closes Gap 2 of v1's 0/32 failure: composite-head A/B's K-step
    rollout simulates opp counter-launches via lib.opp_model.
    lite_greedy_policy; v1 trajectory chooser had no opp model so
    every candidate was scored as if the opp would play idle. With
    this projection, our score_candidate sees the ledger with the
    enemy's likely counter-fleet already accounted for.

    First-pass heuristic; not a full opp model. May:
      - Overestimate opp competence (assumes they pick optimal target).
      - Underestimate launches (only 1 per source).
      - Miss multi-target threats (e.g. gang-ups).
    All tolerable for a first-cut A/B; refine if results are promising.
    """
    opp_arrivals: list[tuple[int, int, int, int]] = []
    all_planets = list(world.planets_by_id.values())
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        opp_planets = [p for p in all_planets if int(p.owner) == opp_id]
        for opp_src in opp_planets:
            if int(opp_src.ships) < OPP_MIN_SHIPS:
                continue
            # Nearest non-opp targets.
            others = sorted(
                ((math.hypot(p.x - opp_src.x, p.y - opp_src.y), p)
                 for p in all_planets
                 if int(p.owner) != opp_id and int(p.id) != int(opp_src.id)),
                key=lambda d_p: d_p[0],
            )
            ships = max(1, int(int(opp_src.ships) * OPP_SHIP_FRACTION))
            for _d, opp_tgt in others[:OPP_NEAREST_K]:
                angle = math.atan2(opp_tgt.y - opp_src.y,
                                   opp_tgt.x - opp_src.x)
                fate = predict_fleet_fate(opp_src, opp_tgt, angle,
                                          ships, world)
                if fate.outcome == "target":
                    opp_arrivals.append(
                        (int(opp_tgt.id), int(fate.step), opp_id, ships),
                    )
                    break  # 1 projection per opp source is enough
    return opp_arrivals


def merge_ledgers(base: dict, projected: list[tuple[int, int, int, int]],
                  ) -> dict:
    """Add projected (target_pid, eta, owner, ships) tuples into a copy
    of `base` (per-planet list of (eta, owner, ships))."""
    out = {pid: list(v) for pid, v in base.items()}
    for tgt_pid, eta, owner, ships in projected:
        out.setdefault(tgt_pid, []).append((eta, owner, ships))
    return out


def choose_trajectory(snap_base, prerank, baseline_favors,
                      me: int, num_seats: int, wallclock_ms: float,
                      min_horizon: int, max_horizon: int, gamma: float,
                      world, model) -> list[list]:
    """Drop-in alternative to `chooser.choose`.

    The `snap_base` / `baseline_favors` / `min_horizon` / `max_horizon`
    / `gamma` args are unused (kept for signature parity with
    `chooser.choose` so the dispatcher in `main.py` is a simple swap).
    The trajectory chooser doesn't roll out and doesn't need an idle
    baseline.

    v2 (2026-05-17 PM):
    - 1-turn opp lookahead: predict_opp_responses projects each opp
      source's best launch; ledger merged BEFORE scoring (every
      predict_garrison_at sees the pessimistic state).
    - Multi-launch budget: drops "1 launch per source" dedup; tracks
      ship sub-budget per source.
    """
    if not prerank:
        return []

    deadline = time.perf_counter() + wallclock_ms / 1000.0

    # v4 (default, 2026-05-17 PM): Δ-from-idle-baseline scoring with
    # favor leaf. Replaces v3's binary owner-check leaf — see concept
    # at knowledge-base/concepts/probability-of-winning-framework.md.
    # Use BASELINE_CHOOSER=trajectory_v3 to force the v3 (binary leaf)
    # path for A/B comparison.
    use_v3 = (
        os.environ.get("BASELINE_CHOOSER", "").strip().lower()
        == "trajectory_v3"
    )
    skip_filter = (
        os.environ.get("TRAJECTORY_SKIP_ADMISSIBILITY", "").strip().lower()
        == "on"
    )
    favor_fn = select_favor_fn()  # honours BASELINE_VALUE_HEAD env var

    # Pre-pass: find the largest horizon we'll need so the baseline runs
    # deep enough for every candidate (including wait_N>0, whose proposer
    # horizon already accounts for the wait via
    # `w_horizon = max(w_wait + w_eta + SIM_SETTLE_TURNS, MIN_HORIZON)`).
    max_horizon_seen = 0
    for cheap_delta, src, tgt, ships, angle, eta_hint, h, wait_N in prerank:
        if int(h) > max_horizon_seen:
            max_horizon_seen = int(h)

    baseline_favors: list[float] = []
    if not use_v3 and max_horizon_seen > 0:
        baseline_favors = build_trajectory_baseline(
            snap_base, me, num_seats, max_horizon_seen, favor_fn, gamma,
        )

    # Wallclock budgeting (mirror composite chooser pattern). Probe per-
    # step + per-leaf cost to size the safe_deadline pre-bail. The hard
    # cap stays at N_VALIDATE (generous); safe_deadline is the real
    # binder so score_candidate_v4's uninterruptible rollout never
    # starts past the cliff. Closes the n=64 A/B max=2416ms overrun
    # (1000ms env cap) without the N_VALIDATE=60 candidate-breadth
    # regression (57.8% vs pre-fix 65.6% in the post-N=60 A/B).
    cap = N_VALIDATE
    per_cand_ms = 0.0
    if not use_v3:
        remaining_ms = max(50.0, (deadline - time.perf_counter()) * 1000.0)
        _, per_cand_ms = affordable_validate_cap(
            snap_base, me, num_seats, max_horizon, remaining_ms,
            min_horizon, gamma,
        )
    safe_deadline = deadline - (per_cand_ms / 1000.0)

    scored: list[tuple] = []
    solo_winners: set[int] = set()  # src_ids whose solo scored Δ>0
    cand_count = 0
    for cheap_delta, src, tgt, ships, angle, eta_hint, prop_horizon, wait_N in prerank:
        if cand_count >= cap:
            break
        if not use_v3 and time.perf_counter() > safe_deadline:
            break
        cand_count += 1
        if use_v3:
            # v3 path: fire-now-only (binary leaf doesn't generalise to
            # wait_N>0 trivially). Skip wait_N>0 in the v3 path.
            if int(wait_N) != 0:
                continue
            score, status, _ = score_candidate_dyn(
                snap_base, src, tgt, int(ships), float(angle),
                me, num_seats, world,
            )
            if status in ("captured",) and score > 0.0:
                scored.append((score, src, tgt, ships, angle, wait_N))
        else:
            score, status, _ = score_candidate_v4(
                snap_base, src, tgt, int(ships), float(angle),
                me, num_seats, world,
                baseline_favors, favor_fn, gamma,
                horizon=int(prop_horizon),
                skip_admissibility=skip_filter,
                wait_N=int(wait_N),
            )
            if status == "scored" and score > 0.0:
                scored.append((score, src, tgt, ships, angle, wait_N))
                # Track sources with viable solo (for joint gating).
                solo_winners.add(int(src.id))

    # Direction B v3 (2026-05-18 PM): 2P-only gate added after v2's
    # 4P regression (4/32 first-place = 12.5pct in 8-seed × 4-seat
    # rotation vs 3x hybrid). 4P joint commits 2 srcs against one of
    # 3 opponents, leaving the other 2 opps free to attack our weakened
    # planets. 2P-only is the same defensive shape as favor_hybrid_spatial
    # in commit 558bd61. 2P joint v2 A/B 38/64 = 59.4pct (Wlo=0.471,
    # INCONCL-but-positive vs hybrid).
    joint_enabled = (
        os.environ.get("BASELINE_JOINT", "0").strip() == "1"
        and int(num_seats) <= 2
    )
    if (joint_enabled and not use_v3
            and time.perf_counter() <= safe_deadline):
        # Group prerank by target_id. Take top-K solo candidates per
        # target by cheap_delta; pair-enumerate.
        by_tgt: dict[int, list] = {}
        for cd, src, tgt, ships, angle, eta_hint, ph, wn in prerank:
            if int(wn) != 0:
                continue  # v1: fire-now joints only
            by_tgt.setdefault(int(tgt.id), []).append(
                (float(cd), src, tgt, int(ships), float(angle), int(ph)),
            )
        joint_count = 0
        for tgt_id, cands in by_tgt.items():
            if len(cands) < 2:
                continue
            cands.sort(key=lambda c: -c[0])
            top = cands[:JOINT_TOP_K_PER_TARGET]
            for i in range(len(top)):
                if joint_count >= JOINT_MAX_PAIRS:
                    break
                if time.perf_counter() > safe_deadline:
                    break
                for j in range(i + 1, len(top)):
                    if joint_count >= JOINT_MAX_PAIRS:
                        break
                    if time.perf_counter() > safe_deadline:
                        break
                    ca, cb = top[i], top[j]
                    if int(ca[1].id) == int(cb[1].id):
                        continue  # same source → not a joint
                    # Gate: at least one constituent must be a FAILING
                    # solo. If both srcs already have viable solo
                    # captures, joint over-bundles them and the emit
                    # logic would lose the cheaper independent path.
                    if (int(ca[1].id) in solo_winners
                            and int(cb[1].id) in solo_winners):
                        continue
                    launches = [
                        (ca[1], ca[2], ca[3], ca[4], 0),
                        (cb[1], cb[2], cb[3], cb[4], 0),
                    ]
                    jh = max(int(ca[5]), int(cb[5]))
                    j_score, j_status = score_candidate_v4_joint(
                        snap_base, launches, me, num_seats, world,
                        baseline_favors, favor_fn, gamma,
                        horizon=jh, skip_admissibility=skip_filter,
                    )
                    joint_count += 1
                    if j_status == "scored" and j_score > 0.0:
                        scored.append((j_score, "joint", launches))

    if not scored:
        return []

    scored.sort(key=lambda c: -c[0])

    # Emit logic — match composite chooser (`agents/baseline/chooser.choose`)
    # for parity. 1 launch per source per turn, 1 per target. For joints
    # (tagged 'joint' tuples), require ALL of its sources and targets to
    # be free; commit all legs together.
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for entry in scored:
        # Joint candidates are 3-tuples: (score, 'joint', launches).
        if len(entry) == 3 and entry[1] == "joint":
            _score, _tag, launches = entry
            if any(int(L[0].id) in used_srcs for L in launches):
                continue
            if any(int(L[1].id) in used_tgts for L in launches):
                continue
            for src, tgt, ships, angle, wait_N in launches:
                used_srcs.add(int(src.id))
                used_tgts.add(int(tgt.id))
                if int(wait_N) == 0:
                    moves.append([int(src.id), float(angle), int(ships)])
            continue
        # Solo: legacy 6-tuple (score, src, tgt, ships, angle, wait_N).
        _score, src, tgt, ships, angle, wait_N = entry
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
    return moves
