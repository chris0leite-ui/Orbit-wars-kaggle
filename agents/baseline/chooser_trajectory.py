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

from agents.baseline.chooser import HARDCAP_BAIL_SENTINEL, WALLCLOCK_HARD_CAP_MS, affordable_validate_cap, opp_actions_for_snap
from agents.baseline.value import DEFAULT_GAMMA, select_favor_fn
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy as _me_policy
from lib.opp_model import me_defensive_action as _me_defends_policy
from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime, predict_garrison_at
from agents.baseline.proposer import MIN_FLEET_SIZE
from agents.baseline.proposer import aim_and_eta as _aim_and_eta
from agents.baseline.proposer import capture_size as _capture_size
from agents.baseline.proposer import hold_need
from agents.baseline.proposer import nearest_k as _nearest_k
from agents.baseline.launch_rules import capture_horizon_k as _capture_horizon_k
from agents.baseline.launch_rules import launch_rules_enabled as _launch_rules_enabled
from agents.baseline.launch_rules import state_driven_k_enabled as _state_driven_k_enabled


EPISODE_STEPS_TOTAL: int = 500
WASTE_WEIGHT: float = 0.5
CAPTURE_REWARD_WEIGHT: float = 0.05

# Leader-focus bonus (2026-05-21). In 4P, multiply capture score by
# LEADER_FOCUS_WEIGHT when the target's owner is the current leader
# (player with highest planet-production-owned). Pushes focal to attack
# the strongest opp rather than spreading attacks across all three.
# Disabled (=1.0) in 2P automatically since there is no leader to
# distinguish. Default 1.0 (no change); opt-in via env var.
LEADER_FOCUS_WEIGHT: float = float(os.environ.get("BASELINE_LEADER_FOCUS", "1.0"))

# Neutral-capture bonus (2026-05-21). In 4P trace of seed=5 (a loss),
# focal made 73 captures-from-enemy but only 6 captures-from-neutral
# while phase_c snowballed to 36 planets via aggressive neutral grab.
# This bonus tilts the chooser toward neutral targets relative to
# enemy targets — neutrals don't have a defender (cheaper) and grow
# production without risk of attrition. Stronger in the opening phase
# where territorial grab dominates outcomes. Default 1.0 (no change);
# opt-in via env var BASELINE_NEUTRAL_BONUS.
NEUTRAL_BONUS_WEIGHT: float = float(os.environ.get("BASELINE_NEUTRAL_BONUS", "1.0"))
NEUTRAL_EARLY_HORIZON: int = int(os.environ.get("BASELINE_NEUTRAL_EARLY_HORIZON", "50"))
NEUTRAL_EARLY_EXTRA: float = float(os.environ.get("BASELINE_NEUTRAL_EARLY_EXTRA", "1.0"))

# Follow-on hold bonus (Fix 2b — 2026-05-27 plan). Opt-in scoring
# bonus on captures that enable a profitable follow-on launch from the
# newly-captured base. Surfaces the B3/B4 modeling-correct snipe
# helpers (`_best_followon` / `_followon_hold_estimate` in
# `lib/missions/snipe.py`) into the live trajectory chooser path —
# previously B3/B4 were in dead code from this agent's perspective.
# Default 0.0 (no-op); bundle wrapper opts in once local A/B confirms
# lift. Calibrated against `CAPTURE_REWARD_WEIGHT=0.05`.
FOLLOWON_BONUS_WEIGHT: float = float(
    os.environ.get("BASELINE_FOLLOWON_BONUS", "0.0"),
)
FOLLOWON_RADIUS: float = float(
    os.environ.get("BASELINE_FOLLOWON_RADIUS", "35.0"),
)

# Expansion credit (2026-06-01) — restore the held-production term v4 dropped.
# v4 scores a candidate purely as the rollout delta `leaf − baseline`, on the
# (proven-false) assumption that `favor_fn` implicitly encodes capture value.
# It does not: the rollout drives ME passively after the launch while every
# opponent reacts each tick, so a freshly captured planet is "eaten" by the
# leaf and delta nets <=0. Result = the hoarding pathology (the agent refuses
# to spend idle ships on free neutrals; see
# audit/2026-06-01-live-replay-diagnosis.md and the step-39 istinetz trace —
# a +916-production neutral scored -0.99). This restores the v2 static
# scorer's additive credit `CAPTURE_REWARD_WEIGHT × production × held` for
# FRESH captures, applied UN-gated (NOT conditional on delta>0, unlike
# NEUTRAL/FOLLOWON which can only amplify already-positive deltas). The
# rollout delta then remains as the recapture-risk CORRECTION on top: good
# expansions clear, captures the rollout shows get eaten still net negative.
# Default 0.0 (OFF, byte-for-byte legacy). 1.0 = full v2-equivalent credit;
# tune the fraction via proper n>=32 + panel A/B (NOT n=16).
EXPAND_CREDIT_WEIGHT: float = float(
    os.environ.get("BASELINE_EXPAND_CREDIT", "0.0"),
)

# Score floor for emit (2026-05-27 — concentration knob). Today every
# candidate with `score > 0.0` fires; in midgame this scatters small
# marginal launches across every owned planet. `MIN_DELTA` raises the
# floor so only candidates above a tunable threshold survive — natural
# concentration without an arbitrary count-cap. Default 0.0 preserves
# byte-for-byte legacy (strict `> 0.0`); positive values install a
# strict `>=` floor. Units are PV-discounted delta (see
# `score_candidate_v4`); tune via local A/B.
MIN_DELTA: float = float(os.environ.get("BASELINE_MIN_DELTA", "0.0"))

# Ship-turn opportunity-cost penalty (2026-05-27 Step 2B). Today the
# leaf `favor` returns ~297-340 for any positive-prod capture regardless
# of eta — pv_horizon(leaf_step, 0) ≈ 99 for any leaf_step in 25..50
# with γ=0.99, t_total=500. Result: slow captures (eta=40) score ~88%
# of fast captures (eta=10) when in reality they tie up ships 4x
# longer. Penalty subtracts κ × ships × (wait_N + eta) from delta to
# price the time the ships are committed and unable to defend/redirect.
# Default 0.0 preserves byte-for-byte legacy. Tune via local A/B.
SHIP_TURN_KAPPA: float = float(
    os.environ.get("BASELINE_SHIP_TURN_KAPPA", "0.0"),
)

# Present-value time-discount on candidate Δ (2026-05-28). The favor
# leaf computes pv_horizon(step, 0) — eta hardcoded to zero — so a
# capture arriving in 10 turns is valued ~99% of a capture arriving
# in 40. This applies γ^(wait_N + eta) to the final Δ, pulling each
# candidate's payoff back to the current step. No new tuning knob: γ
# is the existing chooser discount (BASELINE_GAMMA, peak default 0.99).
# Default OFF preserves byte-for-byte legacy. Tune via local A/B.
PV_ETA_ENABLED: bool = (
    os.environ.get("BASELINE_PV_ETA", "0").strip() == "1"
)


def _leader_owner_from_world(world, me: int) -> int | None:
    """Return the player id (other than `me`) with the highest total
    planet production owned. Returns None when leader is undefined
    (no opps, single opp, or production tie).
    """
    if world is None:
        return None
    prod_by_owner: dict[int, int] = {}
    try:
        plist = list(world.planets_by_id.values())
    except AttributeError:
        return None
    for p in plist:
        o = int(getattr(p, "owner", -1))
        if o < 0 or o == int(me):
            continue
        prod_by_owner[o] = prod_by_owner.get(o, 0) + int(getattr(p, "production", 0))
    if len(prod_by_owner) < 2:
        return None  # 2P or only one opp; no leader distinction
    best = max(prod_by_owner.values())
    leaders = [o for o, v in prod_by_owner.items() if v == best]
    if len(leaders) > 1:
        return None  # tie, no clear leader
    return leaders[0]

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
# Caps env-var configurable so a variant can be A/B'd without rebuilding
# the bundle. Default values match the original 2026-05-18 v3 production
# constants. `BASELINE_JOINT_AGGR=1` ALSO lifts the `used_tgts` lock on
# both solo and joint emits (multi-source-same-target stacking — combat
# rule 1 exploit). Origin: 2026-05-20 PI directive to find a STRUCTURAL
# lift over the n=8 ablation plateau. Risk: ship-waste from over-emit to
# same target; controlled by leaving `used_srcs` lock in place.
# Verified (C)+(E) via scripts/verify_solo_vs_joint.py on live episodes
# of 52754310 (mu=1271.8): solo launches from idle planets capture only
# 21pct of nearest targets (production growth out-paces accumulation);
# joint launches with a neighbor capture 89pct (+68pp lift).
# Opt-in via BASELINE_JOINT=1. Production stays on solo-only path.
JOINT_TOP_K_PER_TARGET: int = int(
    os.environ.get("BASELINE_JOINT_TOP_K", "3")
)
JOINT_MAX_PAIRS: int = int(
    os.environ.get("BASELINE_JOINT_MAX_PAIRS", "20")
)
JOINT_LIFT_USED_TGTS: bool = (
    os.environ.get("BASELINE_JOINT_AGGR", "0").strip() == "1"
)

# Synchronized-arrival JOINT coalitions (2026-05-30, BASELINE_JOINT_SYNC,
# DEFAULT OFF). The existing fire-now joint launches both legs at wait_N=0,
# so sources at different distances arrive on DIFFERENT ticks and never
# stack. This adds coalitions where the closer source WAITS so all legs
# land the same tick and combine forces (combat rule 1, interpreter.py),
# capturing a planet neither source could take alone. Scored by the same
# score_candidate_v4_joint rollout (reactive-opp, so the old v15 dogpile
# falsification — which had no reactive opp — does not transfer). The
# waiting leg fires via the self-contained sync-joint tick in main.py
# (gated on BASELINE_JOINT_SYNC=1; does NOT touch the general ledger).
JOINT_SYNC_ENABLED: bool = (
    os.environ.get("BASELINE_JOINT_SYNC", "0").strip() == "1"
)
JOINT_SYNC_MAX_PAIRS: int = int(
    os.environ.get("BASELINE_JOINT_SYNC_MAX_PAIRS", "30")
)
JOINT_SYNC_SETTLE: int = 2  # ticks past arrival the leaf must see (Rule 47)
JOINT_SYNC_SRC_K: int = int(
    os.environ.get("BASELINE_JOINT_SYNC_SRC_K", "3")
)  # nearest friendly sources pulled per target (2 nearest form the pair;
#    the 3rd is headroom for a future 3-source extension, not built yet)


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

    # Leader-focus bonus: 4P-only (in 2P _leader_owner_from_world returns None).
    bonus = 1.0
    if LEADER_FOCUS_WEIGHT != 1.0:
        leader = _leader_owner_from_world(world, me)
        if leader is not None and int(tgt.owner) == int(leader):
            bonus = LEADER_FOCUS_WEIGHT

    # Neutral-capture bonus: applies when the target is currently neutral
    # (tgt.owner == -1). Optional opening-phase extra multiplier for the
    # first NEUTRAL_EARLY_HORIZON steps to accelerate territorial grab.
    if NEUTRAL_BONUS_WEIGHT != 1.0 and int(tgt.owner) == -1:
        bonus *= NEUTRAL_BONUS_WEIGHT
        if int(world.step) < NEUTRAL_EARLY_HORIZON:
            bonus *= NEUTRAL_EARLY_EXTRA

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
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held) * bonus,
                "captured", eta)

    # Fresh capture (planet was not ours).
    time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
    held = time_remaining
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is not None:
            held = min(held, max(0, life - eta))
    return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held) * bonus,
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
                       eta_hint: int = 0,
                       model=None,
                       hard_deadline: float | None = None,
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
    # For wait_N==0 candidates, eta is computed below from
    # predict_fleet_fate. For wait_N>0 candidates, admissibility is
    # skipped (source orbit drifts between now and the wait point); use
    # the proposer's eta_hint so the downstream PV-discount sees a
    # non-zero arrival time. skip_admissibility=True (debug ablation)
    # keeps the historical eta=0 default to preserve test fixtures.
    eta = int(eta_hint) if (int(wait_N) > 0 and not skip_admissibility) else 0
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

    _last_step_s = 0.0
    for t in range(horizon):
        if snap.fake_env.done:
            break
        # Predictive deadline guard: bail if the NEXT rollout step is
        # projected to cross the hard cap, rather than reacting after a step
        # has already overshot. `opp_actions_for_snap` + `fs_step` is an
        # uninterruptible chunk whose cost scales with board density; at
        # state-driven-K's horizon (ceil 30) a single step in a dense
        # late-game state can run ~250ms, so the old react-after check
        # overshot the 1000ms env cap (observed max=1208ms). Projecting one
        # step ahead self-tunes to the measured per-step cost and never
        # starts a step that would blow the deadline. Modeling-correct: the
        # rollout spends exactly the time it has, then stops cleanly.
        if hard_deadline is not None and \
                time.perf_counter() + _last_step_s > hard_deadline:
            return (HARDCAP_BAIL_SENTINEL, "hardcap_bail", eta)
        _step_t0 = time.perf_counter()
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
        _last_step_s = time.perf_counter() - _step_t0

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    delta = leaf - baseline_favors[horizon]

    # Plumb NEUTRAL_BONUS + LEADER_FOCUS into the live scoring path.
    # The earlier dead-code path (`score_candidate`, v2 static-garrison
    # scorer) read these env vars but was never called; v4 ignored them.
    # Multiply POSITIVE deltas only — the bonus is a tilt toward
    # preferred targets, not a punishment for bad candidates that
    # happen to be neutral/leader. See plan
    # `/root/.claude/plans/fix-one-and-two-cuddly-dewdrop.md` Fix 1.
    if delta > 0.0:
        bonus = 1.0
        if NEUTRAL_BONUS_WEIGHT != 1.0 and int(tgt.owner) == -1:
            bonus *= NEUTRAL_BONUS_WEIGHT
            if int(world.step) < NEUTRAL_EARLY_HORIZON:
                bonus *= NEUTRAL_EARLY_EXTRA
        if LEADER_FOCUS_WEIGHT != 1.0:
            leader = _leader_owner_from_world(world, me)
            if leader is not None and int(tgt.owner) == int(leader):
                bonus *= LEADER_FOCUS_WEIGHT
        delta *= bonus

    # Follow-on hold bonus (Fix 2b — opt-in, env-gated). Surfaces the
    # B3/B4 modeling-correct snipe helpers (`_best_followon` predicts
    # follow-on + target positions at our arrival). Off by default
    # (`BASELINE_FOLLOWON_BONUS=0.0`); the bundle wrapper opts in once
    # the A/B confirms lift. Restricted to fresh captures
    # (`tgt.owner != me`) and positive-delta candidates so the bonus
    # only sweetens already-attractive captures.
    if (FOLLOWON_BONUS_WEIGHT > 0.0 and delta > 0.0
            and int(tgt.owner) != me):
        try:
            from lib.missions.snipe import _best_followon  # local: heavy import
            foothold = _best_followon(
                tgt, world, model, me, FOLLOWON_RADIUS,
                arrival_eta=int(eta),
            )
        except Exception:
            foothold = None
        if foothold is not None:
            _f_target, _f_cost, _f_eta_from_t, f_hold = foothold
            delta += (
                FOLLOWON_BONUS_WEIGHT
                * float(_f_target.production)
                * float(f_hold)
            )

    # Expansion credit (2026-06-01, default OFF) — see EXPAND_CREDIT_WEIGHT.
    # Add the held-production value of a FRESH capture additively and
    # UN-gated, so the rollout's structural passive-self pessimism can no
    # longer zero out expansion. Confirm the candidate actually flips the
    # target to us at eta (predict_garrison_at, same single-tick math the v2
    # scorer used); under-sized launches that don't capture get no credit
    # (the rollout already penalises them). Added BEFORE the PV_ETA discount
    # so it is correctly pulled back to step 0 by γ^(wait_N+eta).
    if EXPAND_CREDIT_WEIGHT > 0.0 and int(wait_N) == 0 and int(tgt.owner) != me:
        our_arrival = (int(eta), int(me), int(ships))
        cap_owner, _cap_g = predict_garrison_at(tgt, int(eta), [our_arrival])
        if cap_owner == me:
            time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - int(eta))
            held = time_remaining
            if int(tgt.id) in world.comet_ids:
                life = comet_remaining_lifetime(int(tgt.id), world)
                if life is not None:
                    held = min(held, max(0, life - int(eta)))
            cap_credit = (CAPTURE_REWARD_WEIGHT * float(tgt.production)
                          * float(held) * EXPAND_CREDIT_WEIGHT)
            # Reuse the neutral early-grab tilt so the credit prefers the
            # same targets the (gated) NEUTRAL_BONUS would have.
            if NEUTRAL_BONUS_WEIGHT != 1.0 and int(tgt.owner) == -1:
                cap_credit *= NEUTRAL_BONUS_WEIGHT
                if int(world.step) < NEUTRAL_EARLY_HORIZON:
                    cap_credit *= NEUTRAL_EARLY_EXTRA
            delta += cap_credit

    if SHIP_TURN_KAPPA > 0.0:
        delta -= SHIP_TURN_KAPPA * float(ships) * float(int(wait_N) + int(eta))

    # PV time-discount (2026-05-28). Pulls the candidate's final Δ back
    # to the current step at the already-active γ — captures the fact
    # that a fleet arriving in `wait_N + eta` turns only starts producing
    # for us at that time, so its value at step 0 is γ^(wait_N+eta) ×
    # value-at-arrival. Default OFF; applied LAST so it discounts the
    # whole Δ together (additive FOLLOWON, multiplicative NEUTRAL/LEADER,
    # and the SHIP_TURN penalty itself if enabled).
    if PV_ETA_ENABLED and (int(wait_N) + int(eta)) > 0:
        delta *= gamma ** (int(wait_N) + int(eta))

    return (delta, "scored", eta)


def score_candidate_v4_joint(snap_base, launches, me: int, num_seats: int,
                              world,
                              baseline_favors: list[float],
                              favor_fn, gamma: float,
                              horizon: int,
                              skip_admissibility: bool = False,
                              hard_deadline: float | None = None,
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
    # Collect per-leg eta for the ship-turn penalty + PV_ETA discount
    # (skip/wait>0 legs → 0, the documented v1 simplification).
    leg_etas: list[int] = []
    for src, tgt, ships, angle, wait_N in launches:
        if skip_admissibility or int(wait_N) != 0:
            leg_etas.append(0)
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
        leg_etas.append(int(fate.step))

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

    _last_step_s = 0.0
    for t in range(horizon):
        if snap.fake_env.done:
            break
        # Predictive deadline guard (see score_candidate_v4): bail before a
        # step projected to cross the cap, not after it overshot.
        if hard_deadline is not None and \
                time.perf_counter() + _last_step_s > hard_deadline:
            return (HARDCAP_BAIL_SENTINEL, "hardcap_bail")
        _step_t0 = time.perf_counter()
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t in inject_at:
            base_actions = list(inject_at[t])
            if t == earliest_inject_t and me_defense_emits:
                base_actions = base_actions + list(me_defense_emits)
            actions[me] = base_actions
        elif _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)
        _last_step_s = time.perf_counter() - _step_t0

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    delta = leaf - baseline_favors[horizon]

    # NEUTRAL_BONUS / LEADER_FOCUS for joints: apply when EVERY leg
    # targets the preferred owner. This keeps the joint Δ unitary
    # without per-leg attribution (the joint EV is one shared rollout).
    if delta > 0.0 and launches:
        bonus = 1.0
        if NEUTRAL_BONUS_WEIGHT != 1.0:
            if all(int(L[1].owner) == -1 for L in launches):
                bonus *= NEUTRAL_BONUS_WEIGHT
                if int(world.step) < NEUTRAL_EARLY_HORIZON:
                    bonus *= NEUTRAL_EARLY_EXTRA
        if LEADER_FOCUS_WEIGHT != 1.0:
            leader = _leader_owner_from_world(world, me)
            if (leader is not None
                    and all(int(L[1].owner) == int(leader) for L in launches)):
                bonus *= LEADER_FOCUS_WEIGHT
        delta *= bonus

    if SHIP_TURN_KAPPA > 0.0:
        penalty = 0.0
        for (src, tgt, ships, angle, wait_N), eta_leg in zip(launches, leg_etas):
            penalty += float(ships) * float(int(wait_N) + int(eta_leg))
        delta -= SHIP_TURN_KAPPA * penalty

    # PV time-discount (2026-05-28). For joints, use max(wait_N+leg_eta)
    # over legs — the coalition's effective payoff is gated by the
    # slowest arrival. leg_etas defaults to 0 for wait_N>0 legs (line
    # 605), which is the documented v1 simplification — multi-wait
    # joints aren't enumerated by the proposer.
    if PV_ETA_ENABLED and launches:
        max_arrival = max(
            int(wn) + int(le)
            for (_, _, _, _, wn), le in zip(launches, leg_etas)
        )
        if max_arrival > 0:
            delta *= gamma ** max_arrival

    return (delta, "scored")


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


def _solve_sync_wait(src, tgt, ships: int, omega: float, world,
                     target_arrival: int):
    """Smallest wait_N (≥0) such that the closer leg, launched after
    waiting, ARRIVES exactly on `target_arrival` (so it stacks with the
    far leg that fired now). Returns `(wait_N, angle)` or None if no
    integer-tick match converges within 3 fixpoint steps.

    Waiting shifts the geometry (orbital planets move), so eta at the
    waited fire time differs from the fire-now eta — we re-aim at each
    step and adjust the wait by the arrival error. None means this pair
    cannot be cleanly synchronized (e.g. fast orbital drift); the caller
    skips it (conservative — the rollout never sees a mis-synced stack).
    """
    _a0, eta0 = _aim_and_eta(src, tgt, ships, omega, wait_N=0, world=world)
    w = int(target_arrival) - int(eta0)
    if w < 0:
        return None
    for _ in range(3):
        angle, eta_w = _aim_and_eta(src, tgt, ships, omega, wait_N=w, world=world)
        arrival = w + int(eta_w)
        if arrival == int(target_arrival):
            return w, float(angle)
        w += int(target_arrival) - arrival
        if w < 0:
            return None
    return None


def choose_trajectory(snap_base, prerank, baseline_favors,
                      me: int, num_seats: int, wallclock_ms: float,
                      min_horizon: int, max_horizon: int, gamma: float,
                      world, model,
                      reserved_srcs: set[int] | None = None,
                      reserved_for_new_commits: set[int] | None = None,
                      agent_deadline: float | None = None,
                      ) -> tuple[list[list], list[dict]]:
    """Drop-in alternative to `chooser.choose`.

    Returns `(moves, commits)`:
      `moves`   — fire-now action list `[[src_id, angle, ships], ...]`
                  to emit this turn.
      `commits` — `wait_N > 0` winners that the agent should remember
                  across turns. Each is a dict with keys `src_id`,
                  `tgt_id`, `ships_planned`, `angle_original`,
                  `wait_remaining`, `commit_step`. The agent's ledger
                  (`agents/baseline/main._PENDING_LAUNCHES`) ticks these
                  down and fires them when `wait_remaining` reaches 0
                  (gated on `BASELINE_LEDGER=on`). When the ledger is
                  off, commits are discarded — behaviour identical to
                  the pre-ledger chooser.

    `reserved_srcs` — set of source ids that the chooser should not
    fire-now-emit from this turn (ledger is firing them via due_moves,
    or hard-ledger blocks them entirely while a commit is in flight).
    `reserved_for_new_commits` — set of source ids that already have a
    surviving ledger entry. The chooser must not add a SECOND wait
    commit for these (stacking causes duplicate emits at fire time).
    When `None`, defaults to `reserved_srcs` (hard semantics).

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
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

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
    # Hard cap: forced bail INSIDE the rollout. Mirrors chooser.choose;
    # protects against fat-tail candidates whose per_cand_ms estimate is
    # wrong. The agent-level deadline (if passed) overrides the internal
    # cap — it accounts for pre/post-chooser overhead the chooser can't see.
    hard_deadline = time.perf_counter() + WALLCLOCK_HARD_CAP_MS / 1000.0
    if agent_deadline is not None:
        hard_deadline = min(hard_deadline, agent_deadline)
        safe_deadline = min(safe_deadline, agent_deadline - per_cand_ms / 1000.0)

    scored: list[tuple] = []
    solo_winners: set[int] = set()  # src_ids whose solo scored Δ>0
    cand_count = 0

    # Value-head additive term (B.3 / Reframe B.2). Per-candidate
    # LightGBM regressor predicting K=10 ship-delta; output is added
    # as λ_vh · head_output. Featurization runs once over the prerank;
    # per-candidate prediction happens inside the loop below because
    # leaf_delta (the score itself) is one of the 15 features.
    # At λ=0 (default), vh_is_enabled() is False → vh_feats stays {}
    # and the loop's `if vh_feats:` is skipped — byte-equivalent to
    # the un-ported chooser.
    # NOTE: vh_feats is keyed by SOLO prerank rows only. Joint paths
    # at score_candidate_v4_joint(...) build their own per-call
    # feats_map at the joint-loop call site (see Phase A patch
    # 2026-06-03), so joint and solo candidates both receive the
    # VH boost when BASELINE_VH_LAMBDA > 0.
    from agents.baseline._value_head import vh_is_enabled, vh_featurize_prerank  # noqa: E501
    vh_feats: dict = {}
    if vh_is_enabled():
        try:
            vh_feats = vh_featurize_prerank(prerank, world, model)
        except Exception:
            vh_feats = {}

    for cheap_delta, src, tgt, ships, angle, eta_hint, prop_horizon, wait_N in prerank:
        if cand_count >= cap:
            break
        if not use_v3 and time.perf_counter() > safe_deadline:
            break
        # Skip candidates the ledger has already accounted for. A
        # wait_N>0 candidate from a src with a surviving commit would
        # stack a second commit — duplicate emit at fire time. A
        # wait_N==0 candidate from a reserved src would conflict with
        # the ledger's fire-now this turn (hard mode) or has no impact
        # in soft mode (where reserved_srcs only includes srcs firing
        # this turn).
        sid_ = int(src.id)
        if int(wait_N) > 0:
            if sid_ in reserved_for_new_commits:
                continue
        else:
            if sid_ in reserved_srcs:
                continue
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
                eta_hint=int(eta_hint),
                model=model,
                hard_deadline=hard_deadline,
            )
            # Value-head additive term (Block B, solo path). Looks up
            # cached features by candidate-key and adds λ_vh · head_output
            # to `score`. At λ=0 vh_feats is empty → branch is skipped.
            if vh_feats:
                from agents.baseline._value_head import vh_get_bias, vh_get_lambda, vh_predict_one  # noqa: E501
                head_out = vh_predict_one(
                    vh_feats, int(src.id), int(tgt.id),
                    int(ships), float(angle), int(wait_N),
                    float(score),  # leaf_delta input
                )
                # Subtract BIAS so the boost is mean-zero (shipped VH
                # is trained on accepted moves; head_out has +102 bias
                # that swamps base score at λ=1.0). At BIAS=0.0 (default)
                # this is byte-equivalent to the pre-fix behavior.
                score = score + vh_get_lambda() * (head_out - vh_get_bias())
            passes = (
                score > MIN_DELTA if MIN_DELTA == 0.0
                else score >= MIN_DELTA
            )
            if status == "scored" and passes:
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
    # 2026-05-21: 4P gate lifted when BASELINE_JOINT_AGGR=1 OR when the
    # explicit BASELINE_JOINT_4P=1 env var is set. Without this, AGGR's
    # `used_tgts` lift creates a silent double-count in 4P: solo emits
    # can stack on the same target but each is scored in an independent
    # rollout that assumed it was alone. Lifting the gate runs the real
    # joint scoring so combined-EV is computed once. Defensive fallout
    # (the 2026-05-18 audit's concern) is now handled by the reinforce
    # post-pass in `agents/baseline/main.emit_threat_reinforcements`.
    joint_4p_allowed = (
        JOINT_LIFT_USED_TGTS
        or os.environ.get("BASELINE_JOINT_4P", "0").strip() == "1"
    )
    joint_enabled = (
        os.environ.get("BASELINE_JOINT", "0").strip() == "1"
        and (int(num_seats) <= 2 or joint_4p_allowed)
    )
    if (joint_enabled and not use_v3
            and time.perf_counter() <= safe_deadline):
        # Group prerank by target_id. Take top-K solo candidates per
        # target by cheap_delta; pair-enumerate.
        by_tgt: dict[int, list] = {}
        for cd, src, tgt, ships, angle, eta_hint, ph, wn in prerank:
            if int(wn) != 0:
                continue  # v1: fire-now joints only
            if int(src.id) in reserved_srcs:
                continue  # ledger is firing from this src this turn
            by_tgt.setdefault(int(tgt.id), []).append(
                (float(cd), src, tgt, int(ships), float(angle),
                 int(eta_hint), int(ph)),
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
                    leg_etas = (int(ca[5]), int(cb[5]))
                    jh = max(int(ca[6]), int(cb[6]))
                    j_score, j_status = score_candidate_v4_joint(
                        snap_base, launches, me, num_seats, world,
                        baseline_favors, favor_fn, gamma,
                        horizon=jh, skip_admissibility=skip_filter,
                        hard_deadline=hard_deadline,
                    )
                    joint_count += 1
                    # Phase A (2026-06-03): VH correction on joint legs.
                    # Pre-fix, joints emitted j_score without VH boost
                    # because vh_feats was keyed by SOLO prerank rows
                    # only — joint legs miss the cache entirely. Build
                    # a per-call feats_map for the legs (with the
                    # correct per-leg eta_hint) and apply
                    # λ * sum(per-leg head_outs) to j_score so joint
                    # and solo candidates get parity treatment.
                    # Diagnostic gate (2026-06-03): BASELINE_VH_JOINT=0
                    # reverts to the pre-Phase-A behavior (solo VH only,
                    # joint legs get no VH boost) — used to A/B whether
                    # the joint aggregation is the source of the VH
                    # collapse on state-driven-K base.
                    if (vh_feats and os.environ.get(
                            "BASELINE_VH_JOINT", "1").strip() == "1"):
                        from agents.baseline._value_head import vh_get_bias, vh_get_lambda, vh_predict_one  # noqa: E501
                        joint_prerank = [
                            (0.0, launches[k][0], launches[k][1],
                             launches[k][2], launches[k][3],
                             leg_etas[k], jh, launches[k][4])
                            for k in (0, 1)
                        ]
                        joint_feats = vh_featurize_prerank(
                            joint_prerank, world, model,
                        )
                        bias = vh_get_bias()
                        leg_sum = 0.0
                        for ln in launches:
                            leg_sum += vh_predict_one(
                                joint_feats, int(ln[0].id), int(ln[1].id),
                                int(ln[2]), float(ln[3]), int(ln[4]),
                                float(j_score),
                            ) - bias
                        j_score = j_score + vh_get_lambda() * leg_sum
                    if j_status == "scored" and j_score > 0.0:
                        scored.append((j_score, "joint", launches))

    # Synchronized-arrival JOINT coalitions (BASELINE_JOINT_SYNC). Separate,
    # isolated from the fire-now loop above so the OFF path is untouched.
    # For a capture target NO single source can take alone, make the closer
    # source WAIT so all legs land the same tick and STACK. Gate re-read at
    # call time (not the module constant) so the env toggle works regardless
    # of import order in tests / A/B subprocesses.
    if (os.environ.get("BASELINE_JOINT_SYNC", "0").strip() == "1"
            and joint_enabled and not use_v3
            and time.perf_counter() <= safe_deadline):
        # Build synchronized two-source coalitions DIRECTLY from world geometry
        # — NOT from `prerank`, which cheap-rejects the bouncing single-source
        # launches a coalition combines (proposer CHEAP_REJECT_THRESHOLD drops
        # any ~20+ ship bounce before the chooser sees it). For each defended
        # planet no single source can take, pull the 2 nearest friendly sources
        # and assemble a MINIMALLY-sized stacked strike: minimal ships-in-flight
        # gives the best chance of clearing the value head's in-flight penalty.
        my_sources = [
            p for p in world.planets_by_id.values()
            if int(p.owner) == me
            and int(p.id) not in reserved_srcs
            and int(p.id) not in reserved_for_new_commits
            and int(p.ships) >= MIN_FLEET_SIZE
        ]
        sync_targets = [
            p for p in world.planets_by_id.values() if int(p.owner) != me
        ]
        # Spend the bounded pair budget on the most valuable captures first.
        sync_targets.sort(key=lambda t: (-int(t.production), int(t.id)))
        # Effective arrival ceiling. BOTH legs land on the synchronized tick
        # `tarr`; if `tarr` exceeds the launch-rules capture-horizon K, the
        # fire-now far leg is silently deleted by enforce_launch_rules
        # (launch_rules.py:163), leaving the near leg to fire alone and bounce
        # — a half-fire that wastes both fleets. So when launch rules are on,
        # only form coalitions whose arrival is within K (Rule 40: align the
        # generator with the downstream filter, don't loosen the K cap).
        sync_arrival_cap = max_horizon
        _per_target_sync_cap = False
        if _launch_rules_enabled():
            sync_arrival_cap = min(
                max_horizon, _capture_horizon_k(getattr(world, "step", None))
            )
            # State-driven lever: the cap is the predictability of the
            # coalition's own target (computed per tgt0 at Gate 2b below).
            _per_target_sync_cap = _state_driven_k_enabled()
        sync_count = 0
        for tgt0 in sync_targets:
            if (sync_count >= JOINT_SYNC_MAX_PAIRS
                    or time.perf_counter() > safe_deadline):
                break
            if len(my_sources) < 2:
                break
            cand_srcs = _nearest_k(my_sources, tgt0, JOINT_SYNC_SRC_K)
            if len(cand_srcs) < 2:
                continue
            s_near, s_far = cand_srcs[0], cand_srcs[1]  # distance-sorted
            # Solo-capture skip: if EITHER nearest source can take tgt0 alone,
            # the solo / fire-now-joint path owns it — sync is strictly for the
            # "neither alone can take it" regime.
            if (int(s_near.ships)
                    >= _capture_size(s_near, tgt0, model, world.omega, me, world)
                    or int(s_far.ships)
                    >= _capture_size(s_far, tgt0, model, world.omega, me, world)):
                continue
            arrivals = model.ledger.get(int(tgt0.id), [])
            # Provisional arrival from full-ship etas → garrison there → `need`.
            _an, eta_near_full = _aim_and_eta(
                s_near, tgt0, int(s_near.ships), world.omega, wait_N=0, world=world)
            _af, eta_far_full = _aim_and_eta(
                s_far, tgt0, int(s_far.ships), world.omega, wait_N=0, world=world)
            provisional_arrival_step = max(int(eta_near_full), int(eta_far_full))
            _oTp, garr_Tp = predict_garrison_at(
                tgt0, provisional_arrival_step, arrivals)
            capture_need = int(garr_Tp) + 1  # smallest int strictly above garrison
            # Lever 1 (size-to-hold): when BASELINE_JOINT_SYNC_HOLD is on, size
            # the stack to SURVIVE the predicted opp counter-recapture, not just
            # to flip the planet. Inverts the _target_holdable_after_capture
            # inequality via hold_need (same counter model — Rule 40). Read at
            # call-time like BASELINE_JOINT_SYNC above, not a module constant.
            sync_hold = (
                os.environ.get("BASELINE_JOINT_SYNC_HOLD", "0").strip() == "1"
            )
            if sync_hold:
                need = hold_need(
                    tgt0, provisional_arrival_step, world, me, capture_need)
            else:
                need = capture_need
            # Minimal sizing: far leg commits ships now, near leg tops up.
            far_ships = min(int(s_far.ships), need)
            near_ships = min(int(s_near.ships), need - far_ships)
            if far_ships + near_ships < need:
                # far too small to lead — load the larger (near) budget first.
                near_ships = min(int(s_near.ships), need)
                far_ships = min(int(s_far.ships), need - near_ships)
            if far_ships + near_ships < need:
                continue  # combined budget can't take tgt0
            if far_ships < MIN_FLEET_SIZE or near_ships < MIN_FLEET_SIZE:
                continue
            # Re-derive arrival from the ASSIGNED counts (fleet speed rises with
            # ship count, so a smaller leg flies slower → arrives later). The
            # leg with the larger eta fires now; the other waits to match it.
            angle_n, eta_n = _aim_and_eta(
                s_near, tgt0, near_ships, world.omega, wait_N=0, world=world)
            angle_f, eta_f = _aim_and_eta(
                s_far, tgt0, far_ships, world.omega, wait_N=0, world=world)
            eta_n, eta_f = int(eta_n), int(eta_f)
            if eta_n == eta_f:
                continue  # already synchronized → the fire-now joint owns it
            if eta_f >= eta_n:
                fire_pl, fire_ships, fire_angle = s_far, far_ships, angle_f
                wait_pl, wait_ships, wait_eta = s_near, near_ships, eta_n
                tarr = eta_f
            else:
                fire_pl, fire_ships, fire_angle = s_near, near_ships, angle_n
                wait_pl, wait_ships, wait_eta = s_far, far_ships, eta_f
                tarr = eta_n
            # Gate 2 (Rule 47): the leaf must see the stacked capture settle.
            if tarr + JOINT_SYNC_SETTLE > max_horizon:
                continue
            # Gate 2b: both legs must arrive within the launch-rules ceiling K,
            # else the fire-now far leg is deleted downstream and the coalition
            # half-fires (see sync_arrival_cap above).
            _cap = sync_arrival_cap
            if _per_target_sync_cap:
                _cap = min(max_horizon, _capture_horizon_k(
                    getattr(world, "step", None), tgt_id=int(tgt0.id),
                    world=world, model=model, me=me))
            if tarr > _cap:
                continue
            _oT, garr_tarr = predict_garrison_at(tgt0, tarr, arrivals)
            # Gate 3: combined force must actually exceed garrison at arrival.
            if fire_ships + wait_ships <= garr_tarr:
                continue
            # Gate 1 (neither solo, post-sizing): neither leg alone may capture
            # at its own arrival tick. SKIPPED when SYNC_HOLD inflates per-leg
            # counts purely to survive the counter (not to solo-capture): the
            # regime is already enforced PRE-sizing at the _capture_size gate
            # above, and comparing hold-inflated counts here would spuriously
            # drop wanted coalitions (an inflated leg ~always exceeds garrison).
            _ow, garr_wait = predict_garrison_at(tgt0, wait_eta, arrivals)
            if not sync_hold and (fire_ships > garr_tarr or wait_ships > garr_wait):
                continue
            # The closer (waiting) leg waits so it arrives exactly at tarr.
            solved = _solve_sync_wait(
                wait_pl, tgt0, wait_ships, world.omega, world,
                target_arrival=tarr,
            )
            if solved is None:
                continue  # can't cleanly synchronize this pair
            wait_n, angle_wait = solved
            launches = [
                (fire_pl, tgt0, fire_ships, fire_angle, 0),
                (wait_pl, tgt0, wait_ships, angle_wait, int(wait_n)),
            ]
            jh = min(tarr + JOINT_SYNC_SETTLE, len(baseline_favors) - 1)
            j_score, j_status = score_candidate_v4_joint(
                snap_base, launches, me, num_seats, world,
                baseline_favors, favor_fn, gamma,
                horizon=jh, skip_admissibility=skip_filter,
                hard_deadline=hard_deadline,
            )
            sync_count += 1
            if j_status == "scored" and j_score > 0.0:
                scored.append((j_score, "joint_sync", launches))

    if not scored:
        return [], []

    scored.sort(key=lambda c: -c[0])

    # Phase H — value-head rerank (Option C, 2026-06-03). When
    # BASELINE_VH_RERANK_K > 0, take the top-K solo entries (in the
    # current base-score-sorted order), re-rank them by vh_predict_one
    # output, and put them back in the same K slots. Base scores are
    # preserved on each tuple — only the order changes. Joints and
    # joint_sync entries are untouched. Sidesteps the magnitude-swamp
    # that closed the VH-additive axis (Rule 37 n=5); consumes the
    # model's verified Spearman ρ=+0.386 rank signal as RANK, not
    # magnitude. At BASELINE_VH_RERANK_K=0 (default) this block is
    # skipped entirely — byte-equivalent to the un-VH chooser.
    if vh_feats:
        from agents.baseline._value_head import vh_rerank_k
        K = vh_rerank_k()
        if K > 0:
            from agents.baseline._value_head import vh_predict_one
            solo_slots: list[int] = []
            for i, entry in enumerate(scored):
                if (len(entry) == 6
                        and not (isinstance(entry[1], str))):
                    solo_slots.append(i)
                    if len(solo_slots) >= K:
                        break
            if len(solo_slots) >= 2:
                solos = [scored[i] for i in solo_slots]
                vh_scores = []
                for (base_score, src, tgt, ships, angle, wait_N) in solos:
                    head_out = vh_predict_one(
                        vh_feats, int(src.id), int(tgt.id),
                        int(ships), float(angle), int(wait_N),
                        float(base_score),
                    )
                    vh_scores.append(head_out)
                order = sorted(range(len(solos)), key=lambda k: -vh_scores[k])
                for new_pos, old_pos in enumerate(order):
                    scored[solo_slots[new_pos]] = solos[old_pos]

    # Emit logic — match composite chooser (`agents/baseline/chooser.choose`)
    # for parity. 1 launch per source per turn, 1 per target. For joints
    # (tagged 'joint' tuples), require ALL of its sources and targets to
    # be free; commit all legs together.
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    commits: list[dict] = []
    # Reframe B.2 / Phase D — solo-only accepted trace. Populated only
    # when BASELINE_ACCEPTED_TRACE env var is set; trace_accepted is a
    # no-op otherwise. Solo-only by design — VH training corpus drops
    # joints (joints aren't VH-corrected by the head).
    accepted_trace: list[dict] = []
    # eta_by_key: map (src_id, tgt_id, ships, angle, wait_N) → eta_hint
    # from the prerank so the emit loop can attach eta to each accepted
    # solo record. Round angle to match _candidate_key precision in
    # _value_head._candidate_key.
    eta_by_key: dict[tuple, int] = {}
    for cheap_delta, src, tgt, ships, angle, eta_hint, prop_horizon, wait_N in prerank:
        key = (int(src.id), int(tgt.id), int(ships),
               round(float(angle), 6), int(wait_N))
        eta_by_key.setdefault(key, int(eta_hint))
    commit_step = int(world.step) if world is not None else 0
    for entry in scored:
        # Synchronized joint (3-tuple, tag 'joint_sync'): far leg fires now,
        # closer leg(s) wait so all arrive the same tick. The waiting leg is
        # surfaced as a commit carrying sync_joint=True; the self-contained
        # sync-joint tick in main.py (gated on BASELINE_JOINT_SYNC=1) fires
        # it when wait_remaining hits 0, independent of the general ledger.
        if len(entry) == 3 and entry[1] == "joint_sync":
            _score, _tag, launches = entry
            if any(int(L[0].id) in used_srcs for L in launches):
                continue
            if (not JOINT_LIFT_USED_TGTS
                    and any(int(L[1].id) in used_tgts for L in launches)):
                continue
            for src, tgt, ships, angle, wait_N in launches:
                used_srcs.add(int(src.id))
                used_tgts.add(int(tgt.id))
                if int(wait_N) == 0:
                    moves.append([int(src.id), float(angle), int(ships)])
                else:
                    commits.append({
                        "src_id": int(src.id),
                        "tgt_id": int(tgt.id),
                        "ships_planned": int(ships),
                        "angle_original": float(angle),
                        "wait_remaining": int(wait_N),
                        "commit_step": commit_step,
                        "sync_joint": True,
                    })
            continue
        # Joint candidates are 3-tuples: (score, 'joint', launches).
        if len(entry) == 3 and entry[1] == "joint":
            _score, _tag, launches = entry
            if any(int(L[0].id) in used_srcs for L in launches):
                continue
            if (not JOINT_LIFT_USED_TGTS
                    and any(int(L[1].id) in used_tgts for L in launches)):
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
        if sid in used_srcs:
            continue
        if not JOINT_LIFT_USED_TGTS and tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        else:
            # Wait-N winner — emit nothing this turn; instead surface
            # as a commit. The agent's ledger (when BASELINE_LEDGER=on)
            # will tick this down and fire at wait_N == 0.
            commits.append({
                "src_id": sid,
                "tgt_id": tid,
                "ships_planned": int(ships),
                "angle_original": float(angle),
                "wait_remaining": int(wait_N),
                "commit_step": commit_step,
            })
        key = (sid, tid, int(ships), round(float(angle), 6), int(wait_N))
        accepted_trace.append({
            "kind": "solo",
            "src_id": sid,
            "tgt_id": tid,
            "ships": int(ships),
            "angle": float(angle),
            "wait_N": int(wait_N),
            "eta": int(eta_by_key.get(key, 0)),
            "delta_pred": float(_score),
        })
    if accepted_trace:
        from agents.baseline._trace_hook import trace_accepted
        trace_accepted(world, model, me, accepted_trace)
    return moves, commits
