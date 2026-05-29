"""Opponent move predictor — pluggable policy for forward-sim rollouts.

Why we need this layer: the lookahead inner loop (`fast_sim.rollout`)
needs a `policy_per_seat[i]` callable for every non-pov seat. Phase 2
used "the opponent plays our own v2/v3_snipe pipeline" as the rollout
policy (audit/2026-05-11-lookahead-phase2-forward-sim.md:140-146); the
v3_lookahead MVP did the same (audit/2026-05-11-v3-lookahead-mvp-
parity.md). That's serviceable but biases the rollout toward what WE
would do, not what the ladder opponent will do.

This module exposes three tiers behind one signature:

    policy_p1 = make_opp_policy(my_id=0, tier=1)
    rollout(snap, K=30, policies=[my_policy, policy_p1])

- **Tier 0** — `mirror_self_policy`: identical to v3_snipe (aggressive
  off). The original Phase 2 default. Use when we want bit-exact
  parity with the v3_lookahead MVP, or when we want the rollout policy
  to be cheap and well-understood.
- **Tier 1** — `top_tier_mirror_policy`: same pipeline but with
  AGGRESSIVE snipe sizing on (matches v3.5.1 — and matches the top-10
  fingerprint per knowledge-base/concepts/top-performer-strategies.md:
  mean fleet 38 vs midpack 29 / mean garrison-at-launch 10.6 vs 22 /
  enemy-target fraction 0.32 vs 0.14). Better proxy for the ≥μ1100
  ladder population we currently lose to.
- **Tier 2** — placeholder; will be a small logistic regression on the
  37k labeled shots in `data/shot_validator/`. NOT trained here;
  ships in a follow-up session. The function signature is reserved so
  consumers can switch over without rewriting.

All three tiers are PURE FUNCTIONS of the opponent's observation —
they don't read any global state and don't carry per-episode memory.
That's intentional: forward-sim rollouts call them at every tick on a
synthetic Snapshot, so any per-episode state would corrupt the search.

The Tier-1 "priors" are constants lifted from the cross-replay
fingerprint analysis. We do not add new priors here — they belong in
the knowledge-base.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

from lib.fleet import speed as _fleet_speed
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


Policy = Callable[[Any], list]

# Global per-tick launch budget for `lite_greedy_policy` (2026-05-28 PM4).
# The current policy lets every owned planet emit a 0.7x ships launch every
# tick. With ~5 owned planets x eta=10 rollout, that's ~50 simulated launches
# per candidate, vs realised top-mu ladder rate ~1.3 launches/turn globally
# (knowledge-base/concepts/top-performer-strategies.md). At K>0 the policy
# keeps only the top-K candidates by ROI (`prod/(d+1)`) per call; K=0
# (default) preserves byte-for-byte legacy. Calibration variants K=1/2/3.
# Intended for OPP seats; do not combine with BASELINE_ME_REACTS=1 (would
# throttle ME-side reactive launches too).
OPP_MAX_LAUNCHES: int = int(
    os.environ.get("BASELINE_OPP_MAX_LAUNCHES", "0")
)


# MLP-as-opp-model threshold (2026-05-29). Used by `mlp_validated_policy`
# to filter `lite_greedy_policy`'s candidate emits through the trained
# 3-MLP shot validator. The MLP returns `P(launch will succeed) in [0, 1]`
# from the opponent's seat; only candidates with P >= threshold are
# emitted. Threshold default 0.5 (high-precision, opp fires only when
# confident); raise to suppress more, lower to emit more. The validator's
# self-side rejection threshold (0.30) was tuned for high recall on
# OUR-side filtering and is the wrong knob for OPP-model use.
OPP_MLP_THRESHOLD: float = float(
    os.environ.get("BASELINE_OPP_MLP_THRESHOLD", "0.5")
)


# ---------------------------------------------------------------------------
# Tier 0 — mirror self (= v3_snipe pipeline, aggressive sizing OFF)
# ---------------------------------------------------------------------------


def mirror_self_policy(obs: Any) -> list:
    """Run the v3_snipe pipeline against `obs`. Bit-exact equivalent of
    `agents/v3_snipe/main.py`'s `agent(obs)` body.

    Drop-in replacement for `lib.lookahead.score_action`'s `policy`
    argument when you want the Phase 2 default ("opponent plays v2").

    Phase 3c: reuses `_shared_world_model` from the obs if present
    (saves ~3.8 ms WorldModel rebuild when score_candidate has already
    computed it for the OTHER seat at the same step).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    cached = getattr(obs, "_shared_world_model", None)
    model = cached if cached is not None else WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=False)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


# ---------------------------------------------------------------------------
# Tier 1 — top-tier mirror (= v3.5.1 pipeline, aggressive sizing ON)
# ---------------------------------------------------------------------------


def top_tier_mirror_policy(obs: Any) -> list:
    """Run the v3.5.1 pipeline against `obs`. Mirrors `agents/v3.5.1/main.py`.

    Why this and not Tier 0: top-10 fingerprints show mean fleet 38 vs
    midpack 29 and mean garrison-at-launch 10.6 vs 22 (knowledge-base/
    concepts/top-performer-strategies.md:171-184). The single behavioural
    change that captures most of that gap is `aggressive=True` in the
    snipe builder — it sizes launches as a fraction of source garrison
    (0.7) rather than minimum-viable target.ships+1.

    Use Tier 1 as the rollout policy when modelling opponents above the
    μ≈1100 band; use Tier 0 for parity with prior probes / lower-ladder
    self-play.

    Phase 3c: reuses `_shared_world_model` from the obs if present
    (saves ~3.8 ms WorldModel rebuild when score_candidate has already
    computed it for the OTHER seat at the same step).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    cached = getattr(obs, "_shared_world_model", None)
    model = cached if cached is not None else WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


# ---------------------------------------------------------------------------
# Tier 2 — placeholder for the trained launch-decision classifier
# ---------------------------------------------------------------------------


def trained_logreg_policy(obs: Any) -> list:
    """Reserved for the trained launch-decision classifier.

    Will read a model artifact (≤200-float logistic regression weights)
    from a fixed path under the submission bundle, score each candidate
    mission with the 24-dim feature schema at
    `data/shot_validator/schema.json`, and emit the argmax-launch set.

    Not implemented in this branch — see plan section "Deliverable 2 /
    Tier 2". Fallback to Tier 1 so downstream consumers can wire it up
    without crashing.
    """
    return top_tier_mirror_policy(obs)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


_TIER_REGISTRY: dict[int, Policy] = {
    0: mirror_self_policy,
    1: top_tier_mirror_policy,
    2: trained_logreg_policy,
}


def lite_greedy_policy(obs: Any) -> list:
    """Cheap opp policy: ROI-greedy launch picker, no WorldModel.

    Per-call cost is ~1-2 ms (raw obs only; no World object,
    no WorldModel.from_world, no mission framework, no mechanism stack).
    The mirror policies (tier 0, 1) take ~10 ms because they rebuild
    the WorldModel timeline every step. Use this when wallclock budget
    matters more than bit-identical top-tier behaviour (e.g. as the
    per-step opp policy in lookahead rollouts).

    Behaviour: for each owned planet with ships >= 5, find the
    enemy/neutral target with the best production/distance ratio and
    launch enough to win the capture, sized by max(aggressive=0.7×src,
    capture_size). Skips if the source can't afford the capture
    (defenders+production_during_flight+1 > src.ships), avoiding the
    bouncing-fleet failure mode where 0.7×src.ships < defenders.
    """
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return []
    targets = [p for p in planets if p[1] != player]
    moves: list = []
    # Top-K mode: collect (-roi, seq_idx, move) for stable descending-ROI
    # sort with planet-walk tie-break. seq_idx tracks emission order so
    # ties resolve deterministically. K<=0 (default) skips the entire
    # candidates path and appends directly to `moves` for byte-parity.
    k_cap = OPP_MAX_LAUNCHES
    candidates: list[tuple[float, int, list]] = []
    seq_idx = 0
    for src in planets:
        if src[1] != player or src[5] < 10:
            continue
        best = None
        best_score = -1.0
        sx = src[2]; sy = src[3]
        for t in targets:
            if t[0] == src[0]:
                continue
            dx = t[2] - sx; dy = t[3] - sy
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                continue
            score = float(t[6]) / (d + 1.0)
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            continue
        # Capture-size estimate: predict defenders at straight-line ETA
        # for an aggressive-sized fleet, only launch if affordable.
        # Straight-line aim/eta — adequate for static targets; orbital
        # targets misaim but the rollout simulator catches the miss.
        budget = int(src[5])
        agg_ships = max(5, int(budget * 0.7))
        if agg_ships > budget:
            agg_ships = budget
        spd = _fleet_speed(agg_ships)
        if spd <= 0:
            continue
        dx = best[2] - sx; dy = best[3] - sy
        d = math.sqrt(dx * dx + dy * dy)
        flight = max(0.0, d - float(src[4]) - float(best[4]) - 0.1)
        eta = max(1, int(math.ceil(flight / spd)))
        # Production accrues only for OWNED planets (env rule:
        # orbit_wars.py:511-514 — neutrals stay at their current count).
        # Treating neutrals as accreting was the bug that made lite_greedy
        # skip capturable openings (e.g. 13-defender prod=1 neutral at
        # d=12 looked like 19 defenders at eta=6, so the policy idled
        # in opp_traj rollouts). Real opps grab near targets at step 4
        # and snowball.
        if int(best[1]) == -1:
            defenders_at_eta = float(best[5])
        else:
            defenders_at_eta = float(best[5]) + float(best[6]) * eta
        needed = int(math.ceil(defenders_at_eta)) + 1
        if needed > budget:
            continue  # can't afford the capture — skip, don't bounce
        ships = max(agg_ships, needed)
        if ships > budget:
            ships = budget
        if ships < 5:
            continue
        angle = math.atan2(best[3] - sy, best[2] - sx)
        move = [src[0], angle, ships]
        if k_cap <= 0:
            moves.append(move)
        else:
            candidates.append((-best_score, seq_idx, move))
            seq_idx += 1
    if k_cap > 0 and candidates:
        candidates.sort()
        moves = [c[2] for c in candidates[:k_cap]]
    return moves


def nearest_opp_policy(obs: Any) -> list:
    """Cheap opp policy: nearest-target launch picker.

    Same scaffolding as `lite_greedy_policy` (ships >= 10 source gate,
    capture-size estimate via straight-line ETA + defender prediction,
    affordability skip, `OPP_MAX_LAUNCHES` top-K dispatch) — the only
    difference is target selection: pick the geographically closest
    non-our planet instead of the highest production/distance ratio.

    Hypothesis (2026-05-29 PM): live-ladder opps at near range act more
    like "grab what's close" than "compute ROI"; using a nearest-pick
    rollout opp may produce more predictive chooser deltas than
    lite_greedy's ROI overestimate of opp's far-target ambition.
    """
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return []
    targets = [p for p in planets if p[1] != player]
    moves: list = []
    k_cap = OPP_MAX_LAUNCHES
    candidates: list[tuple[float, int, list]] = []
    seq_idx = 0
    for src in planets:
        if src[1] != player or src[5] < 10:
            continue
        best = None
        best_score = -1.0  # max-pick on 1/(d+1) → closer = larger score
        sx = src[2]; sy = src[3]
        for t in targets:
            if t[0] == src[0]:
                continue
            dx = t[2] - sx; dy = t[3] - sy
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                continue
            score = 1.0 / (d + 1.0)  # nearest-pick: max score → min distance
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            continue
        budget = int(src[5])
        agg_ships = max(5, int(budget * 0.7))
        if agg_ships > budget:
            agg_ships = budget
        spd = _fleet_speed(agg_ships)
        if spd <= 0:
            continue
        dx = best[2] - sx; dy = best[3] - sy
        d = math.sqrt(dx * dx + dy * dy)
        flight = max(0.0, d - float(src[4]) - float(best[4]) - 0.1)
        eta = max(1, int(math.ceil(flight / spd)))
        if int(best[1]) == -1:
            defenders_at_eta = float(best[5])
        else:
            defenders_at_eta = float(best[5]) + float(best[6]) * eta
        needed = int(math.ceil(defenders_at_eta)) + 1
        if needed > budget:
            continue
        ships = max(agg_ships, needed)
        if ships > budget:
            ships = budget
        if ships < 5:
            continue
        angle = math.atan2(best[3] - sy, best[2] - sx)
        move = [src[0], angle, ships]
        if k_cap <= 0:
            moves.append(move)
        else:
            candidates.append((-best_score, seq_idx, move))
            seq_idx += 1
    if k_cap > 0 and candidates:
        candidates.sort()
        moves = [c[2] for c in candidates[:k_cap]]
    return moves


# ---------------------------------------------------------------------------
# Tier 3 (2026-05-29) — MLP-filtered lite_greedy
# ---------------------------------------------------------------------------
#
# Pivot from the falsified top-K-by-ROI rate cap (Stage-2 A/B 6/16 vs live).
# The rate-cap addressed the right symptom (lite_greedy over-fires vs the
# realised ladder fingerprint) but the wrong axis: the right knob is "fire
# only when the shot is likely to succeed", not "fire at most K per tick".
#
# This tier wraps `lite_greedy_policy`'s candidate emit list with the
# trained 3-MLP shot validator from sub 53131296. The validator scores
# `P(launch will succeed | obs, focal_seat)` from the opponent's seat
# (focal_seat = obs.player when the rollout queries the opp's observation).
# Only candidates with P >= OPP_MLP_THRESHOLD pass through. Self-reinforce
# emits (target already owned by focal) bypass the filter — same convention
# as the validator agent itself.
#
# The MLP was trained on per-shot labels from top-10 ladder replays + 10
# midpack, so the threshold sits closer to the actual ladder-realised
# launch density than any constant K cap can. Fall-through if the weights
# fail to load (degrades gracefully to lite_greedy).


def mlp_validated_policy(obs: Any) -> list:
    """Tier 3 opp policy: lite_greedy candidates filtered by trained MLP.

    Pipeline:
      1. Run `lite_greedy_policy` to propose candidate (src, angle, ships)
         emits for the focal seat. Cheap; per-tick cost matches lite_greedy.
      2. For each candidate, encode 25-d features via
         `lib.shot_features.encode_shot_features` with `focal_seat =
         obs.player` (the seat the rollout is querying — the opponent
         from the chooser's perspective).
      3. Self-reinforce candidates (target already owned by focal seat)
         bypass the filter, matching `baseline_validated`'s convention.
      4. Stack remaining feature vectors; one batched ensemble forward.
      5. Emit those with `p >= OPP_MLP_THRESHOLD`.

    If the MLP weights are missing/malformed (e.g. lib._validator_weights
    blob empty), returns `lite_greedy_policy` output unfiltered so the
    rollout still has a viable opponent.
    """
    # Single-line imports below: the bundler's per-line import-stripping
    # regex leaks continuation lines from a parenthesised multi-line
    # import as indented orphans (IndentationError at runtime). Friction
    # tag: `bundler-modular-agent-namespace-access-breaks-bundle`.
    from lib._validator_mlp import ensemble_proba, is_ready
    from lib.shot_features import FEATURE_DIM, encode_shot_features, target_owned_by

    candidates = lite_greedy_policy(obs)
    if not candidates:
        return []
    if not is_ready():
        return candidates

    focal_seat = (
        obs.get("player", 0) if isinstance(obs, dict)
        else getattr(obs, "player", 0)
    )

    survivors: list = []
    to_score: list[tuple[int, Any]] = []
    for i, emit in enumerate(candidates):
        if target_owned_by(emit, obs, focal_seat):
            survivors.append(emit)
            continue
        feats = encode_shot_features(emit, obs, focal_seat)
        if feats is None or feats.shape[0] != FEATURE_DIM:
            survivors.append(emit)
            continue
        to_score.append((i, feats))

    if to_score:
        import numpy as np
        X = np.stack([f for _, f in to_score]).astype(np.float32)
        probs = ensemble_proba(X)
        for (idx, _), p in zip(to_score, probs):
            if p >= OPP_MLP_THRESHOLD:
                survivors.append(candidates[idx])

    return survivors


# ---------------------------------------------------------------------------
# ME-side defensive policy for the chooser's rollout (bug #14 fix, 2026-05-18)
# ---------------------------------------------------------------------------
#
# The chooser's rollout in agents/baseline/chooser_trajectory.py drives
# opp seats with `lite_greedy_policy` every tick but leaves ME idle
# (except for the candidate injection at `wait_N`). This asymmetry
# under-rates candidates that look attractive but expose our sources to
# counter-attack (rollout shows opp exploiting, we don't defend) and
# over-rates captures whose leaf-state ownership wouldn't survive opp's
# counter past the rollout horizon. Catalog: audit/2026-05-18-bug-
# catalog.md#14.
#
# Option 1 — cheap mirror with `lite_greedy_policy` for ME — failed
# (commit 5f22ea8): lite_greedy is too attack-biased, so the rollout's
# defense-baseline path emitted ATTACK launches from the would-be
# reinforcer planet and the threatened planet fell anyway.
#
# Option 5 (this function): PURELY DEFENSIVE policy for ME. Scan
# inbound enemy fleets, find under-defended owned planets, emit a
# reinforce launch from the nearest viable sister planet. Never emit
# attacks. The chooser's actual attack moves are made on its own turn;
# the rollout's job is to model opp's reaction to OUR move, which
# implies us defending what opp threatens — not us attacking again.
def me_defensive_action(obs: Any, me: int) -> list:
    """Purely-defensive obs-only policy for ME in the rollout.

    Returns env-format launches [[src_id, angle, ships], ...]. Same
    call shape as `lite_greedy_policy`. Stateless. No `WorldModel`
    build (would add 3-5 ms per tick × per candidate × baseline and
    blow the wallclock budget); uses only `fleet_target_planet` +
    arithmetic.

    Algorithm:
    1. Walk obs.fleets; attribute each enemy fleet to a MY planet via
       `lib.world_model.fleet_target_planet` (bug-#11-aware ray-cast).
       Bucket into `{my_pid: [(eta, ships), ...]}`.
    2. For each threatened MY planet P, sum threat force inside the
       bug-#12 window: `threat_force = sum(s for (e, s) in inbound[P]
       if e <= earliest_eta + WAVE_LOOKAHEAD)`. Skip if natural
       production covers it (`P.ships + P.prod × earliest_eta >=
       threat_force + 1`).
    3. Find nearest viable reinforcer Q: own, not P, not in
       `used_srcs`, reinforce-eta < earliest_eta with the sized fleet.
    4. Size: `ships = max(MIN_FLEET_SIZE, ceil(shortfall) +
       SAFETY_MARGIN)`, clamped at `Q.ships`.
    5. Aim: `lib.aim.aim_orbiting` for orbiting P, else `atan2`.
    6. Emit `[int(Q.id), float(angle), int(ships)]`; mark Q used.
    """
    # Local imports to keep the module's top-level fast (this function
    # is on the rollout hot path and must not pay an import cost on
    # first call inside the rollout).
    from lib.aim import aim_orbiting
    from lib.orbit import is_orbiting
    from lib.world_model import WAVE_LOOKAHEAD, fleet_target_planet
    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        Fleet, Planet,
    )

    MIN_FLEET_SIZE = 2
    SAFETY_MARGIN = 1

    planets_raw = (
        obs.get("planets") if isinstance(obs, dict)
        else getattr(obs, "planets", None)
    )
    if not planets_raw:
        return []
    fleets_raw = (
        obs.get("fleets", []) if isinstance(obs, dict)
        else getattr(obs, "fleets", [])
    )
    if not fleets_raw:
        return []
    omega = float(
        obs.get("angular_velocity", 0.0) if isinstance(obs, dict)
        else getattr(obs, "angular_velocity", 0.0) or 0.0
    )

    planets = [Planet(*p) for p in planets_raw]
    fleets = [Fleet(*f) for f in fleets_raw]
    my_planets_by_id = {int(p.id): p for p in planets if int(p.owner) == me}
    if not my_planets_by_id:
        return []

    # 1. Attribute fleets to MY planets. Enemy fleets become threats;
    # friendly fleets become inbound reinforcements that count toward
    # `garrison_at_eta`. The friendly-counting is the critical
    # idempotency property: without it the stateless policy emits a
    # NEW reinforce every rollout tick because each tick re-evaluates
    # the SAME threat against the SAME garrison without crediting the
    # already-launched reinforce. By tick N we've stacked N redundant
    # reinforces, draining the sister and bloating the fleet count
    # (which slows fs_step). Counting friendlies makes the policy
    # converge after one emit per real threat.
    inbound_enemy: dict[int, list[tuple[int, int]]] = {}
    inbound_friendly_ships: dict[int, int] = {}
    for f in fleets:
        target, eta = fleet_target_planet(f, planets, omega)
        if target is None or int(target.owner) != me:
            continue
        if int(f.owner) == me:
            inbound_friendly_ships[int(target.id)] = (
                inbound_friendly_ships.get(int(target.id), 0)
                + int(f.ships)
            )
        else:
            inbound_enemy.setdefault(int(target.id), []).append(
                (int(eta), int(f.ships))
            )
    if not inbound_enemy:
        return []

    moves: list = []
    used_srcs: set[int] = set()

    # Process threats in eta-order so the most-urgent gets dibs on the
    # nearest reinforcer.
    threat_list = sorted(
        inbound_enemy.items(),
        key=lambda kv: min(e for (e, _s) in kv[1]),
    )
    for pid, waves in threat_list:
        p_target = my_planets_by_id[pid]
        earliest_eta = min(e for (e, _s) in waves)
        threat_force = sum(
            s for (e, s) in waves if e <= earliest_eta + WAVE_LOOKAHEAD
        )
        if threat_force <= 0:
            continue
        garrison_at_eta = (
            float(p_target.ships)
            + float(p_target.production) * float(earliest_eta)
            + float(inbound_friendly_ships.get(pid, 0))
        )
        if garrison_at_eta >= float(threat_force) + 1.0:
            continue  # natural production + in-flight reinforces cover it

        shortfall = float(threat_force) + 1.0 - garrison_at_eta
        # Find nearest viable reinforcer.
        best: tuple | None = None
        best_dist_sq = float("inf")
        for q in planets:
            if int(q.owner) != me or int(q.id) == pid:
                continue
            if int(q.id) in used_srcs:
                continue
            dx = float(p_target.x) - float(q.x)
            dy = float(p_target.y) - float(q.y)
            dist_sq = dx * dx + dy * dy
            if dist_sq >= best_dist_sq:
                continue
            # Single-iteration chicken-and-egg fix: assume worst-case
            # reinforce-eta = earliest_eta - 1, compute ships, verify
            # the resulting eta is still < earliest_eta.
            worst_eta = max(1, int(earliest_eta) - 1)
            ships_guess = max(
                MIN_FLEET_SIZE,
                int(math.ceil(shortfall)) + SAFETY_MARGIN,
            )
            if ships_guess > int(q.ships):
                ships_guess = int(q.ships)
            if ships_guess < MIN_FLEET_SIZE:
                continue
            spd = _fleet_speed(ships_guess)
            if spd <= 0:
                continue
            d = math.sqrt(dist_sq)
            flight = max(
                0.0, d - float(q.radius) - float(p_target.radius) - 0.1
            )
            eta_q = max(1, int(math.ceil(flight / spd)))
            if eta_q >= int(earliest_eta):
                continue  # too slow
            best = (q, ships_guess, eta_q)
            best_dist_sq = dist_sq
        if best is None:
            continue

        q, ships, eta_q = best
        # Aim: orbital lead for orbiting targets, static atan2 otherwise.
        if is_orbiting(p_target):
            target_tuple = (
                int(p_target.id), int(p_target.owner),
                float(p_target.x), float(p_target.y),
                float(p_target.radius),
                int(p_target.ships), int(p_target.production),
            )
            aim_res = aim_orbiting(
                (float(q.x), float(q.y)),
                float(q.radius),
                target_tuple,
                float(p_target.radius),
                int(ships),
                float(omega),
            )
            if aim_res is None:
                continue
            angle = float(aim_res[0])
        else:
            angle = math.atan2(
                float(p_target.y) - float(q.y),
                float(p_target.x) - float(q.x),
            )

        moves.append([int(q.id), float(angle), int(ships)])
        used_srcs.add(int(q.id))

    return moves


def make_opp_policy(tier: int = 1) -> Policy:
    """Return a `Callable(obs) -> action` for the given tier.

    Tier defaults to 1 (top-tier mirror) because that's the better
    proxy for the average ladder opponent above μ1100; downgrade to
    Tier 0 in unit tests / parity replays where the legacy Phase 2
    behavior is wanted.
    """
    if tier not in _TIER_REGISTRY:
        raise ValueError(f"unknown opp_model tier: {tier}")
    return _TIER_REGISTRY[tier]


def predict_opponent_action(obs: Any, tier: int = 1) -> list:
    """One-shot prediction — convenience wrapper around `make_opp_policy`.

    The opp's `player` field on `obs` determines which seat is acting;
    we don't override it here. Caller should ensure `obs.player ==
    opp_id`."""
    return make_opp_policy(tier)(obs)


def opponent_action_distribution(
    obs: Any, *, tier: int = 1, samples: int = 1,
) -> list[list]:
    """Return `samples` plausible action sets, weighted by prior probability.

    Stubbed at Tiers 0/1: returns `[deterministic_action] * samples`
    because the underlying policy is deterministic. Reserved for the
    Tier-2 (trained) consumer, which will sample from the
    logistic-regression score distribution. PIMC-style rollout
    consumers (next session) call this; for now it's a single-point
    distribution.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    base = predict_opponent_action(obs, tier=tier)
    return [base for _ in range(samples)]
