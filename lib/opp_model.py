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
from typing import Any, Callable

from lib.fleet import speed as _fleet_speed
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


Policy = Callable[[Any], list]


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
        moves.append([src[0], angle, ships])
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
