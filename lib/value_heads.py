"""Value heads for v9 super-version lookahead rollouts.

A "value head" turns the terminal Snapshot of a K-step rollout into a
single scalar score. The lookahead chooser uses the score to rank
candidate actions. Today we use `delta_us_minus_them` (our minus
their total ships). For v9 we add `inflight_value` which extends the
K-step horizon by an extra 30 turns of static-substrate look-ahead
to credit in-flight fleets that will arrive AFTER the rollout
horizon.

Motivation: the receding-horizon pathology (documented in
`audit/2026-05-12-v4-planner-receding-horizon-pathology.md`).
At K=10 the rollout's leaf state may still have ships in flight
toward a planet they'll capture at eta=11–15. The naive ship-delta
head rates these ships as pure COST (ships not yet on a planet). A
candidate that does NOT launch them keeps them at home and scores
higher — even though the launching candidate would actually win
the capture. This biases lookahead toward "wait" / "don't fire".

`inflight_value` fixes this by building a `WorldModel` over the
terminal Snapshot and reading predicted ownership at terminal+30.
Fleets that arrive within that extended horizon and flip ownership
in our favour contribute their target's production × time-to-hold
to the score.

`composite_capture_value` (2026-05-14) extends this with two more
terms motivated by the PI observation that depth-2 search was biased
toward passive play and we were wasting ships:
- **Capture bonus:** for each of our in-flight fleets, predict whether
  it will land successfully and, if so, credit the captured planet's
  production × remaining episode steps. Directly rewards "go conquer
  the right planets."
- **Waste penalty:** subtract a fraction of ship counts for fleets
  that won't capture — either because they're targeting nothing
  (OOB / sun trajectory) or because they'll bounce off a stronger
  defender. Directly penalises "don't waste ships by sending too few
  or by not conquering."
"""

from __future__ import annotations

from typing import Any

from lib.fast_sim import ship_totals
from lib.intent import World
from lib.world_model import DEFAULT_HORIZON, WorldModel, fleet_target_planet


# Phase 2 audit established AUC ≈ oracle at K=50. K=10 + 30 extra of
# static substrate ≈ K=40 effective; close enough.
INFLIGHT_EXTRA_HORIZON: int = 30

# Patchable value-head selector used by agents/baseline/value.select_favor_fn.
# 0 = favor (default v15 baseline)
# 1 = composite_capture_value
# 2 = projected_rank_diff (production-compounding unified head)
# No type annotation — scripts/ab_variants.py regex-patches `NAME = number`
# lines and does not match annotated assignments. Falls back to env var
# BASELINE_VALUE_HEAD when VALUE_HEAD_CHOICE == 0.
VALUE_HEAD_CHOICE = 0

# How much weight to give the in-flight production credit relative to
# ship-delta. 0.5 chosen so a captured 3-production planet (worth
# ~3*30=90 production-points) approximately balances 90 ships of
# delta. Calibration knob.
INFLIGHT_WEIGHT: float = 0.5


def delta_us_minus_them_obs(obs: Any, my_id: int) -> float:
    """Plain `(our ships) − (their ships)` from a Snapshot's primary
    observation. Phase 2 validated this at AUC ≈ oracle for K=50.

    Renamed from `delta_us_minus_them` to avoid bundle-shadow collision
    with the identically-named `lib.fast_sim.delta_us_minus_them(snap, ...)`.
    The fast_sim version takes a Snapshot; this one takes an obs.
    Same logic, different first-arg type.

    `obs` is `snap.state[my_id].observation` (a `Struct`). Sums
    planet garrisons + in-flight fleet ship counts for owned planets/
    fleets; subtracts each other seat's total.
    """
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ours = 0.0
    theirs = 0.0
    for p in planets:
        owner = int(p[1])
        if owner == my_id:
            ours += float(p[5])
        elif owner >= 0:
            theirs += float(p[5])
    for f in fleets:
        owner = int(f[1])
        if owner == my_id:
            ours += float(f[6])
        elif owner >= 0:
            theirs += float(f[6])
    return ours - theirs


def inflight_value(
    obs: Any, my_id: int,
    *, extra_horizon: int = INFLIGHT_EXTRA_HORIZON,
    weight: float = INFLIGHT_WEIGHT,
) -> float:
    """Composite scoring head: `delta_us_minus_them + weight × inflight_credit`.

    The credit term reads the predicted owner of each planet at
    `step + extra_horizon` from the terminal Snapshot's WorldModel
    (which integrates in-flight fleets). For planets that flip TO
    us within the extended horizon, the credit is `production`.
    Sum across all such planets, weight by `weight`.

    The default weight=0.5 is the calibration knob the v9_inflight
    A/B is gated on. Phase 2 said AUC≈oracle at K=50; this head
    effectively extends K from 10 to ~40 via the static substrate
    while keeping the same rollout cost.

    Empty world (no planets) → returns the base ship-delta only
    (which is 0).
    """
    base = delta_us_minus_them_obs(obs, my_id)
    # Build World from the terminal observation. fast_sim's Snapshot
    # uses Struct, so World.from_obs accepts it.
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base

    # Build a per-planet timeline that integrates current in-flight
    # fleets out to `extra_horizon`. WorldModel.simulate_planet_timeline
    # is O(horizon) per planet and ~1ms total at horizon=30 for a
    # typical board (see audit/2026-05-12-fast-sim-bench.md).
    model = WorldModel.from_world(world, horizon=extra_horizon)

    bonus = 0.0
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            # Already ours; no in-flight credit needed (counted in base).
            continue
        pred_owner = model.owner_at(p.id, extra_horizon)
        if pred_owner == my_id:
            # We'll own it within extra_horizon → credit the production.
            bonus += float(p.production)
    return base + weight * bonus


# ---------------------------------------------------------------------------
# composite_capture_value — anti-waste + capture-aware (v7.4)
# ---------------------------------------------------------------------------


# Coefficients tuned so the three terms are comparable in scale on a
# typical mid-game board (ship-delta in the ~10-50 range, capture bonus
# ~0.05 × 3 × 300 = 45 per high-value capture, waste penalty ~0.5 × ships).
CAPTURE_REWARD_WEIGHT: float = 0.05
WASTE_PENALTY_WEIGHT: float = 0.5
EPISODE_STEPS_TOTAL: int = 500


def composite_capture_value(
    obs: Any, my_id: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
) -> float:
    """Ship-delta + per-fleet capture/waste credit.

    For each of OUR in-flight fleets:
    - Predict the target planet via ray-cast (`fleet_target_planet`).
    - If no target → fleet will OOB or hit sun. Penalise `waste_weight × ships`.
    - If target exists and we'll successfully capture (our ships > predicted
      defenders at arrival, AND target won't already be ours) →
      reward `capture_weight × production × (episode_remaining)`.
    - If target exists but we'll bounce (our ships ≤ predicted defenders) →
      penalise `waste_weight × ships`.
    - If target will already be ours by ETA (over-reinforcement) → no
      reward, no penalty (neutral).

    This directly addresses two pathologies of `delta_us_minus_them`:
    (i) ships in flight count as "lost" in the terminal sum, biasing
    the chooser toward not launching; and (ii) there's no signal that
    a launch is *failing* (bouncing or escaping to OOB), so the chooser
    can't differentiate productive launches from wasteful ones.
    """
    base = delta_us_minus_them_obs(obs, my_id)
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base

    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    if not fleets_raw:
        return base

    # Reuse the kaggle namedtuple so `fleet_target_planet` gets the same
    # type it expects.
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # noqa: E402
    fleets = [Fleet(*f) for f in fleets_raw]
    planets_list = list(world.planets_by_id.values())
    model = WorldModel.from_world(world, horizon=horizon)
    step_now = int(world.step)

    delta = 0.0
    for f in fleets:
        if int(f.owner) != my_id:
            continue
        ships = float(f.ships)
        target, eta = fleet_target_planet(f, planets_list)
        if target is None:
            # No planet on our trajectory — destined for OOB or sun.
            delta -= waste_weight * ships
            continue
        # Predict ownership and garrison at ETA.
        pred_owner = model.owner_at(target.id, eta)
        pred_ships = model.ships_at(target.id, eta) or 0.0
        if pred_owner == my_id:
            # Already ours — reinforcement; no extra credit (already in base).
            continue
        if ships > pred_ships:
            # Will capture. Credit by production × remaining game time.
            time_remaining = max(0, EPISODE_STEPS_TOTAL - step_now - eta)
            delta += capture_weight * float(target.production) * float(time_remaining)
        else:
            # Will bounce — wasted attack.
            delta -= waste_weight * ships

    return base + delta


# ---------------------------------------------------------------------------
# projected_rank_diff — production-compounding unified value head
# ---------------------------------------------------------------------------
#
# Working backward from Kaggle's evaluation (TrueSkill on ordinal rank by
# total ships at T=500), the cheap sufficient statistic for a seat's
# final score is:
#
#     ProjectedTotal_i = ships_now_i
#                      + in_flight_capture/waste_credit_i
#                      + PROJECTION_LAMBDA × Σ_p (P_p × turns_remaining)
#                        for planets p owned by seat i at the leaf state
#
# V(s) = ProjectedTotal_us − max_{j != us} ProjectedTotal_j.
#
# Generalises composite_capture_value's "P × turns_remaining" credit from
# in-flight fleets only to ALL planets at the leaf, and replaces the 2P
# `delta_us_minus_them` aggregation with `max` over opponents — which is
# what TrueSkill ordinal ranking measures against.
#
# Compounding pressure: a P=5 planet owned at step 100 contributes
# 0.05 × 5 × 400 = 100 to your projection — orders of magnitude bigger
# than ship-balance differentials. Pressure to launch (rather than hoard)
# is built into the math, not a tuned passivity-penalty term.


PROJECTION_LAMBDA: float = 0.05  # same scale as CAPTURE_REWARD_WEIGHT


def _per_seat_in_flight_credit(
    obs: Any,
    num_seats: int,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Per-seat in-flight capture / waste credit.

    Generalises composite_capture_value's per-fleet logic: instead of
    attributing the credit to one seat (`my_id`), attributes each fleet's
    predicted fate to its OWN seat's bucket. WorldModel built once.
    """
    credits = {i: 0.0 for i in range(num_seats)}
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return credits
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    if not fleets_raw:
        return credits

    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # noqa: E402
    fleets = [Fleet(*f) for f in fleets_raw]
    planets_list = list(world.planets_by_id.values())
    model = WorldModel.from_world(world, horizon=horizon)
    step_now = int(world.step)

    for f in fleets:
        owner = int(f.owner)
        if owner < 0 or owner >= num_seats:
            continue
        ships = float(f.ships)
        target, eta = fleet_target_planet(f, planets_list)
        if target is None:
            credits[owner] -= waste_weight * ships
            continue
        pred_owner = model.owner_at(target.id, eta)
        pred_ships = model.ships_at(target.id, eta) or 0.0
        if pred_owner == owner:
            continue  # over-reinforcement — neutral
        if ships > pred_ships:
            time_remaining = max(0, EPISODE_STEPS_TOTAL - step_now - eta)
            credits[owner] += capture_weight * float(target.production) * float(time_remaining)
        else:
            credits[owner] -= waste_weight * ships
    return credits


def projected_rank_diff(
    obs: Any,
    my_id: int,
    num_seats: int = 2,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    projection_lambda: float = PROJECTION_LAMBDA,
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Production-compounding unified value head.

    V(s) = ProjectedTotal_us − max_{j != us} ProjectedTotal_j
    where ProjectedTotal_i  = ships_i(now)
                            + in_flight_credit_i
                            + λ · Σ_p P_p · (T − step) for p owned by i at leaf.

    `max` matches TrueSkill ordinal ranking — the next opponent above us
    is what we're racing. In 4P, opponents fighting each other shrinks
    `max` for free (high-risk shots that move us past the leader pay off
    even when the bottom opp does well in absolute terms).

    Linear time-remaining (no γ-discount) per the PV-off finding
    (live A/B 81.2% on submission 52784853). T=500 is a hard horizon;
    exponential decay double-counts.

    Compounding emerges from `P_p × (T − step)`: a P=3 capture at step 100
    is worth ≈ 60 ship-units (0.05 × 3 × 400); at step 400, ≈ 15. Early
    captures are super-linear in elapsed-game-time; the chooser will
    prefer launching to hoarding without a passivity penalty.
    """
    if isinstance(obs, dict):
        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []
        step = int(obs.get("step", 0))
    else:
        planets = getattr(obs, "planets", []) or []
        fleets = getattr(obs, "fleets", []) or []
        step = int(getattr(obs, "step", 0))
    rem = max(0, EPISODE_STEPS_TOTAL - step)

    ships_per = {i: 0.0 for i in range(num_seats)}
    proj_per = {i: 0.0 for i in range(num_seats)}
    for p in planets:
        owner = int(p[1])
        if owner < 0 or owner >= num_seats:
            continue
        ships_per[owner] += float(p[5])
        proj_per[owner] += float(p[6]) * rem
    for f in fleets:
        owner = int(f[1])
        if owner < 0 or owner >= num_seats:
            continue
        ships_per[owner] += float(f[6])

    credits = _per_seat_in_flight_credit(
        obs, num_seats,
        capture_weight=capture_weight, waste_weight=waste_weight,
        horizon=horizon,
    )

    totals = {
        i: ships_per[i] + credits[i] + projection_lambda * proj_per[i]
        for i in range(num_seats)
    }
    my_total = totals[my_id]
    if num_seats <= 1:
        return my_total
    opp_total = max(v for k, v in totals.items() if k != my_id)
    return my_total - opp_total
