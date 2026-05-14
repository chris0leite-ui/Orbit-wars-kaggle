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
# defensibility_value — penalise undefended captures (2026-05-14 PI obs.)
# ---------------------------------------------------------------------------


# Horizon over which we score "could an enemy plausibly recapture this
# planet?". 20 steps ≈ the time a mid-distance enemy planet needs to
# launch a fleet that arrives. Short enough that close-by threats
# dominate, long enough that "leave 1 ship, get recaptured" cases fire.
DEFENSIBILITY_HORIZON: int = 20

# Coefficient on the per-planet vulnerability penalty. Sweep candidate.
DEFENSIBILITY_WEIGHT: float = 1.0


def defensibility_value(
    obs: Any, my_id: int,
    *,
    horizon: int = DEFENSIBILITY_HORIZON,
    weight: float = DEFENSIBILITY_WEIGHT,
) -> float:
    """Ship-delta minus production-weighted per-planet vulnerability.

    For each of OUR planets:
    - `threat_eta = WorldModel.time_to_enemy_threat(p, my_id, world)` —
      earliest turn any enemy could plausibly arrive (combines in-flight
      fleets + worst-case launches from each enemy planet at current
      garrison).
    - If `threat_eta` is `None` or beyond `horizon`, the planet is safe.
    - Otherwise estimate vulnerability as `our_garrison_at_eta - max_enemy_ships`
      where `max_enemy_ships` is the largest single enemy garrison
      (pessimistic single-source threat — fast and pessimistic-correct
      for ranking candidates against each other).
    - Sum the NEGATIVE part of the margin, weighted by planet production.

    Pathology this targets (2026-05-14 PI observation): the chooser
    completes a successful capture of a neutral planet but leaves only
    a handful of garrison ships behind. With `delta_us_minus_them`,
    that capture scores well (+production for ship gain, no cost for
    low garrison). The enemy then recaptures cheaply on the next turn.
    Defensibility penalises this *at the leaf state* so the chooser
    prefers candidates that either (a) commit enough ships to hold
    the captured planet or (b) capture planets that aren't in
    enemy reach.

    Cost: ~3-5 ms per call (WorldModel build + planet loop). value_fn
    is called once per candidate at the rollout leaf, so the per-turn
    overhead is ~50-100 ms for typical candidate counts — fits inside
    the 700 ms wallclock budget.
    """
    base = delta_us_minus_them_obs(obs, my_id)
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base
    model = WorldModel.from_world(world, horizon=horizon)

    # Max single-source enemy threat (worst-case single-planet launch).
    max_enemy_ships = 0.0
    for pp in world.planets_by_id.values():
        if pp.owner == my_id or pp.owner == -1:
            continue
        if pp.ships > max_enemy_ships:
            max_enemy_ships = float(pp.ships)

    penalty = 0.0
    for p in world.planets_by_id.values():
        if p.owner != my_id:
            continue
        threat_eta = model.time_to_enemy_threat(p.id, my_id, world)
        if threat_eta is None or threat_eta > horizon:
            continue
        # Our garrison at the threat-arrival time (accounts for in-flight
        # friendly reinforcements). Fall back to current garrison if the
        # timeline is unavailable.
        our_garrison = model.ships_at(p.id, threat_eta)
        if our_garrison is None:
            our_garrison = float(p.ships)
        margin = float(our_garrison) - max_enemy_ships
        if margin < 0:
            # Vulnerable. Magnitude scales with severity (margin) and
            # planet value (production).
            penalty += margin * float(p.production)  # margin<0 ⇒ negative

    return base + weight * penalty


def composite_plus_defensibility(
    obs: Any, my_id: int,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    defensibility_horizon: int = DEFENSIBILITY_HORIZON,
    defensibility_weight: float = DEFENSIBILITY_WEIGHT,
) -> float:
    """Layered head: composite_capture_value + defensibility penalty.

    Combines waste-aware launch scoring (composite) with
    undefended-capture penalty (defensibility). Use when V1 and V2 both
    lift independently — the combined signal should lift more than
    either alone.
    """
    composite = composite_capture_value(
        obs, my_id,
        capture_weight=capture_weight,
        waste_weight=waste_weight,
    )
    # Subtract the base (delta_us_minus_them_obs) from defensibility
    # because composite already includes it; we want to add only the
    # defensibility *increment*.
    defensibility_full = defensibility_value(
        obs, my_id,
        horizon=defensibility_horizon,
        weight=defensibility_weight,
    )
    base = delta_us_minus_them_obs(obs, my_id)
    return composite + (defensibility_full - base)


# ---------------------------------------------------------------------------
# territory_value — production-weighted ongoing position quality
# ---------------------------------------------------------------------------


# Cap on expected-hold turns so a planet deep in our own cluster (no enemy
# threat → expected_hold saturates at remaining_episode) doesn't explode the
# magnitude. EPISODE_STEPS_TOTAL=500 already exists above; reusing it here
# under a clearer name for the head.
TERRITORY_HORIZON_CAP: int = EPISODE_STEPS_TOTAL

# Outer coefficient. production×hold can sum to ~5000-10000 across our planets
# in mid-game; delta is ~10-100. Caller (the iter dispatcher) is expected to
# pass a small weight (typical 0.005-0.02) so the territory term and delta
# end up at comparable magnitudes.
TERRITORY_WEIGHT: float = 1.0


def territory_value(
    obs: Any, my_id: int,
    *,
    weight: float = TERRITORY_WEIGHT,
    horizon_cap: int = TERRITORY_HORIZON_CAP,
) -> float:
    """Ship-delta + weight × (my territorial production-time − their territorial production-time).

    For each non-neutral planet, compute `production × expected_hold` where
    expected_hold is the number of turns the current owner retains the planet
    given predicted threats from the OPPOSITE side. Sum for our planets,
    subtract for enemy planets, scale by `weight`.

    Targets the conquer-then-undefend pathology: a planet just captured with
    1 ship in enemy reach has expected_hold ~5; the same planet with 50 ships
    has expected_hold capped at remaining-episode. The leaf head sees the
    difference and the chooser prefers candidates that produce DEFENSIBLE
    captures, not just any capture.

    We inline the threat-eta computation (rather than call
    `lib.scoring.expected_hold`) because that helper hard-codes `world.my_id`
    in its threat lookup — useless for an enemy planet. The reused helper
    `model.time_to_enemy_threat(p.id, p.owner, world)` returns "earliest
    threat from any non-owner", which is correct for both our and enemy
    planets (in 2P it's symmetric; in 4P it's the worst-case across the
    3 non-owners, an appropriate conservatism).

    Cost: ~one WorldModel build (~1-3 ms) + a planet loop. Comparable to
    composite_capture_value; fits inside the 700 ms per-turn budget.
    """
    base = delta_us_minus_them_obs(obs, my_id)
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base
    model = WorldModel.from_world(world)
    step_now = int(world.step)
    remaining = max(0, horizon_cap - step_now)
    if remaining == 0:
        return base

    me_value = 0.0
    them_value = 0.0
    for p in world.planets_by_id.values():
        if p.owner == -1:
            continue  # neutrals contribute nothing
        threat = model.time_to_enemy_threat(p.id, p.owner, world)
        if threat is None:
            hold = remaining
        else:
            hold = min(remaining, max(0, int(threat)))
        contribution = float(p.production) * float(hold)
        if p.owner == my_id:
            me_value += contribution
        else:
            them_value += contribution
    return base + weight * (me_value - them_value)


def composite_plus_territory(
    obs: Any, my_id: int,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    territory_weight: float = TERRITORY_WEIGHT,
    territory_horizon_cap: int = TERRITORY_HORIZON_CAP,
) -> float:
    """Layered head: composite_capture_value increment + territory_value increment.

    Both heads include the same `delta_us_minus_them_obs` base. We subtract
    the base from each so the combined return is `base + composite_incr +
    territory_incr` — counting delta once. The two increments measure
    orthogonal signals (launch quality vs ongoing position) so layering
    should lift more than either alone.
    """
    base = delta_us_minus_them_obs(obs, my_id)
    composite = composite_capture_value(
        obs, my_id,
        capture_weight=capture_weight,
        waste_weight=waste_weight,
    )
    territory = territory_value(
        obs, my_id,
        weight=territory_weight,
        horizon_cap=territory_horizon_cap,
    )
    return base + (composite - base) + (territory - base)
