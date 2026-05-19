"""trajectory_roi v1 — outcome-first analytical agent.

For each turn:
  1. Enumerate candidate OUTCOMES: per (source, target) pair, compute the
     analytical capture-ships and arrival-turn from closed-form combat
     math (defenders grow with target.production during flight time).
  2. Each outcome carries a ROI = value / (cost + 1), where
     value = target.production · (horizon − eta) and cost = ships.
  3. Joint-pack outcomes greedily by ROI, respecting per-source ship
     budgets (a planet's ships are shared across all outcomes that
     would draw from it).
  4. Emit only outcomes whose launch turn is now (turn 0 of the plan).
     Re-plan from scratch next turn.

This naturally produces:
  - Distant planets fire long-range strikes when no closer source can
    afford them — addresses (e) distant-idleness without any special-
    case redeploy logic.
  - Bouncing is impossible by construction — we never commit a fleet
    smaller than defenders_at_arrival + 1 under the assumed opp.
  - Multi-source bundling — when an outcome can't be funded by one
    source, the joint-pack naturally combines neighbours (v2; v1 is
    single-source per outcome).

Opp model is a swappable closed-form function. v1: opp planets grow at
their production, no launches. Pluggable for v2 / v3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.aim import aim_orbiting, flight_distance
from lib.fleet import speed as fleet_speed
from lib.trajectory_layer import World


# ---- tuneables ------------------------------------------------------------

HORIZON = 30                  # turns of future value to credit captures with
MAX_ETA_BEFORE_SKIP = 60      # outcomes arriving later than this are skipped
SAFETY_SHIP_MARGIN = 1        # extra ships above defenders_at_arrival
MIN_LAUNCH_SHIPS = 5          # env minimum
# Source-budget reservation: never drain a source below this fraction of its
# current garrison in a single turn. v1 caps drainage at 100% (no reservation)
# but the knob exists for v2 tightening when we add source-exposure tracking.
SOURCE_DRAIN_FRACTION = 1.00


# ---- per-outcome candidate ------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    src_id: int
    target_id: int
    ships: int            # ships to launch
    eta: int              # arrival turn (1-based; >=1)
    aim_angle: float      # radians, pre-computed
    value: float          # production × (horizon − eta)
    roi: float            # value / (ships + 1)


# ---- opp projection -------------------------------------------------------


def project_opp_grow_only(world: World, planet, turns_ahead: int) -> float:
    """v1 opp projection: planets grow at their production, no launches.

    Returns projected ship count on `planet` after `turns_ahead` turns
    under this opp model. For neutrals (owner=-1) ships do NOT accrete —
    env rule (orbit_wars.py:511-514). For owned planets (any player)
    ships grow by `production` per turn.

    Swap this function out for stronger projections in v2 / v3.
    """
    if planet.owner == -1:
        return float(planet.ships)
    return float(planet.ships) + float(planet.production) * float(turns_ahead)


def _fleet_arrival_eta(fleet, target_planet) -> int | None:
    """Closed-form: integer eta at which `fleet` reaches `target_planet`,
    or None if the fleet's trajectory doesn't intersect the target.

    Static raycast (treats target at its current position — exact for
    non-orbiting; an approximation for orbiting targets). Same shape as
    `lib.world_model._static_first_hit` but tighter (one fleet, one
    target).
    """
    fx, fy = fleet.current_x, fleet.current_y
    spd = fleet.speed
    if spd <= 0:
        return None
    tx, ty, tr = target_planet.current_x, target_planet.current_y, target_planet.radius
    dx, dy = tx - fx, ty - fy
    dir_x, dir_y = math.cos(fleet.angle), math.sin(fleet.angle)
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    r_sq = tr * tr
    if perp_sq >= r_sq:
        return None
    hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
    return int(math.ceil(hit_d / spd))


def project_in_flight(world: World, target_planet, by_eta: int,
                      my_id: int) -> tuple[float, float]:
    """Sum in-flight fleet arrivals at `target_planet` by `by_eta`.

    Returns (our_ships_arriving, opp_ships_arriving). Used to correct
    `defenders_at_eta` for fleets ALREADY in flight in the obs:
    - Our fleets arriving REDUCE the defender count we need to beat
      (they fight defenders for us before our new launch arrives).
    - Opp fleets arriving INCREASE it (reinforcing the target).

    Closed-form static raycast per fleet. Cheap (~planets fleets per
    call, each O(1)).
    """
    ours = 0.0
    theirs = 0.0
    for fleet in world.fleets:
        eta = _fleet_arrival_eta(fleet, target_planet)
        if eta is None or eta > by_eta:
            continue
        if fleet.owner == my_id:
            ours += float(fleet.ships)
        else:
            theirs += float(fleet.ships)
    return ours, theirs


# ---- ETA / aim helpers ----------------------------------------------------


def _aim_and_eta(src, target, ships: int, omega: float):
    """Compute (aim_angle, eta_steps) for `ships` from `src` to `target`.

    For non-orbital targets: straight-line aim, ETA = ceil(flight / speed).
    For orbital targets: 5-iter lead-fixed-point from `lib.aim.aim_orbiting`,
    with safe-intercept fallback. Returns None if no valid intercept.
    """
    src_xy = (src.current_x, src.current_y)
    target_tuple = (
        target.id, target.owner,
        target.current_x, target.current_y,
        target.radius, target.ships, target.production,
    )
    v = fleet_speed(ships)
    if v <= 0 or ships < MIN_LAUNCH_SHIPS:
        return None
    if not getattr(target, "is_rotating", False) or omega == 0.0:
        # Static (or non-rotating environment): direct aim.
        flight = flight_distance(src_xy, src.radius,
                                 (target.current_x, target.current_y),
                                 target.radius)
        if flight <= 0:
            return None
        eta = int(math.ceil(flight / v))
        if eta <= 0 or eta > MAX_ETA_BEFORE_SKIP:
            return None
        ang = math.atan2(target.current_y - src.current_y,
                         target.current_x - src.current_x)
        return (ang, eta)
    # Orbital — fixed-point lead.
    res = aim_orbiting(src_xy, src.radius, target_tuple, target.radius,
                       ships, omega)
    if res is None:
        return None
    angle, _arrival_xy, eta_f = res
    eta = int(math.ceil(eta_f))
    if eta <= 0 or eta > MAX_ETA_BEFORE_SKIP:
        return None
    return (angle, eta)


# ---- per-(src, target) capture solver -------------------------------------


def solve_capture(src, target, world: World) -> Candidate | None:
    """Find the minimum-ship launch from `src` that captures `target`,
    using closed-form combat math under the grow-only opp model.

    Iterative because ETA depends on fleet size (speed-by-ships curve).
    Two passes typically converge; we do up to 4.

    Returns None if no affordable capture exists.
    """
    # Initial guess: enough ships to beat target.ships ignoring flight prod.
    initial_K = max(MIN_LAUNCH_SHIPS, int(target.ships) + SAFETY_SHIP_MARGIN)
    K = initial_K

    my_id = world.my_id

    for _ in range(4):
        ae = _aim_and_eta(src, target, K, world.omega)
        if ae is None:
            return None
        angle, eta = ae

        # Analytical defenders at arrival under v1 opp + in-flight fleets.
        base_defenders = project_opp_grow_only(world, target, eta)
        ours_in_flight, theirs_in_flight = project_in_flight(
            world, target, eta, my_id,
        )
        # If we already have a friendly fleet en route to a target we
        # don't own, we don't need to bring as much firepower. If opp has
        # a fleet en route to a target they don't own (or back to one
        # they do), we need more.
        net_defenders = base_defenders + theirs_in_flight - ours_in_flight
        if target.owner == my_id:
            # Reinforcing one of our own planets — not in scope for this
            # candidate type (we'd compute "defense ROI" differently).
            return None
        K_needed = max(MIN_LAUNCH_SHIPS,
                       int(math.ceil(net_defenders)) + SAFETY_SHIP_MARGIN)
        if K_needed <= MIN_LAUNCH_SHIPS and net_defenders < 0:
            # Already over-captured by our existing in-flight fleets —
            # skip; no new launch needed.
            return None
        if K_needed <= K:
            K = K_needed
            break
        K = K_needed

    # Affordability under current source garrison.
    src_budget = int(src.ships * SOURCE_DRAIN_FRACTION)
    if K > src_budget:
        return None

    # Final aim/eta re-pass at the chosen K (so eta matches K exactly).
    ae = _aim_and_eta(src, target, K, world.omega)
    if ae is None:
        return None
    angle, eta = ae

    # Re-check the budget actually captures under the full model.
    base_defenders_final = project_opp_grow_only(world, target, eta)
    ours_final, theirs_final = project_in_flight(world, target, eta, my_id)
    net_defenders_final = base_defenders_final + theirs_final - ours_final
    if K < max(MIN_LAUNCH_SHIPS, net_defenders_final + SAFETY_SHIP_MARGIN):
        return None

    # Value & ROI.
    horizon_remaining = max(0, HORIZON - eta)
    value = float(target.production) * float(horizon_remaining)
    if value <= 0:
        return None
    roi = value / (float(K) + 1.0)

    return Candidate(
        src_id=src.id,
        target_id=target.id,
        ships=K,
        eta=eta,
        aim_angle=angle,
        value=value,
        roi=roi,
    )


# ---- main entry -----------------------------------------------------------


def agent(obs, configuration=None):
    """Kaggle entry: obs (dict) → list of [src_id, aim_angle, ships]."""
    world = World.from_obs(obs, configuration)
    my_id = world.my_id

    my_planets = [p for p in world.planets if p.owner == my_id]
    targets = [p for p in world.planets if p.owner != my_id]
    if not my_planets or not targets:
        return []

    # Step 1+2+3+4: enumerate (src, target) outcomes via closed-form capture.
    candidates: list[Candidate] = []
    for src in my_planets:
        if src.ships < MIN_LAUNCH_SHIPS:
            continue
        for t in targets:
            c = solve_capture(src, t, world)
            if c is not None:
                candidates.append(c)

    # Step 5: joint-pack greedy by ROI under per-source ship budgets.
    candidates.sort(key=lambda c: -c.roi)
    remaining_budget = {p.id: int(p.ships * SOURCE_DRAIN_FRACTION)
                        for p in my_planets}
    used_target = set()  # one launch per target per turn (avoid dogpile)

    chosen: list[Candidate] = []
    for c in candidates:
        if c.target_id in used_target:
            continue
        if remaining_budget.get(c.src_id, 0) < c.ships:
            continue
        chosen.append(c)
        remaining_budget[c.src_id] -= c.ships
        used_target.add(c.target_id)

    # Step 6: emit turn-0 actions.
    return [[c.src_id, c.aim_angle, c.ships] for c in chosen]
