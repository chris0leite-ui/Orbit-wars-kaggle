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

    for _ in range(4):
        ae = _aim_and_eta(src, target, K, world.omega)
        if ae is None:
            return None
        angle, eta = ae

        defenders = project_opp_grow_only(world, target, eta)
        K_needed = max(MIN_LAUNCH_SHIPS, int(math.ceil(defenders)) + SAFETY_SHIP_MARGIN)
        if K_needed <= K:
            # Converged or K is already over-margin → use K_needed (tighter).
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

    # Verify the budget actually captures (under our v1 opp model).
    defenders_final = project_opp_grow_only(world, target, eta)
    if K < defenders_final + SAFETY_SHIP_MARGIN:
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
