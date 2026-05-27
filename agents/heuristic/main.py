"""Heuristic agent — physics-grounded greedy ROI on top of lib/ primitives.

Pipeline (per turn, deterministic):
  1. World.from_obs + WorldModel — board snapshot + per-planet arrival ledger.
  2. Enumerate candidate launches:
       OFFENSE: (my_src, enemy_or_neutral_tgt) — capture if ROI > 0.
       DEFENSE: (my_src, my_threatened_planet) — reinforce if predicted to
                flip within DEFEND_HORIZON turns and we can plug the deficit
                before the flip step.
     Each candidate:
       a. Aim — aim_orbiting (static/rotating, fixed-point lead) or aim_comet.
       b. ETA from the converged aim.
       c. ships_needed:
            offense: ceil(WorldModel.ships_at(tgt, eta)) + 1   (outnumber).
            defense: ceil(deficit_at_first_enemy_arrival) + 1.
       d. Physics gate: predict_fleet_fate.outcome == "target" (Rule 47 —
          no launch we can't verify lands).
       e. ROI = (production × time_to_hold × bonus) / ships_needed.
          OFFENSE: bonus = ENEMY_DENIAL_BONUS for enemy-owned, 1 for neutral.
          DEFENSE: no bonus (planet is already ours, we just keep it).
          Comets: time_to_hold clamped by remaining lifetime.
  3. Sort candidates by ROI desc, allocate greedy per source. One fleet per
     target per turn (combat rule 1 stacks same-owner same-step arrivals).

Phases shipped so far:
  v0      — offense-only greedy ROI with physics gate.
  Phase 2a — defensive reinforcement of own planets predicted to flip.

Not yet shipped: opening planner against rotation, expansion bias toward
opponent centroid, multi-source coordination on single capture.
"""

from __future__ import annotations

import math

from lib.aim import aim_comet, aim_orbiting, estimate_eta
from lib.intent import World
from lib.orbit import is_orbiting
from lib.trajectory import predict_fleet_fate
from lib.world_model import WorldModel, _comet_paths_by_id, comet_remaining_lifetime

EPISODE_STEPS = 500
GARRISON_BUFFER = 1          # ships above predicted garrison (+1 outnumbers)
MIN_SHIPS_TO_LAUNCH = 2      # fleets of 1 ship are speed=1.0 and rarely worth it
ENEMY_DENIAL_BONUS = 2.0     # ROI multiplier for capturing enemy planets
DEFEND_HORIZON = 30          # how far ahead to look for predicted flips of own planets
DEFEND_BUFFER = 1            # ships above the deficit so we survive combat


def _aim(src, tgt, ships, world):
    """Return (angle, arrival_xy, eta_float) or None."""
    src_xy = (src.x, src.y)
    tgt_tuple = [tgt.id, tgt.owner, tgt.x, tgt.y, tgt.radius, tgt.ships, tgt.production]

    # Comet: discrete path, not orbital rotation.
    if int(tgt.id) in world.comet_ids:
        paths = _comet_paths_by_id(world)
        entry = paths.get(int(tgt.id))
        if entry is None:
            return None
        path, idx = entry
        return aim_comet(src_xy, src.radius, tgt_tuple, tgt.radius, ships, path, idx)

    # Static planet — straight-line aim is exact.
    if world.omega == 0.0 or not is_orbiting(tgt_tuple):
        eta = estimate_eta(src_xy, src.radius, (tgt.x, tgt.y), tgt.radius, ships)
        if eta is None:
            return None
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        return angle, (tgt.x, tgt.y), eta

    # Orbiting planet — fixed-point lead with safe-intercept fallback.
    return aim_orbiting(src_xy, src.radius, tgt_tuple, tgt.radius, ships, world.omega)


def _ships_for_capture(predicted_garrison):
    """Ships needed to flip on arrival: ceil(garrison) + 1 (outnumber)."""
    g = math.ceil(max(0.0, float(predicted_garrison)))
    return max(MIN_SHIPS_TO_LAUNCH, int(g) + GARRISON_BUFFER)


def _roi(src, tgt, eta_float, ships_needed, world, my_id):
    """ROI = (prod * time_to_hold * bonus) / ships_needed. None if invalid."""
    eta_int = max(1, int(math.ceil(eta_float)))
    time_to_hold = EPISODE_STEPS - world.step - eta_int
    if time_to_hold <= 0:
        return None

    if int(tgt.id) in world.comet_ids:
        lifetime = comet_remaining_lifetime(int(tgt.id), world)
        if lifetime is None:
            return None
        time_to_hold = min(time_to_hold, lifetime - eta_int)
        if time_to_hold <= 0:
            return None

    value = tgt.production * time_to_hold
    if tgt.owner != my_id and tgt.owner != -1:
        value *= ENEMY_DENIAL_BONUS

    return value / ships_needed


def _earliest_flip(planet_id, wm, my_id, horizon):
    """Smallest K in [1..horizon] where pred_owner(planet, K) != my_id, else None."""
    for K in range(1, horizon + 1):
        if wm.owner_at(planet_id, K) != my_id:
            return K
    return None


def _defense_candidate(src, own_planet, world, wm, my_id):
    """If `own_planet` is predicted to flip soon, find a reinforcement from
    `src` that arrives BEFORE the flip with enough ships to hold. Returns
    (roi, ships, angle, eta_int) or None.
    """
    flip_K = _earliest_flip(own_planet.id, wm, my_id, DEFEND_HORIZON)
    if flip_K is None:
        return None

    # First aim with a cheap guess to get an eta estimate.
    aim = _aim(src, own_planet, MIN_SHIPS_TO_LAUNCH, world)
    if aim is None:
        return None
    _, _, eta_float = aim
    eta_int = max(1, int(math.ceil(eta_float)))
    if eta_int >= flip_K:
        return None  # arrives too late

    # Deficit at the flip step: enemy ships that arrive vs our pre-combat garrison.
    arrivals = wm.ledger.get(own_planet.id, [])
    enemy_at_flip = sum(s for (e, o, s) in arrivals
                        if o != my_id and int(math.ceil(e)) == flip_K and s > 0)
    if enemy_at_flip <= 0:
        return None  # no inbound at flip step — flip cause unclear, skip for v0

    g_prev = wm.ships_at(own_planet.id, flip_K - 1) or 0.0
    prev_owner = wm.owner_at(own_planet.id, flip_K - 1)
    g_pre_combat = float(g_prev) + (own_planet.production if prev_owner == my_id else 0.0)
    deficit = enemy_at_flip - g_pre_combat
    if deficit <= 0:
        return None

    ships = int(math.ceil(deficit)) + DEFEND_BUFFER
    if ships < MIN_SHIPS_TO_LAUNCH:
        ships = MIN_SHIPS_TO_LAUNCH
    if ships > src.ships:
        return None

    # Re-aim with the correct ship count (speed depends on ships).
    aim = _aim(src, own_planet, ships, world)
    if aim is None:
        return None
    angle, _, eta_float2 = aim
    eta_int2 = max(1, int(math.ceil(eta_float2)))
    if eta_int2 >= flip_K:
        return None

    fate = predict_fleet_fate(src, own_planet, angle, ships, world)
    if fate.outcome != "target" or fate.hit_planet_id != own_planet.id:
        return None

    # Value: production we keep × time_to_hold. No ENEMY_DENIAL bonus (already ours).
    time_to_hold = EPISODE_STEPS - world.step - eta_int2
    if time_to_hold <= 0:
        return None
    roi = (own_planet.production * time_to_hold) / ships
    return roi, ships, angle, eta_int2


def agent(obs):
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    my_id = world.my_id

    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    all_planets = list(world.planets_by_id.values())

    # Identify own planets predicted to flip soon — defense candidates.
    threatened = [p for p in my_planets
                  if _earliest_flip(p.id, wm, my_id, DEFEND_HORIZON) is not None]

    candidates = []  # (roi, src_id, tgt_id, ships, angle)
    for src in my_planets:
        if src.ships < MIN_SHIPS_TO_LAUNCH:
            continue

        # Defensive reinforcement: src -> own threatened planet.
        for own in threatened:
            if own.id == src.id:
                continue
            res = _defense_candidate(src, own, world, wm, my_id)
            if res is None:
                continue
            droi, dships, dangle, _ = res
            candidates.append((droi, src.id, own.id, dships, dangle))

        for tgt in all_planets:
            if tgt.id == src.id or tgt.owner == my_id:
                continue

            # First aim pass with a ship-count guess so speed is approx right.
            guess = max(MIN_SHIPS_TO_LAUNCH, int(tgt.ships) + GARRISON_BUFFER)
            aim = _aim(src, tgt, guess, world)
            if aim is None:
                continue
            _, _, eta_float = aim
            eta_int = max(1, int(math.ceil(eta_float)))

            pred_owner = wm.owner_at(tgt.id, eta_int)
            pred_garrison = wm.ships_at(tgt.id, eta_int)
            if pred_owner == my_id:
                continue  # will already be ours by arrival
            ships = _ships_for_capture(pred_garrison if pred_garrison is not None else tgt.ships)
            if ships > src.ships:
                continue

            # Re-aim with the actual ship count (speed depends on ship count).
            aim = _aim(src, tgt, ships, world)
            if aim is None:
                continue
            angle, _, eta_float = aim

            # Rule 47 physics gate: refuse any launch we can't predict lands.
            fate = predict_fleet_fate(src, tgt, angle, ships, world)
            if fate.outcome != "target" or fate.hit_planet_id != tgt.id:
                continue

            roi = _roi(src, tgt, eta_float, ships, world, my_id)
            if roi is None or roi <= 0:
                continue
            candidates.append((roi, src.id, tgt.id, ships, angle))

    candidates.sort(key=lambda c: -c[0])

    src_ships = {p.id: p.ships for p in my_planets}
    used_targets: set[int] = set()
    moves = []
    for roi, src_id, tgt_id, ships, angle in candidates:
        if tgt_id in used_targets:
            continue
        if src_ships.get(src_id, 0) < ships:
            continue
        src_ships[src_id] -= ships
        used_targets.add(tgt_id)
        moves.append([src_id, angle, ships])

    return moves
