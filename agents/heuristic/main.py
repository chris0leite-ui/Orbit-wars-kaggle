"""Heuristic agent — physics-grounded greedy ROI on top of lib/ primitives.

Pipeline (per turn, deterministic):
  1. World.from_obs + WorldModel — board snapshot + per-planet arrival ledger.
  2. Enumerate candidate launches as (roi, tgt_id, fleets) tuples, where
     `fleets` is a list of (src_id, angle, ships) describing one or more
     fleets that together effect the action. Candidate kinds:
       OFFENSE : 1 fleet to a non-owned planet, if any single source can
                 afford the predicted garrison + hold margin.
       DEFENSE : 1 fleet to a threatened own planet, sized to plug the
                 deficit at the predicted flip step.
     Common pipeline per fleet:
       a. Aim — aim_orbiting (static/rotating, fixed-point lead) or aim_comet.
       b. ETA from the converged aim.
       c. ships_needed:
            offense: ceil(WorldModel.ships_at(tgt, eta)) + 1 + hold_margin.
            defense: ceil(deficit_at_first_enemy_arrival) + 1.
       d. Physics gate: predict_fleet_fate.outcome == "target" (Rule 47 —
          no launch we can't verify lands).
       e. ROI = (production × time_to_hold × bonus) / total_ships.
          OFFENSE: bonus = ENEMY_DENIAL_BONUS for enemy-owned, 1 for neutral.
          DEFENSE: no bonus (planet is already ours, we just keep it).
          Comets: time_to_hold clamped by remaining lifetime.
  3. Sort candidates by ROI desc, allocate greedily per source budget.
     Per-target lock prevents over-commit to the same target.

Phases shipped so far:
  v0        — offense-only greedy ROI with physics gate.
  Phase 2a  — defensive reinforcement of own planets predicted to flip.
  Phase 2b  — hold-aware ship sizing (pre-fund the post-capture defense
              against the next in-flight enemy wave, net of own production).
  Phase 3   — source-defense reservation (Rule 40): per-source sendable cap
              bounded by predicted timeline so a launch can't strip a planet
              that's about to be hit by an inbound enemy wave.

Falsified (do not re-add without n=64+ evidence):
  - Static-opening / rotation-factor ROI biases (regressed 12.5% -> 6-9%).
  - Expand-toward-opponent-centroid ROI bias (regressed 12.5% -> 0%).
  - keep-0-ships-behind for no-threat sources (regressed Phase 3 baseline).
  - 2-source joint capture with later-eta alignment (no lift at n=32;
    likely opportunity-cost of double-committing two sources).
  - Multi-wave time_to_hold cap via in-flight ledger sim (no lift at n=32;
    likely over-discounts captures the agent could reinforce later).
  - Idle drain (forward leftover ships to the most-frontier own planet)
    regressed 18.8% -> 0% — recipient is often itself threatened and the
    in-flight stack lands into a doomed garrison.
  Lesson: modeling fixes that PREVENT bad launches (source reservation,
  hold-aware sizing) work; modeling fixes that ENABLE more launches
  (joint, drain) are easy to over-extend into self-harm.

Not yet shipped: comet-specialised sizing, opp-launch reactive capture
(track depleted source post-launch), safer idle-drain (reservation-aware
recipient selection).
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
SOURCE_RESERVE_HORIZON = 30  # how far ahead to project src safety when reserving


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


def _max_sendable(p, wm, my_id, horizon=SOURCE_RESERVE_HORIZON):
    """Max ships we can ship off `p` without making it flip in [1..horizon].

    Reasoning (Rule 40 modeling fix):
      `wm.ships_at(p, K)` is post-combat garrison at step K under the
      no-action timeline. If we ship X off now, every garrison in the
      timeline shifts down by X — UNTIL the first combat that flips on
      account of the reduced ships. The linear approximation breaks
      when we cross zero; up to that point, the bound is
          X < min_{K in 1..H} ships_at(p, K),
      so we can ship up to floor(min - 1) and still retain on every
      pre-projected wave.

      If owner_at(p, K) != my_id for any K in 1..H, we'll lose p anyway
      under the no-action timeline → ship everything (a doomed garrison
      is wasted in place; better to spend it where it does work).

      No-threat case: cap at p.ships - 1 (keep 1 garrison so a single
      enemy snipe doesn't walk in for free).
    """
    has_threat = False
    min_post_combat = float("inf")
    for K in range(1, int(horizon) + 1):
        owner_K = wm.owner_at(p.id, K)
        if owner_K is None:
            break
        if owner_K != my_id:
            return int(p.ships)  # doomed — ship everything
        ships_K = wm.ships_at(p.id, K)
        if ships_K is None:
            break
        if ships_K < min_post_combat:
            min_post_combat = ships_K
        # detect actual enemy pressure on the timeline
        for (eta, owner, ships) in wm.ledger.get(p.id, []):
            if owner != my_id and ships > 0 and int(math.ceil(eta)) == K:
                has_threat = True
                break
    if not has_threat:
        return max(0, int(p.ships) - 1)  # no inbound enemy — keep 1 ship behind
    if min_post_combat == float("inf"):
        return max(0, int(p.ships) - 1)
    return max(0, int(math.floor(min_post_combat - 1)))


def _ships_for_capture(predicted_garrison):
    """Ships needed to flip on arrival: ceil(garrison) + 1 (outnumber)."""
    g = math.ceil(max(0.0, float(predicted_garrison)))
    return max(MIN_SHIPS_TO_LAUNCH, int(g) + GARRISON_BUFFER)


def _hold_margin(target_id, target_production, our_eta, wm, my_id):
    """Extra ships needed to hold the target past the next enemy arrival.

    After we capture at `our_eta`, the planet starts producing for us. Any
    enemy fleet in-flight that arrives at step `E > our_eta` will hit us
    with `enemy_ships_at_E` against our post-capture garrison (which started
    at GARRISON_BUFFER and grew by `production * (E - our_eta)`).

    Returns ceil(max(0, enemy_ships_at_E - production_growth)). When there
    is no inbound enemy fleet post-capture the margin is 0.

    This is Rule 40-aligned: a modeling fix, not a constant bump. Captures
    that v0 made with bare-minimum ships often instantly got re-flipped by
    the next enemy wave; sizing here pre-funds the hold.
    """
    next_enemy = wm.incoming_enemy_eta_after(target_id, my_id, our_eta)
    if next_enemy is None:
        return 0
    arrivals = wm.ledger.get(target_id, [])
    enemy_ships = sum(s for (e, o, s) in arrivals
                      if o != my_id and int(math.ceil(e)) == next_enemy and s > 0)
    if enemy_ships <= 0:
        return 0
    growth = target_production * max(0, next_enemy - our_eta)
    deficit = enemy_ships - growth
    return max(0, int(math.ceil(deficit)))


def _roi(src, tgt, eta_float, ships_needed, world, my_id, wm=None):
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


def _defense_candidate(src, own_planet, world, wm, my_id, src_sendable):
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
    if ships > src_sendable:
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


def _solo_offense_candidate(src, tgt, world, wm, my_id, src_sendable):
    """Single-source offensive capture. Returns (roi, ships, angle, eta_int)
    or None if infeasible.
    """
    guess = max(MIN_SHIPS_TO_LAUNCH, int(tgt.ships) + GARRISON_BUFFER)
    aim = _aim(src, tgt, guess, world)
    if aim is None:
        return None
    _, _, eta_float = aim
    eta_int = max(1, int(math.ceil(eta_float)))

    pred_owner = wm.owner_at(tgt.id, eta_int)
    pred_garrison = wm.ships_at(tgt.id, eta_int)
    if pred_owner == my_id:
        return None  # will already be ours by arrival
    ships = _ships_for_capture(pred_garrison if pred_garrison is not None else tgt.ships)
    ships += _hold_margin(tgt.id, tgt.production, eta_int, wm, my_id)
    if ships > src_sendable:
        return None

    aim = _aim(src, tgt, ships, world)
    if aim is None:
        return None
    angle, _, eta_float = aim
    eta_int = max(1, int(math.ceil(eta_float)))

    fate = predict_fleet_fate(src, tgt, angle, ships, world)
    if fate.outcome != "target" or fate.hit_planet_id != tgt.id:
        return None

    roi = _roi(src, tgt, eta_float, ships, world, my_id, wm)
    if roi is None or roi <= 0:
        return None
    return roi, ships, angle, eta_int


def agent(obs):
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    my_id = world.my_id

    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    all_planets = list(world.planets_by_id.values())

    # Identify own planets predicted to flip soon — defense candidates.
    threatened = [p for p in my_planets
                  if _earliest_flip(p.id, wm, my_id, DEFEND_HORIZON) is not None]

    # Source-defense reservation: per-source cap so a launch can't strip
    # a planet that's about to be hit by an inbound enemy wave.
    sendable = {p.id: _max_sendable(p, wm, my_id) for p in my_planets}

    # Each candidate: (roi, tgt_id, fleets) with fleets = [(src_id, angle, ships), ...].
    candidates: list[tuple[float, int, list[tuple[int, float, int]]]] = []
    solo_targets: set[int] = set()

    for src in my_planets:
        if sendable[src.id] < MIN_SHIPS_TO_LAUNCH:
            continue

        # Defensive reinforcement: src -> own threatened planet.
        for own in threatened:
            if own.id == src.id:
                continue
            res = _defense_candidate(src, own, world, wm, my_id, sendable[src.id])
            if res is None:
                continue
            droi, dships, dangle, _ = res
            candidates.append((droi, own.id, [(src.id, dangle, dships)]))

        for tgt in all_planets:
            if tgt.id == src.id or tgt.owner == my_id:
                continue
            res = _solo_offense_candidate(src, tgt, world, wm, my_id, sendable[src.id])
            if res is None:
                continue
            roi, ships, angle, _ = res
            candidates.append((roi, tgt.id, [(src.id, angle, ships)]))
            solo_targets.add(tgt.id)

    candidates.sort(key=lambda c: -c[0])

    src_budget = dict(sendable)
    used_targets: set[int] = set()
    moves = []
    for roi, tgt_id, fleets in candidates:
        if tgt_id in used_targets:
            continue
        if any(src_budget.get(sid, 0) < sh for sid, _, sh in fleets):
            continue
        for sid, _, sh in fleets:
            src_budget[sid] -= sh
        used_targets.add(tgt_id)
        for sid, ang, sh in fleets:
            moves.append([sid, ang, sh])

    return moves
