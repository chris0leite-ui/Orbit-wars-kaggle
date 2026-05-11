"""Forward rollout of the game state.

Given a world and a set of scheduled fleet arrivals (ours + enemies in flight),
roll forward step-by-step applying the engine's turn order. Tracks per-planet
ownership and garrison over time; produces a terminal score for scoring plans.

The rollout does NOT search over enemy actions — it just propagates the
deterministic consequences of currently-launched fleets. Enemy responses are
modeled by the planner's depth-1 minimax counter step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from agents.precision import sim, intercept


@dataclass
class Arrival:
    """A scheduled fleet arrival at a planet."""
    step: int          # absolute env step at which combat resolves
    planet_id: int
    owner: int
    ships: int


@dataclass
class RolloutResult:
    horizon: int
    final_owner: dict[int, int]
    final_ships: dict[int, int]
    final_score_per_player: dict[int, int]
    # When `track_per_step=True`, these are populated; otherwise empty.
    # step_state[step][planet_id] = (owner, ships) snapshot AFTER step processing.
    step_state: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)
    # arrivals_by_step[step] = list of Arrival impacting that step.
    arrivals_by_step: dict[int, list["Arrival"]] = field(default_factory=dict)
    # Per-player production rate over time: production_per_player[step][player] = int.
    production_per_player: dict[int, dict[int, int]] = field(default_factory=dict)
    # Per-planet timeline (only with track_per_step=True): planet_timeline[pid] = list of
    # (step, owner, ships) tuples — one entry per processed step.
    planet_timeline: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)


def _planet_pos_at_abs_step(initial_x, initial_y, planet_radius, omega, abs_step):
    """Planet position at engine state-after-step abs_step (== env.steps[abs_step] for abs_step>=1)."""
    if not sim.is_orbiting(initial_x, initial_y, planet_radius):
        return (initial_x, initial_y)
    orb_r = math.hypot(initial_x - sim.CENTER, initial_y - sim.CENTER)
    init_ang = math.atan2(initial_y - sim.CENTER, initial_x - sim.CENTER)
    rotations = max(0, abs_step - 1) if abs_step >= 1 else 0
    ang = init_ang + omega * rotations
    return (sim.CENTER + orb_r * math.cos(ang), sim.CENTER + orb_r * math.sin(ang))


def project_enemy_fleet_arrival(
    fleet: list,
    world: dict,
    horizon_steps: int = 120,
) -> Arrival | None:
    """Project an in-flight enemy fleet's first impact (planet, sun, or OOB).

    Returns Arrival if the fleet hits one of our or neutral planets, else None.
    """
    fid, owner, fx, fy, angle, src_id, ships = fleet
    v = sim.fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cur_step = world["step"]
    obs_step = world["step"]
    omega = world["omega"]
    all_planets = world["planets"]

    for j in range(1, horizon_steps + 1):
        A = (fx + (j - 1) * v * cos_a, fy + (j - 1) * v * sin_a)
        B = (A[0] + v * cos_a, A[1] + v * sin_a)
        # Check planet hits (in obs0.planets order — engine first-hit semantics)
        for p in all_planets:
            sweep = intercept.target_sweep(p, omega, j - 1, obs_step)
            if sweep is None:
                continue
            if p.is_comet and sweep[0] == sweep[1] and sweep[0][0] < 0:
                continue
            if sim.swept_pair_hit(A, B, sweep[0], sweep[1], p.radius):
                return Arrival(
                    step=cur_step + j,
                    planet_id=p.id,
                    owner=owner,
                    ships=ships,
                )
        # Check sun/OOB → fleet dies, no arrival
        if sim.segment_oob(A, B):
            return None
        if sim.segment_crosses_sun(A, B, margin=0.0):
            return None
    return None


def rollout(
    world: dict,
    planned_shots: list[intercept.Shot],
    horizon_steps: int = 200,
    track_per_step: bool = False,
) -> RolloutResult:
    """Forward-roll the world by horizon_steps applying scheduled arrivals.

    Inputs:
      - world: parsed world dict (current state).
      - planned_shots: list of Shot objects launched THIS turn (from us).
        Each shot will arrive at step `world["step"] + shot.eta`. Source planet's
        garrison loses shot.ship_count immediately.

    The rollout:
      - Schedules our launches (debits source garrisons, queues arrivals).
      - Projects all in-flight enemy fleets to find their impact step/planet.
      - Steps forward 1..horizon_steps:
        - At each step: production for owned planets, apply arrivals (combat).
      - Returns final state + score per player.
    """
    cur_step = world["step"]
    end_step = min(cur_step + horizon_steps, sim.EPISODE_STEPS - 2)

    # Initial planet state. Copy: id -> [owner, ships, production].
    state = {p.id: [p.owner, p.ships, p.production] for p in world["planets"]}

    # Comet exit step (when path runs out). We treat comets as dropping out
    # of the game; their garrison is lost.
    comet_exit_step: dict[int, int] = {}
    for grp in world["comets"]:
        path_index = grp["path_index"]
        for i, pid in enumerate(grp["planet_ids"]):
            remaining = len(grp["paths"][i]) - path_index
            if remaining >= 0:
                comet_exit_step[pid] = cur_step + remaining

    # Arrivals queue, keyed by step.
    arrivals_by_step: dict[int, list[Arrival]] = {}

    def schedule(arr: Arrival):
        arrivals_by_step.setdefault(arr.step, []).append(arr)

    # Schedule our planned shots; debit source garrison NOW.
    for shot in planned_shots:
        if shot.src_id not in state:
            continue
        cur = state[shot.src_id]
        if cur[1] < shot.ship_count:
            continue  # over-allocated; skip
        cur[1] -= shot.ship_count
        schedule(Arrival(
            step=cur_step + shot.eta,
            planet_id=shot.tgt_id,
            owner=world["player"],
            ships=shot.arrival_ships,
        ))

    # Project enemy fleets currently in flight.
    for fleet in world["fleets"]:
        owner = fleet[1]
        if owner == world["player"]:
            continue  # our own (already-launched) fleets, if any
        arr = project_enemy_fleet_arrival(fleet, world, horizon_steps=horizon_steps)
        if arr is not None:
            schedule(arr)

    # Per-step trajectories, only filled when requested (cheap to skip in hot path).
    step_state: dict[int, dict[int, tuple[int, int]]] = {}
    production_per_player: dict[int, dict[int, int]] = {}
    planet_timeline: dict[int, list[tuple[int, int, int]]] = {}
    if track_per_step:
        planet_timeline = {pid: [] for pid in state}

    # Advance step-by-step.
    for s in range(cur_step + 1, end_step + 1):
        # Comet exits at start of step (engine line 410+).
        for pid, exit_at in list(comet_exit_step.items()):
            if s >= exit_at:
                state.pop(pid, None)
                comet_exit_step.pop(pid, None)

        # Production: each owned planet/comet gains its production. Engine
        # line 511-514 (after action processing).
        for pid, (owner, ships, prod) in state.items():
            if owner != -1:
                state[pid][1] = ships + prod

        # Apply arrivals at this step (combat).
        arrs = arrivals_by_step.get(s, [])
        if arrs:
            # Group by planet.
            by_planet: dict[int, list[tuple[int, int]]] = {}
            for arr in arrs:
                if arr.planet_id not in state:
                    continue
                by_planet.setdefault(arr.planet_id, []).append((arr.owner, arr.ships))
            for pid, atks in by_planet.items():
                gowner, gships, _ = state[pid]
                new_owner, new_ships = sim.combat_resolve(gowner, gships, atks)
                state[pid][0] = new_owner
                state[pid][1] = new_ships

        if track_per_step:
            step_state[s] = {pid: (v[0], v[1]) for pid, v in state.items()}
            prod_per: dict[int, int] = {}
            for pid, (owner, _, prod) in state.items():
                if owner != -1:
                    prod_per[owner] = prod_per.get(owner, 0) + prod
            production_per_player[s] = prod_per
            for pid, v in state.items():
                planet_timeline[pid].append((s, v[0], v[1]))

    # Compute scores: total ships on owned planets per player.
    score: dict[int, int] = {}
    for pid, (owner, ships, _) in state.items():
        if owner == -1:
            continue
        score[owner] = score.get(owner, 0) + ships

    final_owner = {pid: v[0] for pid, v in state.items()}
    final_ships = {pid: v[1] for pid, v in state.items()}
    return RolloutResult(
        horizon=horizon_steps,
        final_owner=final_owner,
        final_ships=final_ships,
        final_score_per_player=score,
        step_state=step_state,
        arrivals_by_step=arrivals_by_step,
        production_per_player=production_per_player,
        planet_timeline=planet_timeline,
    )


def production_by_player(world: dict) -> dict[int, int]:
    """Aggregate production rate per player from the CURRENT observation.

    Returns {player_id: total_production_ships_per_turn} including neutral
    keyed under -1 if any neutral planets exist (neutrals don't actually
    produce — engine line 513 skips owner=-1 — but useful for diagnostics).
    """
    totals: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner == -1:
            continue
        totals[p.owner] = totals.get(p.owner, 0) + p.production
    return totals


def planet_arrivals_timeline(
    world: dict,
    planned_shots: list[intercept.Shot],
    horizon_steps: int = 200,
) -> dict[int, list[Arrival]]:
    """For each planet, list every scheduled arrival sorted by step.

    Includes our planned launches (this turn) + projected enemy-fleet impacts.
    Does NOT include future enemy launches (they're not yet observable).
    Lookup: timeline[planet_id] -> [Arrival(step, planet_id, owner, ships), ...]
    """
    cur_step = world["step"]
    by_planet: dict[int, list[Arrival]] = {p.id: [] for p in world["planets"]}

    for shot in planned_shots:
        by_planet.setdefault(shot.tgt_id, []).append(Arrival(
            step=cur_step + shot.eta,
            planet_id=shot.tgt_id,
            owner=world["player"],
            ships=shot.arrival_ships,
        ))

    for fleet in world["fleets"]:
        if fleet[1] == world["player"]:
            continue
        arr = project_enemy_fleet_arrival(fleet, world, horizon_steps=horizon_steps)
        if arr is not None:
            by_planet.setdefault(arr.planet_id, []).append(arr)

    for pid in by_planet:
        by_planet[pid].sort(key=lambda a: a.step)
    return by_planet


def planet_garrison_projection(
    world: dict,
    planned_shots: list[intercept.Shot],
    planet_id: int,
    horizon_steps: int = 200,
) -> list[tuple[int, int, int]]:
    """For a single planet, return the (step, owner, ships) trajectory.

    Combines production accrual + scheduled-arrival combat. Useful for
    "will I still own this planet at step k? How many ships?"
    """
    res = rollout(world, planned_shots, horizon_steps=horizon_steps, track_per_step=True)
    return res.planet_timeline.get(planet_id, [])


def plan_score(
    world: dict,
    planned_shots: list[intercept.Shot],
    horizon_steps: int = 200,
) -> float:
    """Higher = better for us. Margin = my_score - max(opp_score).

    Score is total-ships-on-owned-planets per player after `horizon_steps`.
    """
    me = world["player"]
    res = rollout(world, planned_shots, horizon_steps=horizon_steps)
    my = res.final_score_per_player.get(me, 0)
    opp_max = 0
    for player, sc in res.final_score_per_player.items():
        if player != me and sc > opp_max:
            opp_max = sc
    return float(my - opp_max)
