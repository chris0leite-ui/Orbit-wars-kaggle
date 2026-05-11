"""ROI metric for individual shots and multi-source waves.

Value = production captured * remaining-game-steps (compounds linearly with time).
For enemy-owned targets, doubled (denied + acquired).
Cost = ships invested.
ROI = value / cost, with rejection if predicted defender > attacker.
"""
from __future__ import annotations

import math
from agents.precision import fast_sim, intercept, prediction, sim


def _capture_value(
    target: intercept.PlanetView,
    arrival_step: int,
    end_step: int,
    world: dict | None = None,
    enemy_arrivals: list[prediction.Arrival] | None = None,
) -> int:
    """Lifetime production value of capturing this target at arrival_step.

    Uses the PROJECTED owner at arrival (not target.owner from the current obs).
    A neutral planet about to be captured by an enemy fleet IS effectively
    enemy-owned at our arrival — doubled value for capturing it from them.
    """
    remaining = max(0, end_step - arrival_step)
    base = target.production * remaining
    # Determine projected owner at arrival.
    projected_owner = target.owner  # fallback if no world context provided
    if world is not None:
        timeline = fast_sim.planet_garrison_projection(
            world, [], target.id,
            horizon_steps=arrival_step - world["step"] + 2,
            extra_arrivals=enemy_arrivals or [],
        )
        pre = [t for t in timeline if t[0] < arrival_step]
        if pre:
            projected_owner = pre[-1][1]
    me = world["player"] if world is not None else None
    if projected_owner == -1:
        return base
    if me is not None and projected_owner == me:
        return 0  # capturing our own planet has no value
    return 2 * base  # enemy-owned (current or projected): denied + acquired


def _defender_at(target: intercept.PlanetView, arrival_step: int, world: dict,
                 enemy_arrivals: list[prediction.Arrival] | None = None) -> int:
    """Predict the defender ship count at `arrival_step`, accounting for projected
    enemy launches (if provided). Assumes no other action by us."""
    plan = []
    extra = enemy_arrivals or []
    timeline = fast_sim.planet_garrison_projection(world, plan, target.id,
                                                   horizon_steps=arrival_step - world["step"] + 2,
                                                   extra_arrivals=extra)
    pre = [t for t in timeline if t[0] < arrival_step]
    if not pre:
        return target.ships
    last = pre[-1]
    # last = (step, owner, ships). At the START of arrival_step processing (before combat),
    # the planet has its end-of-(arrival_step-1) state.
    return last[2]


def shot_roi(
    shot: intercept.Shot,
    target: intercept.PlanetView,
    world: dict,
    end_step: int = sim.EPISODE_STEPS,
    enemy_arrivals: list[prediction.Arrival] | None = None,
) -> float:
    """ROI for a single-source shot. Zero if the shot wouldn't capture."""
    arrival_step = world["step"] + shot.eta
    defender = _defender_at(target, arrival_step, world, enemy_arrivals)
    # Strictly more to flip ownership (engine line 672).
    if shot.ship_count <= defender:
        return 0.0
    value = _capture_value(target, arrival_step, end_step, world=world,
                          enemy_arrivals=enemy_arrivals)
    cost = max(1, shot.ship_count)
    return float(value) / float(cost)


def wave_roi(
    wave_shots: list[intercept.Shot],
    target: intercept.PlanetView,
    world: dict,
    end_step: int = sim.EPISODE_STEPS,
    enemy_arrivals: list[prediction.Arrival] | None = None,
) -> float:
    """ROI for a synchronized wave (all shots arriving at the same step)."""
    if not wave_shots:
        return 0.0
    arrival_step = world["step"] + wave_shots[0].eta
    # All shots must arrive at the same step.
    if any(s.eta != wave_shots[0].eta for s in wave_shots):
        return 0.0
    total = sum(s.ship_count for s in wave_shots)
    defender = _defender_at(target, arrival_step, world, enemy_arrivals)
    if total <= defender:
        return 0.0
    value = _capture_value(target, arrival_step, end_step, world=world,
                          enemy_arrivals=enemy_arrivals)
    return float(value) / float(total)


def defense_reserve_table(world: dict, projected_enemy_arrivals: list[prediction.Arrival],
                          horizon: int = 60,
                          include_in_flight: bool = True) -> dict[int, int]:
    """For each owned planet, ships to hold back to survive the worst projected
    enemy arrival within `horizon` ticks.

    Sources of threats considered:
      - Projected future enemy launches (from `enemy_model`).
      - In-flight enemy fleets currently observable in `world["fleets"]`
        (projected to their first-impact planet via
        `prediction.project_enemy_fleet_arrival`).

    reserve = max(0, max_simultaneous_enemy_arrival - growth_to_that_step - 1)
    """
    me = world["player"]
    threats_by_planet: dict[int, list[prediction.Arrival]] = {}

    def _add(arr: prediction.Arrival):
        if arr.owner == me:
            return
        threats_by_planet.setdefault(arr.planet_id, []).append(arr)

    for arr in projected_enemy_arrivals:
        _add(arr)

    if include_in_flight:
        for fleet in world["fleets"]:
            if fleet[1] == me or fleet[1] == -1:
                continue
            arr = prediction.project_enemy_fleet_arrival(fleet, world, horizon_steps=horizon)
            if arr is not None:
                _add(arr)

    reserve: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner != me:
            continue
        threats = threats_by_planet.get(p.id, [])
        worst = 0
        for arr in threats:
            ticks = arr.step - world["step"]
            if ticks <= 0 or ticks > horizon:
                continue
            growth = p.production * ticks
            need = max(0, arr.ships - growth - 1)
            if need > worst:
                worst = need
        reserve[p.id] = min(worst, p.ships)
    return reserve
