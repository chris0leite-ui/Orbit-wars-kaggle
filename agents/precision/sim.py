"""Deterministic physics for Orbit Wars.

Mirrors kaggle_environments/envs/orbit_wars/orbit_wars.py exactly.
Constants and formulas verified against engine source.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
MAX_SHIP_SPEED = 6.0
COMET_SPEED = 4.0
SPAWN_OFFSET = 0.1
SUN_SAFETY_MARGIN = 1.0
EPISODE_STEPS = 500


def fleet_speed(ships: int, max_speed: float = MAX_SHIP_SPEED) -> float:
    """Per-tick straight-line speed for a fleet of `ships`. Engine line 577-578.

    Engine has no lower clamp; ships<=0 are rejected at action validation.
    We clamp to 1.0 defensively for ship_count=1 case (log(1)=0 -> exactly 1.0).
    """
    if ships <= 1:
        return 1.0
    v = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(v, max_speed)


def ships_for_speed(target_v: float, max_speed: float = MAX_SHIP_SPEED) -> int:
    """Inverse of fleet_speed: smallest int S with fleet_speed(S) >= target_v."""
    if target_v <= 1.0:
        return 1
    if target_v >= max_speed:
        return 1000
    ratio = (target_v - 1.0) / (max_speed - 1.0)
    log_ratio = ratio ** (1.0 / 1.5)
    s = math.exp(log_ratio * math.log(1000.0))
    s_int = max(1, int(math.ceil(s)))
    while s_int > 1 and fleet_speed(s_int - 1) >= target_v:
        s_int -= 1
    while fleet_speed(s_int) < target_v and s_int < 1000:
        s_int += 1
    return s_int


def is_orbiting_sim(planet_x: float, planet_y: float, planet_radius: float) -> bool:
    """Engine line 540: orbiting iff orbital_radius + planet_radius < 50."""
    orb_r = math.hypot(planet_x - CENTER, planet_y - CENTER)
    return orb_r + planet_radius < ROTATION_RADIUS_LIMIT


def orbit_params(initial_x: float, initial_y: float) -> tuple[float, float]:
    """(orbital_radius, initial_angle) from a planet's INITIAL position."""
    dx = initial_x - CENTER
    dy = initial_y - CENTER
    return math.hypot(dx, dy), math.atan2(dy, dx)


def planet_pos_after_rotations(
    initial_x: float, initial_y: float, omega: float, num_rotations: int
) -> tuple[float, float]:
    """Planet position after `num_rotations` rotations from its INITIAL position.

    Engine line 542: `current_angle = initial_angle + angular_velocity * step`.
    So planet_pos_after_rotations(init, omega, k) gives position at end of step k.
    At observation time of step N, the planet has been rotated N-1 times (the
    last rotation happened during the engine's processing of step N-1).
    """
    orb_r, init_angle = orbit_params(initial_x, initial_y)
    angle = init_angle + omega * num_rotations
    return (
        CENTER + orb_r * math.cos(angle),
        CENTER + orb_r * math.sin(angle),
    )


def planet_rotations_after_ticks(obs_step: int, ticks_ahead: int) -> int:
    """Engine planet-rotation count `ticks_ahead` ticks ahead, given current obs.step.

    The engine sets planet absolute angle to `theta0 + omega * obs0.step` (line 542
    of orbit_wars.py). On the very first interpreter call after init, obs0.step=0,
    so that tick applies 0 rotation. Every subsequent tick applies +1 rotation.
    """
    if ticks_ahead <= 0:
        return 0
    if obs_step == 0:
        # First action-processing tick applies 0 rotation; thereafter +1 each.
        return ticks_ahead - 1
    return ticks_ahead


def predict_planet_pos(
    observed_x: float,
    observed_y: float,
    planet_radius: float,
    omega: float,
    steps_ahead: int,
    obs_step: int = 1,
) -> tuple[float, float]:
    """Predict planet position `steps_ahead` ticks from now (observed = step 0 ahead).

    Static planets stay put. Orbiting planets rotate by omega per tick.
    steps_ahead=0 returns observed position. `obs_step` is the current obs.step
    value; defaults to 1 (normal-play case where each tick rotates by +1).
    """
    if not is_orbiting_sim(observed_x, observed_y, planet_radius):
        return (observed_x, observed_y)
    orb_r = math.hypot(observed_x - CENTER, observed_y - CENTER)
    cur_angle = math.atan2(observed_y - CENTER, observed_x - CENTER)
    rotations = planet_rotations_after_ticks(obs_step, steps_ahead)
    new_angle = cur_angle + omega * rotations
    return (
        CENTER + orb_r * math.cos(new_angle),
        CENTER + orb_r * math.sin(new_angle),
    )


def planet_sweep_segment(
    observed_x: float,
    observed_y: float,
    planet_radius: float,
    omega: float,
    ticks_ahead_start: int,
    obs_step: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Swept segment (pos_at_start_of_tick, pos_at_end_of_tick) for a planet.

    During engine processing of a tick, the planet swept segment goes from its
    pre-rotation position to its post-rotation position. `ticks_ahead_start`
    counts ticks-after-now; tick 0 is the launch tick.
    """
    p_old = predict_planet_pos(observed_x, observed_y, planet_radius, omega, ticks_ahead_start, obs_step)
    p_new = predict_planet_pos(observed_x, observed_y, planet_radius, omega, ticks_ahead_start + 1, obs_step)
    return p_old, p_new


def point_to_segment_distance_sim(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Engine line 34-43. Closest point on segment a->b to point p."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    cx, cy = a[0] + t * dx, a[1] + t * dy
    return math.hypot(p[0] - cx, p[1] - cy)


def swept_pair_hit_sim(
    A: tuple[float, float],
    B: tuple[float, float],
    P0: tuple[float, float],
    P1: tuple[float, float],
    r: float,
) -> bool:
    """Engine line 46-64. True iff fleet (A->B) and planet (P0->P1) come within r."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def segment_crosses_sun(
    a: tuple[float, float], b: tuple[float, float], margin: float = 0.0
) -> bool:
    """Continuous: does line segment a->b come within SUN_RADIUS + margin of sun?

    Engine line 607: strict less-than against SUN_RADIUS. We use >= margin
    for safety. Margin defaults to 0 (engine match); pass SUN_SAFETY_MARGIN
    for our launches to avoid round-off-destruction.
    """
    return point_to_segment_distance_sim((CENTER, CENTER), a, b) < SUN_RADIUS + margin


def segment_oob(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Engine line 602: fleet endpoint must be in [0, 100]. We're conservative:
    reject if EITHER endpoint is outside (caller should pass spawn and end-of-tick).
    """
    for x, y in (a, b):
        if not (0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE):
            return True
    return False


def spawn_position(
    planet_x: float, planet_y: float, planet_radius: float, angle: float
) -> tuple[float, float]:
    """Engine line 493-494: fleet spawn point just outside source planet."""
    return (
        planet_x + math.cos(angle) * (planet_radius + SPAWN_OFFSET),
        planet_y + math.sin(angle) * (planet_radius + SPAWN_OFFSET),
    )


def combat_resolve(
    garrison_owner: int,
    garrison_ships: int,
    arrivals: Iterable[tuple[int, int]],
) -> tuple[int, int]:
    """Engine lines 635-674. Returns (new_owner, new_ships).

    arrivals: iterable of (owner, ships) tuples. Sums per-owner, ranks, etc.
    """
    player_ships: dict[int, int] = {}
    for owner, ships in arrivals:
        player_ships[owner] = player_ships.get(owner, 0) + ships
    if not player_ships:
        return garrison_owner, garrison_ships

    sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
    top_owner, top_ships = sorted_players[0]
    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            return garrison_owner, garrison_ships
        survivor_ships = top_ships - second_ships
        survivor_owner = top_owner
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return garrison_owner, garrison_ships

    if survivor_owner == garrison_owner:
        return garrison_owner, garrison_ships + survivor_ships

    new_ships = garrison_ships - survivor_ships
    if new_ships < 0:
        return survivor_owner, -new_ships
    return garrison_owner, new_ships


def min_capture_ships(garrison_ships: int) -> int:
    """Strictly more than garrison required to capture. Engine line 672 (<0)."""
    return garrison_ships + 1


def comet_pos_at(
    comet_group: dict, ticks_ahead: int, comet_idx: int
) -> tuple[float, float] | None:
    """Predict comet position `ticks_ahead` from now.

    comet_group["path_index"] is the CURRENT path index (engine increments by
    1 each tick). ticks_ahead=0 means the position right now.
    Returns None if expired.
    """
    next_idx = comet_group["path_index"] + ticks_ahead
    path = comet_group["paths"][comet_idx]
    if next_idx < 0 or next_idx >= len(path):
        return None
    pt = path[next_idx]
    return (pt[0], pt[1])


def comet_sweep_segment(
    comet_group: dict, ticks_ahead_start: int, comet_idx: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Swept segment (pos_at_start_of_tick, pos_at_end_of_tick) for a comet.

    During engine processing of a tick, comet path_index advances by 1.
    Old_pos is its current location; new_pos is path[idx+1].
    Returns None if the comet has expired (end of path) by tick start, or
    is not yet on the board.
    """
    old = comet_pos_at(comet_group, ticks_ahead_start, comet_idx)
    if old is None:
        return None
    new = comet_pos_at(comet_group, ticks_ahead_start + 1, comet_idx)
    if new is None:
        # Comet expires this tick; engine sets new_pos = old_pos for collision.
        return (old, old)
    return (old, new)
