"""Full-trajectory ray-cast for fleet path prediction.

Why this exists: every existing path-guard mechanism (`sun_avoid`,
`path_clears_other_planets`, `oob_guard`) only simulates the fleet's
flight UP TO the predicted target arrival point. If the lead-prediction
misses (orbital drift past the predicted spot, tangent shot, target
captured by someone else before we arrive), the fleet doesn't stop —
it keeps flying in a straight line at constant speed until it actually
hits something. The "something" is what the existing guards miss.

Live-replay evidence (audit/live-episodes/52532938/, 21 episodes,
3304 of our fleet disappearances):
    planet_combat  89.3%
    oob             7.5%      <-- predicted endpoint was inside the
    sun             3.2%          board / clear of sun, but actual
    other           0.0%          flight overshot.

This module replaces the endpoint-only check with a full-trajectory
ray-cast that walks forward step by step until the fleet's path
intersects ANY object (target / non-target planet / sun / OOB box edge),
and returns the FIRST hit.

Public API:
    predict_fleet_fate(src, target, aim_angle, ships, world)
        -> FleetFate(outcome, hit_planet_id, step_of_hit)

`outcome` is one of `"target"`, `"planet"`, `"sun"`, `"oob"`,
`"timeout"`. The four downstream guards become trivial wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.aim import swept_pair_hit
from lib.fleet import speed as fleet_speed
from lib.geometry import BOARD_SIZE, CENTER, SUN_RADIUS
from lib.orbit import is_orbiting, predict_relative

# Max steps we simulate before giving up. A 1-ship fleet at speed 1.0
# can cross the 141.4-unit board diagonal in 142 steps; 200 covers
# every realistic case with comfortable margin.
DEFAULT_MAX_STEPS = 200

# Safety margin around the sun (units). The env's sun-check uses
# point-to-segment distance; we add a 0.5-unit cushion so float drift
# on tangent paths doesn't flip the verdict between sim and reality.
SUN_SAFETY = 0.5


@dataclass(frozen=True)
class FleetFate:
    outcome: str               # "target" | "planet" | "sun" | "oob" | "timeout"
    hit_planet_id: int | None  # set when outcome in {"target", "planet"}
    step: int                  # 1-based step at which the event occurred


def predict_fleet_fate(
    src, target, aim_angle: float, ships: int,
    world, max_steps: int = DEFAULT_MAX_STEPS,
) -> FleetFate:
    """Ray-cast a fleet's full trajectory until the first collision.

    Walks the fleet forward at `fleet_speed(ships)` per step, checking
    EACH per-step segment against:

    1. The sun (continuous point-to-segment distance to (CENTER, CENTER),
       with `SUN_SAFETY` cushion).
    2. The OOB box edges (segment endpoint outside [0, BOARD_SIZE]).
    3. Every planet's per-step swept segment (orbital chord for orbiting
       planets, constant position for static). Uses the env-mirroring
       `swept_pair_hit` so the prediction matches the env's collision
       resolution.

    Returns the FIRST hit. If we reach `max_steps` without collision the
    fate is `"timeout"` — should be rare on a 100x100 board.

    O(max_steps * planets) per call. On a 24-planet mid-game board with
    max_steps=200 that's ~4800 swept_pair_hit calls = ~1-2 ms.

    Env-parity note: the env's interpreter rotates planets by
    `omega * obs.step`, so at `obs.step == 0` the first tick keeps
    planets stationary (rotation count 0). At every subsequent
    `obs.step >= 1` the agent's current planet position already
    absorbs the `(N-1)*omega` rotation that prior ticks applied, so
    `predict_relative(current, omega, t)` correctly gives the
    position `t` ticks ahead. At `obs.step == 0`, however, applying
    the same formula over-rotates by one step; the env's first tick
    will be stationary while this function would otherwise advance
    the planet by `omega`. The `rot_offset` shift below mirrors
    `lib/trajectory_layer.py::_effective_t_for_orbital` so both
    code paths agree at game start. `world.step` is read via
    `getattr(..., 1)` so callers passing mock worlds without a step
    attribute get the `obs.step >= 1` behaviour.
    """
    omega = world.omega
    obs_step = int(getattr(world, "step", 1))
    rot_offset = 1 if obs_step == 0 else 0

    # Spawn position (env: src.center + (radius + 0.1) * direction).
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src.x + cos_a * (src.radius + 0.1)
    spawn_y = src.y + sin_a * (src.radius + 0.1)
    speed_val = fleet_speed(ships)
    if speed_val <= 0:
        # Shouldn't happen (fleet_speed is monotonically >= 1.0 for ships >= 1).
        return FleetFate("oob", None, 0)

    # Pre-compute per-planet positions at every step (orbital chord).
    planet_positions: dict[int, list[tuple[float, float]]] = {}
    for pid, p in world.planets_by_id.items():
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and omega != 0.0:
            planet_positions[pid] = [
                predict_relative(p_tuple, omega, max(t - rot_offset, 0))
                for t in range(max_steps + 1)
            ]
        else:
            planet_positions[pid] = [(p.x, p.y)] * (max_steps + 1)

    target_id = target.id
    src_id = src.id
    for step in range(max_steps):
        fleet_old = (
            spawn_x + cos_a * speed_val * step,
            spawn_y + sin_a * speed_val * step,
        )
        fleet_new = (
            spawn_x + cos_a * speed_val * (step + 1),
            spawn_y + sin_a * speed_val * (step + 1),
        )

        # 1. Sun check — point-to-segment distance.
        sun_d = _segment_to_point_distance(fleet_old, fleet_new, (CENTER, CENTER))
        if sun_d < SUN_RADIUS + SUN_SAFETY:
            return FleetFate("sun", None, step + 1)

        # 2. OOB check — segment endpoint outside the box.
        if (
            fleet_new[0] < 0.0 or fleet_new[0] > BOARD_SIZE
            or fleet_new[1] < 0.0 or fleet_new[1] > BOARD_SIZE
        ):
            return FleetFate("oob", None, step + 1)

        # 3. Planet collision — swept_pair_hit against every planet.
        for pid, positions in planet_positions.items():
            # Spawn-step skip: env explicitly does not collide a fresh
            # fleet with its source planet on its first move.
            if pid == src_id and step == 0:
                continue
            p_old = positions[step]
            p_new = positions[step + 1]
            prad = world.planets_by_id[pid].radius
            if swept_pair_hit(fleet_old, fleet_new, p_old, p_new, prad):
                outcome = "target" if pid == target_id else "planet"
                return FleetFate(outcome, pid, step + 1)

    return FleetFate("timeout", None, max_steps)


def _segment_to_point_distance(a, b, p) -> float:
    """Shortest distance from segment a->b to point p."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)
