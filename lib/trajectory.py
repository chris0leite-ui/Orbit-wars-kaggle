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
        -> FleetFate(outcome, hit_planet_id, step)

`outcome` is one of `"target"`, `"planet"`, `"sun"`, `"oob"`,
`"timeout"`. The four downstream guards become trivial wrappers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from lib.aim import swept_pair_hit
from lib.fleet import speed as fleet_speed
from lib.geometry import BOARD_SIZE, CENTER, SUN_RADIUS
from lib.orbit import is_orbiting, predict_relative
from lib.world_model import _comet_paths_by_id


def _kinematic_table_enabled() -> bool:
    """Phase γ opt-in: when `KINEMATIC_TABLE_ENABLED=1` AND the
    module-level singleton has been primed via begin_turn(world) AND
    its window is large enough, predict_fleet_fate's position build
    uses the cached lookup instead of re-computing predict_relative
    per (planet, step).
    """
    return os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() in (
        "1", "true", "on", "yes",
    )

# Max steps we simulate before giving up. A 1-ship fleet at speed 1.0
# can cross the 141.4-unit board diagonal in 142 steps; 200 covers
# every realistic case with comfortable margin.
DEFAULT_MAX_STEPS = 200

# Safety margin around the sun (units). The env's sun-check uses
# point-to-segment distance < SUN_RADIUS strict; we MUST match exactly
# or we false-reject trajectories that pass within 10.0-10.5 units of
# centre (the engine accepts them).
#
# Origin: 2026-05-17 Direction A A/B. v4 with this cushion at 0.5 cost
# ~14pp of winrate vs v15 (n=64: 31/64 filter-OFF → 22/64 filter-ON,
# ~10pp difference; the 14pp is including some non-sun rejections).
# The 0.5 cushion was filed in 2026-05-11 as a "float drift cushion"
# but in fact created systematic false-rejections.
SUN_SAFETY = 0.0


@dataclass(frozen=True)
class FleetFate:
    outcome: str               # "target" | "planet" | "sun" | "oob" | "timeout"
    hit_planet_id: int | None  # set when outcome in {"target", "planet"}
    step: int                  # 1-based step at which the event occurred


def predict_fleet_fate(
    src, target, aim_angle: float, ships: int,
    world, max_steps: int = DEFAULT_MAX_STEPS,
    wait_N: int = 0,
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

    `wait_N>0`: the fleet is scheduled to launch wait_N ticks from now.
    Source position, planet positions, and the spawn point are all
    advanced by wait_N orbital ticks before the ray-cast begins. Use this
    when validating wait-then-fire candidates whose fire-time geometry
    differs from the current world snapshot.

    O(max_steps * planets) per call. On a 24-planet mid-game board with
    max_steps=200 that's ~4800 swept_pair_hit calls = ~1-2 ms.
    """
    omega = world.omega

    # Source position at fire time (t + wait_N).
    src_tuple = [src.id, src.owner, src.x, src.y, src.radius,
                 src.ships, src.production]
    if wait_N > 0 and is_orbiting(src_tuple) and omega != 0.0:
        src_x_fire, src_y_fire = predict_relative(src_tuple, omega, wait_N)
    else:
        src_x_fire, src_y_fire = src.x, src.y

    # Spawn position (env: src.center + (radius + 0.1) * direction).
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src_x_fire + cos_a * (src.radius + 0.1)
    spawn_y = src_y_fire + sin_a * (src.radius + 0.1)
    speed_val = fleet_speed(ships)
    if speed_val <= 0:
        # Shouldn't happen (fleet_speed is monotonically >= 1.0 for ships >= 1).
        return FleetFate("oob", None, 0)

    # Pre-compute per-planet positions at every step from t+wait_N onward.
    #
    # COMET HANDLING: comets follow discrete paths from `obs["comets"]`,
    # NOT orbital paths. Predicting them with `predict_relative` is
    # wrong — the prior bug produced 47 OOB events in seed 42 self-play
    # (all post-step-50 when comets enter): fleets aimed at "comet at
    # predicted orbital position" missed the real comet and flew off
    # the board. Look up the comet's actual path and use it; for steps
    # past the path's end, mark the comet as "gone" with sentinel
    # positions far outside the board so swept_pair_hit can't match.
    OFF_BOARD = (-1e6, -1e6)  # sentinel for "comet has left the board"
    # Env semantics (verified against
    # kaggle_environments/envs/orbit_wars/orbit_wars.py lines 480-595):
    # at env step T+1's fleet-movement check, the planet's old_pos is
    # the position from obs T (planet[2], planet[3]) and new_pos is the
    # advanced position. positions[0] is therefore the obs-T position
    # (path[path_index] for comets; predict_relative(.., 0) for orbital);
    # positions[1] is the obs-T+1 position. With wait_N>0 the fleet
    # appears at env step T+1+wait_N and positions[0] = obs-T+wait_N.
    #
    # Phase γ — when KINEMATIC_TABLE_ENABLED=1 and the table is primed
    # for this world AND covers our (wait_N + max_steps) window, the
    # `planet_positions` dict comes from a one-call lookup into the
    # table's per-turn cache. On any miss — env-var off, table not
    # primed, max_lead too small — fall through to the inline build.
    planet_positions = _table_window_or_none(world, wait_N, max_steps + 1)
    if planet_positions is None:
        comet_paths = _comet_paths_by_id(world) if world.comet_ids else {}
        planet_positions = {}
        for pid, p in world.planets_by_id.items():
            if int(pid) in comet_paths:
                # Comet: use its discrete path.
                path, path_index = comet_paths[int(pid)]
                positions: list[tuple[float, float]] = []
                for t in range(max_steps + 1):
                    path_t = int(path_index) + int(wait_N) + t
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        positions.append((float(pt[0]), float(pt[1])))
                    else:
                        positions.append(OFF_BOARD)
                planet_positions[pid] = positions
                continue
            p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            if is_orbiting(p_tuple) and omega != 0.0:
                planet_positions[pid] = [
                    predict_relative(p_tuple, omega, wait_N + t)
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
            # NB: the env DOES check fleet-vs-source at step 0 (see
            # orbit_wars.py L588-597 — no exclusion of from_id). For
            # STATIC sources the geometry handles it: spawn is at
            # `src.center + (radius + 0.1) * direction`, the fleet
            # moves AWAY, swept_pair_hit never matches.
            # For MOVING sources (comets, fast-orbiting planets), the
            # source can catch the fleet within 1 step — that's a real
            # collision the env applies. The earlier `if pid == src_id
            # and step == 0: continue` skip falsely declared "target
            # reached" for drain trajectories whose comet-source
            # caught up and absorbed the fleet (root cause of stranded
            # ships on captured comets).
            p_old = positions[step]
            p_new = positions[step + 1]
            # Comet expiry guard: if EITHER endpoint is the off-board
            # sentinel, the comet has expired during this step — skip
            # the collision check entirely. Without this guard,
            # swept_pair_hit would treat the comet as moving along the
            # huge sentinel-going segment, falsely matching any fleet
            # trajectory (the env, however, removes expired comets from
            # collision resolution — see orbit_wars.py L558-561). This
            # was the cause of the residual seed-13 OOB: fleet aimed at
            # "comet 38 at fleet_step 20" — but the comet's path ended
            # at index 33 (path[14] + 20 == 34), so positions[20] is
            # OFF_BOARD and the swept check produced a phantom hit;
            # the env had no comet there, so the fleet sailed past and
            # exited the board.
            if (p_old[0] < 0 or p_old[0] > BOARD_SIZE
                    or p_old[1] < 0 or p_old[1] > BOARD_SIZE
                    or p_new[0] < 0 or p_new[0] > BOARD_SIZE
                    or p_new[1] < 0 or p_new[1] > BOARD_SIZE):
                continue
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


def _table_window_or_none(world, wait_N: int, length: int):
    """Phase γ: pull positions from the kinematic table when enabled
    and primed for this world. Returns the planet_positions dict the
    inline build would have produced, or None to signal "fall through
    to inline build".

    Bit-parity contract: the table is rebuilt every turn from
    `world.planets_by_id` using the SAME `predict_relative` calls the
    inline build makes (orbital), the SAME `(p.x, p.y)` constants
    (static), and the SAME path lookups (comets).
    """
    if not _kinematic_table_enabled():
        return None
    # Lazy import keeps default-path module-load time unchanged.
    from lib.kinematic_table import get_default
    table = get_default()
    pids = list(world.planets_by_id.keys())
    if not pids:
        return None
    needed_lead = int(wait_N) + int(length) - 1
    if not table.covers(pids, needed_lead):
        return None
    # Sanity: the table must be primed for THIS turn's obs. Trust
    # begin_turn fingerprinting to keep this fresh.
    return table.window(pids, start_offset=int(wait_N), length=int(length))
