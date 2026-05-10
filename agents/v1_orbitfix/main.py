"""Orbit Wars agent v1 — `orbitfix`.

Two-line delta vs the comp-shipped Nearest Planet Sniper:

1. **Orbit-aware aim** (closes ISSUES.md::B.2 + relies on A.1). For each
   target that is currently orbiting (orbital_radius + planet_radius <
   ROTATION_RADIUS_LIMIT) and is NOT a comet, project where it will be
   when our fleet arrives, using the relative-prediction formula and the
   fleet-size-aware speed curve. One fixed-point iteration is enough for
   the inner-planet rotation rates (omega ≤ 0.05 rad/turn, orb_r ≤ 30).
2. **Tie-break randomisation** (closes A.6). When two candidate targets
   sit at equal distance, the shipped baseline picks whichever was
   iterated first — deterministic, and the P0/P1 turn-order asymmetry
   then routes both players toward identical neutrals, giving P0 (lower
   id launches first) the immediate kill and P1 the "free" runner-up.
   Seeding `random.Random(step ^ player_id)` breaks the mirror.

Comets are treated as static for aim purposes here — they move on
elliptical paths, not the rotation formula. A comet-aware lead is
deferred to v2.
"""

from __future__ import annotations

import math
import random

# Importing from `lib.*` works locally and after the bundler inlines lib/
# into a single submission file (see scripts/bundle_agent.py).
from lib.fleet import speed as fleet_speed
from lib.geometry import dist
from lib.orbit import is_orbiting, predict_relative

# Comp-shipped namedtuple — re-exported by the env at runtime.
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def _aim_angle(mine: Planet, target: Planet, ships: int,
               omega: float, is_orbit: bool) -> float:
    """Angle (radians) from `mine` toward where `target` will be at arrival.

    For static targets (or comets, treated as static here) this collapses
    to the shipped baseline's `atan2(dy, dx)` aim. For orbiting targets,
    one fixed-point iteration over (arrival_time, lead_position) is
    sufficient at the env's omega range.
    """
    tx, ty = target.x, target.y
    if is_orbit and omega != 0.0:
        v = fleet_speed(ships)
        # Iterate twice: first lead from current pos, refine using leaded distance.
        for _ in range(2):
            d = math.hypot(tx - mine.x, ty - mine.y)
            t = d / v
            tx, ty = predict_relative(
                [target.id, target.owner, target.x, target.y, target.radius,
                 target.ships, target.production],
                omega, t,
            )
    return math.atan2(ty - mine.y, tx - mine.x)


def agent(obs):
    moves: list[list] = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    omega = float(obs.get("angular_velocity", 0.0)) if isinstance(obs, dict) else float(getattr(obs, "angular_velocity", 0.0))
    comet_ids: set[int] = set()
    raw_comet_ids = obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", [])
    if raw_comet_ids:
        comet_ids = set(int(c) for c in raw_comet_ids)
    step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves

    rng = random.Random(step ^ (player + 1) * 1009)

    for mine in my_planets:
        # Score targets by distance; randomise the order so equal-distance
        # ties don't all collapse onto the same target across players.
        scored: list[tuple[float, float, Planet]] = []
        for t in targets:
            d = dist((mine.x, mine.y), (t.x, t.y))
            scored.append((d, rng.random(), t))
        scored.sort(key=lambda e: (e[0], e[1]))
        nearest = scored[0][2] if scored else None
        if nearest is None:
            continue

        ships_needed = nearest.ships + 1
        if mine.ships < ships_needed:
            continue

        is_orbit = (
            is_orbiting(
                [nearest.id, nearest.owner, nearest.x, nearest.y, nearest.radius,
                 nearest.ships, nearest.production]
            )
            and nearest.id not in comet_ids
        )
        angle = _aim_angle(mine, nearest, ships_needed, omega, is_orbit)
        moves.append([mine.id, angle, ships_needed])

    return moves
