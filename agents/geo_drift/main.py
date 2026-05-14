"""geo_drift — geo v3.1 + drift-aware score discount.

Hypothesis: orbiting planets captured WHILE in our Voronoi cell can
drift INTO opponent-dominated areas, becoming indefensible. Geo's
scoring uses static current-step distance/Voronoi, so we capture
planets we can't hold.

Mechanism: per turn, precompute `hold_prob[pid]` = fraction of the next
HOLD_HORIZON turns where target `pid` (predicted forward via
`lib.orbit.predict_relative`) is closer to our nearest-friendly than to
any enemy planet. Static planets get hold_prob=1.0 by construction.

Apply by rescaling each base Mission's score by hold_prob BEFORE
settle_plan picks. K=10 lookahead validates the re-ranked candidate.

Single-axis variant: NO other change vs agents/geo/main.py.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from lib.intent import World
from lib.mission import Mission
from lib.orbit import predict_relative, is_orbiting

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "geo_base_for_drift", _REPO / "agents" / "geo" / "main.py",
)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)

HOLD_HORIZON = 25     # turns; matches median capture-lifetime in losses
SAMPLE_STEPS = 5      # number of probe points in HOLD_HORIZON
MIN_DISCOUNT = 0.2    # floor; never zero a candidate out entirely

_geo_orig_build = _geo._build_base_missions


def _hold_prob_map(world: World) -> dict[int, float]:
    """For each planet, fraction of next HOLD_HORIZON turns it sits
    closer to our nearest friendly than to any enemy planet."""
    my_id = world.my_id
    omega = world.omega
    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if p.owner == my_id]
    enemy_planets = [
        p for p in planets
        if p.owner != my_id and p.owner != -1
        and p.id not in world.comet_ids
    ]
    if not my_planets:
        return {p.id: 1.0 for p in planets}

    sample_ts = [
        HOLD_HORIZON * (i / max(1, SAMPLE_STEPS - 1))
        for i in range(SAMPLE_STEPS)
    ]

    def positions_at(group, t):
        out = []
        for p in group:
            tup = (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            if is_orbiting(tup):
                x, y = predict_relative(tup, omega, t)
            else:
                x, y = p.x, p.y
            out.append((x, y))
        return out

    my_pos_t = [positions_at(my_planets, t) for t in sample_ts]
    en_pos_t = (
        [positions_at(enemy_planets, t) for t in sample_ts]
        if enemy_planets else None
    )

    out: dict[int, float] = {}
    for p in planets:
        tup = (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
        in_cell = 0
        for i, t in enumerate(sample_ts):
            if is_orbiting(tup):
                tx, ty = predict_relative(tup, omega, t)
            else:
                tx, ty = p.x, p.y
            d_us = min(
                math.hypot(tx - fx, ty - fy) for fx, fy in my_pos_t[i]
            )
            if en_pos_t and en_pos_t[i]:
                d_them = min(
                    math.hypot(tx - ex, ty - ey)
                    for ex, ey in en_pos_t[i]
                )
            else:
                d_them = float("inf")
            if d_us <= d_them:
                in_cell += 1
        hp = in_cell / max(1, len(sample_ts))
        out[p.id] = max(MIN_DISCOUNT, hp)
    return out


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    hp_map = _hold_prob_map(world)

    def patched_build(world_, model_):
        missions = _geo_orig_build(world_, model_)
        out = []
        for m in missions:
            hp = hp_map.get(m.target_id, 1.0)
            if hp >= 0.999 or m.target_id in world_.comet_ids:
                out.append(m)
            else:
                out.append(Mission(
                    mission_class=m.mission_class,
                    src_id=m.src_id, target_id=m.target_id,
                    ships=m.ships, score=m.score * hp,
                    eta=m.eta, note=m.note,
                ))
        return out

    _geo._build_base_missions = patched_build
    try:
        return _geo.agent(obs, configuration)
    finally:
        _geo._build_base_missions = _geo_orig_build
