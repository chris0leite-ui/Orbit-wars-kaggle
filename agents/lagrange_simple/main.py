"""lagrange_simple — agent entry point.

Per turn:
  1. Build World + WorldModel from obs.
  2. Enumerate (src, tgt, launch_tick) candidates with precision physics
     (predict_fleet_fate + B1-B7 orbital safety).
  3. Run a 3-sweep Lagrangian over per-source ship budgets.
  4. Emit moves: [src_id, angle, ships] for each picked candidate.
"""
from __future__ import annotations

import os

# Load-bearing: the +63μ orbitfix pass (B1-B7) gates on this.
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# Kinematic-table cache for predict_fleet_fate (~50-100 ms / turn saving).
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.intent import World
from lib.world_model import WorldModel

from agents.lagrange_simple.dual import solve as solve_dual
from agents.lagrange_simple.score import enumerate_candidates


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0) or 0.0),
    }


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if int(p.owner) == me]
    if not my_planets:
        return []
    if not any(int(p.owner) != me for p in planets):
        return []

    world = World.from_obs(obs_d)
    if os.environ.get("KINEMATIC_TABLE_ENABLED", "1").strip().lower() in (
        "1", "true", "on", "yes",
    ):
        try:
            from lib.kinematic_table import begin_turn as _kt_begin_turn
            _kt_begin_turn(world)
        except Exception:
            pass
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(c) for c in (obs_d.get("comet_planet_ids", []) or []))

    candidates = enumerate_candidates(world, model, me, omega, comet_ids)
    if not candidates:
        return []
    budgets = {int(p.id): int(p.ships) for p in my_planets}
    prods = {int(p.id): int(p.production) for p in my_planets}
    picked = solve_dual(candidates, budgets, prods)
    return [[int(c.src_id), float(c.angle), int(c.ships)] for c in picked]
