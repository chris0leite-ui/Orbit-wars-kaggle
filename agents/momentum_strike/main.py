"""momentum_strike — production-first expansion with CCW tie-break + defense.

V2 architecture (rewritten 2026-05-24 after V1's per-source rolled aim
lost 7/8 to `nearest`):

  - Defense pass first: reinforce planets predicted to flip.
  - Expansion: production-first target selection, CCW tie-breaker.
  - All emissions route through `lib.intent.realize` with
    `DEFAULT_MECHANISMS`, getting auto-aim, auto-sizing, sun avoidance,
    OOB protection, and path-clears-planets for free.

V1's hand-rolled aim + `predict_fleet_fate` was the proximate cause of
the regression vs. `nearest`: every micro-tuning iteration left
mechanisms on the table. V2 inherits the same mechanism stack
`agents/simple/production` and `agents/simple/nearest` use.

Phase logic (EXPAND/STRIKE) and synchronized salvo are deferred to V3
once V2 is verified to beat the simple baselines.
"""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.world_model import WorldModel

from agents.momentum_strike.proposer import propose_defense, propose_expand

DEBUG = os.environ.get("MOMENTUM_DEBUG", "0") == "1"


def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
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
    enemy_planets = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]
    neutrals = [p for p in planets if int(p.owner) == -1]

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)

    used: set[int] = set()
    intents: list[Intent] = []

    # 1. Defense — reinforce any of my planets predicted to flip.
    intents.extend(propose_defense(my_planets, world, model, me, used))

    # 2. Expansion — production-first, CCW tie-break.
    intents.extend(propose_expand(
        my_planets, neutrals, enemy_planets, world, model, me, used,
    ))

    if DEBUG and (obs_d.get("step", 0) % 20 == 0 or len(intents) > 0):
        step = obs_d.get("step", 0)
        print(f"[momentum] step={step} my_p={len(my_planets)} "
              f"intents={len(intents)} "
              f"(def={sum(1 for i in intents if 'defense' in i.note)}, "
              f"exp={sum(1 for i in intents if i.note=='expand')})",
              flush=True)

    return realize(intents, obs_d, mechanisms=DEFAULT_MECHANISMS, model=model)
