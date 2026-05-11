"""roi — target by COST-AWARE return on investment.

HYPOTHESIS (revised 2026-05-11): the score must capture both VALUE
(production × time we'll hold the planet) and COST (ships we spend to
capture). The original `production / (distance + 1)` ignored cost
entirely; a production-5 planet with 99 ships outranked three
production-3 planets with 10 ships even though the latter trio yields
more total ships per invested ship (`docs/strategies/simple-roi.md`
"Where ROI can lose" lines 64-69).

Score (additive cost so it doesn't dominate ranking the way pure
value/cost does — pure value/cost picks 1-ship 1-prod targets over
20-ship 5-prod targets, which over-corrects):

    value = production × max(1, 500 - step - eta)
    score = value / (ships_to_send + distance + 1)

`eta` uses `fleet_speed(cost)` for the launched fleet size. Tiebreaker:
distance ascending.
"""

from __future__ import annotations

import math
import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.fleet import speed as fleet_speed
from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS

EPISODE_STEPS = 500


def _score(mine: Planet, target: Planet, step: int,
           comet_lifetime: int | None) -> tuple:
    d = dist((mine.x, mine.y), (target.x, target.y))
    cost = max(1, int(target.ships) + 1)
    v = fleet_speed(cost)
    eta = int(math.ceil(d / max(v, 1e-6)))
    # Comets leave the board on a fixed schedule; if the comet exits
    # before our fleet arrives, time_to_hold collapses to 0 and the
    # score correctly drops to 0 (don't send to dying comets).
    if comet_lifetime is not None:
        time_to_hold = max(0, comet_lifetime - eta)
    else:
        time_to_hold = max(1, EPISODE_STEPS - step - eta)
    value = target.production * time_to_hold
    roi = value / (cost + d + 1.0)
    return (-roi, d)   # argmax roi, tiebreak: nearest


def propose_intents(obs) -> list[Intent]:
    # Local import (not module-top) so simple/roi.py stays bundlable —
    # bundle_agent.DEFAULT_LIB_ORDER inlines world_model before
    # mechanism, so this resolves correctly post-bundle.
    from lib.intent import World
    from lib.world_model import comet_remaining_lifetime

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    step = (
        int(obs.get("step", 0))
        if isinstance(obs, dict)
        else int(getattr(obs, "step", 0))
    )

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    world = World.from_obs(obs)  # for comet-lifetime lookup
    rng = random.Random(step ^ (player + 1) * 1009)
    intents: list[Intent] = []
    for mine in my_planets:
        scored = []
        for t in targets:
            lifetime = (
                comet_remaining_lifetime(t.id, world)
                if t.id in world.comet_ids else None
            )
            scored.append((_score(mine, t, step, lifetime), rng.random(), t))
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
