"""roi_baseline — frozen pre-physics-upgrade ROI for A/B against the new stack.

Identical to `agents/simple/roi.py` in target-selection (production / distance),
but pins `realize(..., mechanisms=DEFAULT_MECHANISMS_PRE_PHYSICS)` so we can
measure the lift contributed by Block A (5-iter aim + sun_avoid arrival-aware
+ path_clears_other_planets + oob_guard) directly on the local panel.

This is NOT a submission candidate. It exists only as the local control arm
for the strategy_panel A/B.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS_PRE_PHYSICS


def _score(mine: Planet, target: Planet) -> tuple:
    d = dist((mine.x, mine.y), (target.x, target.y))
    roi = target.production / (d + 1.0)
    return (-roi, d)


def propose_intents(obs) -> list[Intent]:
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

    rng = random.Random(step ^ (player + 1) * 1009)
    intents: list[Intent] = []
    for mine in my_planets:
        scored = [(_score(mine, t), rng.random(), t) for t in targets]
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS_PRE_PHYSICS)
