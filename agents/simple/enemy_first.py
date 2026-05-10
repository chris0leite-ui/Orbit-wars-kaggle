"""enemy_first — prefer enemy-owned targets; fall back to nearest neutral.

HYPOTHESIS: pressure-on-the-opponent matters more than economy. Once the
opponent has any planet, every turn we leave their production untouched
they out-grow us; capturing one of theirs both adds to ours and subtracts
from theirs (a 2-sided swing). Until contact (no enemy planets in the
target list), it degenerates to nearest-greedy.

Score: (is_neutral, distance). Enemy planets (owner ≠ -1, owner ≠ me)
sort first, neutrals second; within each group, nearest wins.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS


def _score(mine: Planet, target: Planet) -> tuple:
    is_neutral = 1 if target.owner == -1 else 0
    return (is_neutral, dist((mine.x, mine.y), (target.x, target.y)))


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
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
