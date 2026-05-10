"""roi_discounted_cap — hard ETA cutoff on top of roi_arrival.

V2 (hard cap) in the strategy A/B test. Targets with eta > ETA_CAP are
discarded entirely. ETA_CAP = 30 turns ~ 6% of a 500-turn game; far
enough that planets routinely change hands during the flight, so we
just don't go.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.scoring import eta_proxy, projected_garrison

ETA_CAP: int = 30


def _score(mine: Planet, target: Planet) -> tuple:
    eta = eta_proxy(mine, target)
    g = projected_garrison(target, eta)
    d = dist((mine.x, mine.y), (target.x, target.y))
    roi = target.production / (g + d + 1.0)
    return (-roi, d), eta


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
        scored = []
        for t in targets:
            (key, eta) = _score(mine, t)
            if eta > ETA_CAP:
                continue
            scored.append((key, rng.random(), t))
        if not scored:
            continue
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
