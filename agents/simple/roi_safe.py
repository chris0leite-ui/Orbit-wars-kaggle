"""roi_safe — sun-aware ROI with strategy-side fallback.

V4 in the strategy A/B test. Score is roi_arrival's projected-garrison
ROI; sort targets; iterate down the sorted list and pick the first
whose direct path from `mine` clears the sun. This sidesteps the prior
sun_avoid ablation failure, which was a no-pivot deadlock — nearest-
greedy got stuck on sun-blocked targets and dropped every intent. ROI
naturally has alternatives ranked behind the blocked one.

Mechanism stack adds the central-pipeline `sun_avoid` as the last
stage as a safety net for residual fail-cases (e.g. an orbiting target
that drifts behind the sun by arrival).
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist, path_clears_sun
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS, sun_avoid
from lib.scoring import eta_proxy, projected_garrison

# Strategy-side mechanism stack: DEFAULT + sun_avoid as the final guard.
MECHANISMS = list(DEFAULT_MECHANISMS) + [sun_avoid]


def _score(mine: Planet, target: Planet) -> tuple:
    eta = eta_proxy(mine, target)
    g = projected_garrison(target, eta)
    d = dist((mine.x, mine.y), (target.x, target.y))
    roi = target.production / (g + d + 1.0)
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
        # Walk the ranked list; pick the first whose direct path clears the sun.
        chosen: Planet | None = None
        for _key, _r, t in scored:
            if path_clears_sun((mine.x, mine.y), (t.x, t.y), safety=1.0):
                chosen = t
                break
        if chosen is None:
            continue
        intents.append(
            Intent(src_id=mine.id, target_id=chosen.id, ships=chosen.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=MECHANISMS)
