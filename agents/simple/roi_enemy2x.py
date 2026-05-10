"""roi_enemy2x — ROI with enemy-target zero-sum bonus.

V5a in the strategy A/B test. Score multiplies `target.production` by
the margin multiplier `m`: 1 for neutral, 2 for enemy. Self-targets are
filtered out upstream by `targets = [p for p in planets if p.owner != player]`.

Hypothesis: enemy captures contribute 2x to margin (we gain P/turn AND
deny them P/turn). Scaling the numerator by `m` is the cheapest fix to
encode this, isolated from the horizon-decay axis.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.scoring import eta_proxy, margin_multiplier, projected_garrison


def _score(mine: Planet, target: Planet, my_id: int) -> tuple:
    eta = eta_proxy(mine, target)
    g = projected_garrison(target, eta)
    d = dist((mine.x, mine.y), (target.x, target.y))
    m = margin_multiplier(target, my_id)
    roi = (m * target.production) / (g + d + 1.0)
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
        scored = [(_score(mine, t, player), rng.random(), t) for t in targets]
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
