"""roi_combo — V6: roi_enemy2x + sun-aware target pivot + sun_avoid mechanism.

V6 in the strategy A/B test. Combines the two axes that had positive
lift at 8 seeds:

- Score = m * production / (projected_garrison + dist + 1) where m is
  the margin multiplier (1 neutral, 2 enemy). From V5a (`roi_enemy2x`,
  98.4% mean panel WR vs ROI's 85.9%).
- Strategy-side fallback: walk the ranked target list and skip any
  whose direct path crosses the sun. From V4 (`roi_safe`, 74% mean WR;
  the pivot itself is good even though `roi_safe` lost head-to-head to
  ROI standalone).
- Mechanism stack adds `sun_avoid` as a final guard for late-game cases
  where the pivoted target drifts behind the sun by arrival.

Sizing remains `target.ships + 1` (the mechanism layer's `arrival_size`
inflates to the production-aware S_needed). No internal dominance gate;
that axis was net-negative at 8 seeds.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist, path_clears_sun
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS, sun_avoid
from lib.scoring import eta_proxy, margin_multiplier, projected_garrison

MECHANISMS = list(DEFAULT_MECHANISMS) + [sun_avoid]


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
