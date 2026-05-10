"""roi_dominance_120 — dominance gate, alpha = 1.20.

V3 in the strategy A/B test. Score is roi_arrival's projected-garrison
ROI; argmax target. Then the gate: only fire if `mine.ships >= ALPHA *
S_needed`, otherwise return no intent for this source — ships
accumulate naturally turn-over-turn (cheap implicit bundling).

ALPHA = 1.20 — modest overkill threshold.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.scoring import eta_proxy, projected_garrison, s_needed

ALPHA: float = 1.20


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
            key, eta = _score(mine, t)
            scored.append((key, eta, rng.random(), t))
        scored.sort(key=lambda e: (e[0], e[2]))
        for _key, eta, _r, t in scored:
            need = s_needed(t, eta)
            send = int(round(ALPHA * need))
            if mine.ships >= send:
                intents.append(
                    Intent(src_id=mine.id, target_id=t.id, ships=send)
                )
                break
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
