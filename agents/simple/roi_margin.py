"""roi_margin — full margin-EV scoring (pure score axis).

V5c in the strategy A/B test. Score is the deterministic margin
contribution of capturing the target:

    score = m * P * H - S_needed

where m is the owner-flip multiplier (1 neutral, 2 enemy), P the
target's production, H the remaining-turns horizon, S_needed the
minimum fleet to capture at arrival. Argmax over targets; sizing
remains `target.ships + 1` (the mechanism layer's `arrival_size` will
inflate to S_needed). No dominance gate here — that's V3's job;
combining the two axes in one file confounded the ablation in the
8-seed smoke (roi_margin went 0/16 because the gate held all garrison).

This isolates the score-axis reshape so its lift is attributable to
the score function alone.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.scoring import (
    eta_proxy,
    horizon,
    margin_multiplier,
    s_needed,
)


def _score(mine: Planet, target: Planet, step: int, my_id: int) -> tuple:
    eta = eta_proxy(mine, target)
    h = horizon(step, eta)
    m = margin_multiplier(target, my_id)
    need = s_needed(target, eta)
    margin_ev = m * target.production * h - need
    return (-margin_ev,)


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
        scored = [(_score(mine, t, step, player), rng.random(), t) for t in targets]
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
