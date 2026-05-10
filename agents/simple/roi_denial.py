"""roi_denial — pure spoiler: only attack enemy-owned targets.

V5d in the strategy A/B test. Score is the same margin EV as roi_margin
(`m * P * H - S_needed`) BUT neutral targets are excluded entirely; only
enemy-owned planets are eligible. Sizing is `target.ships + 1` (no
internal gate; gating is V3's axis).

This will likely lose standalone — neutrals are usually the lowest-cost
captures early-game — but the magnitude of its loss vs roi_margin tells
us how much of any margin lift comes from neutral-with-zero-sum-weight
vs from enemy-only denial.
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
    enemy_targets = [
        p for p in planets if p.owner != player and p.owner != -1
    ]
    if not my_planets or not enemy_targets:
        return []

    rng = random.Random(step ^ (player + 1) * 1009)
    intents: list[Intent] = []
    for mine in my_planets:
        scored = [(_score(mine, t, step, player), rng.random(), t) for t in enemy_targets]
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
