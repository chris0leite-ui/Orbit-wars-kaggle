"""v1 orbitfix — orbit-aware aim + tiebreak randomisation.

Strategic write-up: docs/strategies/v1_orbitfix.md.

Implementation split (post Step 3.5.A refactor):
- `propose_intents(obs)` — the strategy: score targets by distance, break
  ties with a per-turn rng salted by player_id (closes A.6), emit one
  Intent per owned planet pointing at its nearest non-owned planet.
- `agent(obs) = realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)`
  — the mechanism layer (lib/mechanism.py) drops invalid intents and
  populates the orbit-aware aim angle. Adding more mechanisms (arrival_size,
  comet_aim, sun_avoid) lifts v1 to v1.1 without a strategy change — see
  the plan's Step 3.5.B/C/D.
"""

from __future__ import annotations

import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import dist
from lib.intent import Intent, realize
from lib.mechanism import DEFAULT_MECHANISMS


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
        scored: list[tuple[float, float, Planet]] = []
        for t in targets:
            d = dist((mine.x, mine.y), (t.x, t.y))
            scored.append((d, rng.random(), t))
        scored.sort(key=lambda e: (e[0], e[1]))
        nearest = scored[0][2]
        intents.append(
            Intent(
                src_id=mine.id,
                target_id=nearest.id,
                ships=nearest.ships + 1,
            )
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
