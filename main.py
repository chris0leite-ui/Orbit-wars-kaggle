"""main.py — our Orbit Wars agent (kaggle-submittable as-is).

Starter = the comp's shipped baseline ("Nearest Planet Sniper"), kept
deliberately minimal so we can build a real strategy from scratch.

Submit with:  ./submit.sh "message describing the change"
Eval with:    python eval.py --vs nearest          (quick smoke)
              python eval.py --panel               (3-opponent panel, n=24)

The submission entrypoint must be a callable named `agent` at module
level. Multi-file submissions are also fine — tar.gz with main.py at the
root — but until we need more than one file, keep it here.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(
            targets,
            key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y),
        )
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
