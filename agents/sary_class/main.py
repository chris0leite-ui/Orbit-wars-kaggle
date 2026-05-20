"""sary_class — aggressive panel anchor for local A/B.

Stand-alone myopic agent that targets a sary-class emission cadence
(target ~1.5 launches/turn vs simple-agents' ~1.0). Used as a panel
anchor: a candidate that loses to sary_class while baseline doesn't
has the under-emission failure that motivated this build.

Strategy: only fire from well-armed sources (≥ MIN_FIRE_SHIPS); each
firing source sends one launch sized to overwhelm the target's
defenders + production accrual + holding buffer. Multi-source firing
per turn (no per-turn cap on the agent) is what produces a sary-class
cadence even though each source emits ≤ 1/turn.

Plan: /root/.claude/plans/so-now-research-and-zany-widget.md
"""

from __future__ import annotations

import math
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.aim import aim_orbiting
from lib.fleet import speed as fleet_speed

# Tunable knobs.
MIN_RESERVE = 3             # never drain a planet below this
MIN_FIRE_SHIPS = 8          # don't fire from sources below this; let them accumulate
ATTACK_BUFFER = 4           # send target.ships + this many extra (no production-flight term)
MIN_TARGET_PRODUCTION = 1
TOP_K_TARGETS = 6           # consider top K ROI targets per src; pick first affordable


def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []

    omega = float(obs_d.get("angular_velocity", 0.0))
    comet_ids = set(obs_d.get("comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if int(p.owner) == me
                  and int(p.ships) >= MIN_FIRE_SHIPS]
    targets = [
        p for p in planets
        if int(p.owner) != me
        and int(p.production) >= MIN_TARGET_PRODUCTION
        and int(p.id) not in comet_ids
    ]
    if not my_planets or not targets:
        return []

    moves: list[list] = []

    for src in my_planets:
        src_xy = (float(src.x), float(src.y))
        src_r = float(src.radius)
        available = int(src.ships) - MIN_RESERVE
        if available < ATTACK_BUFFER:
            continue

        # Score targets by production / distance² (ROI-like).
        scored: list[tuple[float, Planet, float]] = []
        for tgt in targets:
            dx = float(tgt.x) - float(src.x)
            dy = float(tgt.y) - float(src.y)
            d2 = dx * dx + dy * dy
            if d2 <= 0.0:
                continue
            score = float(tgt.production) / d2
            scored.append((score, tgt, math.sqrt(d2)))
        if not scored:
            continue
        scored.sort(key=lambda e: -e[0])

        # Pick the first affordable target in score order. One launch
        # per source per turn — selectivity is what keeps cadence at
        # ~1.5/turn (depends on how many srcs are armed per turn).
        # Nuke variant: pick the top-scoring target, send all available
        # ships. Bigger fleets land harder. If target dies, we hold it
        # with the residue + production accrual. If we bounce, target
        # is dented + we drain — but we've made our move with full force.
        # This is the "ambitious aggressive" shape sary-class needs to
        # be a real panel anchor.
        for score, tgt, dist in scored[:TOP_K_TARGETS]:
            tgt_tuple = (
                int(tgt.id), int(tgt.owner),
                float(tgt.x), float(tgt.y), float(tgt.radius),
                int(tgt.ships), int(tgt.production),
            )
            ships_to_send = available
            try:
                aim = aim_orbiting(
                    src_xy, src_r, tgt_tuple, float(tgt.radius),
                    ships_to_send, omega,
                )
            except Exception:
                continue
            if aim is None:
                continue
            angle, _arrival, _eta = aim
            moves.append([int(src.id), float(angle), int(ships_to_send)])
            break  # one launch per src per turn — drain to a single target

    return moves
