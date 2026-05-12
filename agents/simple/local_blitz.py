"""local_blitz — heavily distance-penalised, all-in local conquest.

Strategy in plain English (PI direction, 2026-05-12):

- Heavily penalise long flights: target score = production × time_left
  × exp(-d / LENGTH_SCALE) / (cost + 1). Targets near the source dominate
  the ranking; an enemy across the board is essentially invisible.
- Empty every source planet every turn (garrison floor = 0). If we launch,
  we launch `src.ships` — the whole garrison.
- If no nearby enemy/neutral is affordable from this source, fall back to
  reinforcing the friendly planet closest to the enemy front (forward-stage
  the ships toward the action).
- Every source with ships > 0 emits an intent every turn — never sit idle.

Mechanism stack: `[validate, lead_aim_v2, sun_avoid, path_clears_other_planets,
oob_guard]`. NOTE the absence of `arrival_size`: that mechanism bumps
intent.ships above the source garrison to cover production growth during
flight, which then triggers `validate`'s `ships > src.ships` drop and the
source goes idle. The growth-aware filter below already happens in the
proposer (we just refuse to propose unaffordable attacks), so the
mechanism-layer bump is redundant and counterproductive here.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World, realize
from lib.mechanism import (
    lead_aim_v2,
    oob_guard,
    path_clears_other_planets,
    sun_avoid,
    validate,
)

# exp(-d/L): d=15 keeps 37%, d=30 keeps 14%, d=45 keeps 5%.
LENGTH_SCALE = 15.0
# Ships kept at the source after every launch. 0 = empty source every turn
# (the PI's original direction). Variants set higher floors (5, 10) to test
# whether reserving a small defender garrison saves enough planets to win
# more games against ROI-class agents.
GARRISON_FLOOR = 0
EPISODE_STEPS = 500

MECHANISMS = [
    validate,
    lead_aim_v2,
    sun_avoid,
    path_clears_other_planets,
    oob_guard,
]


def propose_intents(obs) -> list[Intent]:
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if p.owner == world.my_id]
    if not my_planets:
        return []
    enemies = [p for p in planets if p.owner != world.my_id]
    step = world.step
    time_left = max(1, EPISODE_STEPS - step)

    intents: list[Intent] = []
    for src in my_planets:
        # Available ships to launch = garrison above the floor. If we don't
        # have at least 1 ship over the floor, skip this source entirely.
        available = int(src.ships) - GARRISON_FLOOR
        if available <= 0:
            continue

        # 1. Attack: best affordable enemy/neutral by exp-weighted ROI.
        v = fleet_speed(available)
        best_target = None
        best_score = -1.0
        for t in enemies:
            d = math.hypot(t.x - src.x, t.y - src.y)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            if t.owner == -1:
                cost = max(1, int(t.ships) + 1)
            else:
                # Enemy: account for production growth during flight.
                cost = max(1, int(t.ships) + int(t.production) * eta + 1)
            if cost > available:
                continue
            value = t.production * max(1, time_left - eta)
            score = value * math.exp(-d / LENGTH_SCALE) / (cost + 1.0)
            if score > best_score:
                best_score = score
                best_target = t

        if best_target is not None:
            intents.append(Intent(
                src_id=src.id,
                target_id=best_target.id,
                ships=available,
                note="attack",
            ))
            continue

        # 2. Reinforce fallback: friendly planet closest to enemy front.
        allies = [p for p in my_planets if p.id != src.id]
        if not allies:
            continue
        if enemies:
            best_ally = min(
                allies,
                key=lambda a: min(
                    math.hypot(a.x - e.x, a.y - e.y) for e in enemies
                ),
            )
        else:
            best_ally = min(
                allies,
                key=lambda a: math.hypot(a.x - src.x, a.y - src.y),
            )
        intents.append(Intent(
            src_id=src.id,
            target_id=best_ally.id,
            ships=available,
            note="reinforce",
        ))
    return intents


def agent(obs):
    # Re-anchor module constants on every call so variants in agents/simple/
    # that import this module and mutate them can't leak across calls in a
    # re-used multiprocessing worker (see friction 2026-05-12:
    # module-mutation-patching-has-worker-reuse-race).
    global LENGTH_SCALE, GARRISON_FLOOR
    LENGTH_SCALE = 15.0
    GARRISON_FLOOR = 0
    return realize(propose_intents(obs), obs, mechanisms=MECHANISMS)
