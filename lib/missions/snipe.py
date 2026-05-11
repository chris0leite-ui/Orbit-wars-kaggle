"""Snipe mission builder — capture enemy / neutral planets via cost-aware ROI.

For every (our-planet, non-our-planet) pair, produce one Mission candidate.
**2026-05-11 ROI upgrade**: the score now trades off VALUE against COST
in ships (and travel time), addressing the gap the doc flagged
(`docs/strategies/simple-roi.md` "Where ROI can lose" lines 64-69):

    value = production × max(1, 500 - step - eta)
    score = value / (ships_to_send + distance + 1)

Additive (not multiplicative) cost in the denominator: pure value/cost
over-corrects toward 1-ship 1-prod targets, which is a different bug.
Keeping distance in the denominator preserves the travel-time discount.

Filter: drop pairs where the WorldModel predicts the target will already
be ours with surplus garrison at our fleet's arrival step.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

# Total game length in steps (Configuration table, data/README.md).
EPISODE_STEPS = 500


def propose_snipe_missions(world: World, model: WorldModel) -> list[Mission]:
    """Build one snipe Mission per (our source, non-our target) pair."""
    if not world.planets_by_id:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            base_ships = max(1, int(t.ships) + 1)
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                # Target will be ours with surplus garrison; redundant.
                continue
            # Cost-aware ROI: value / (cost + distance + 1). Additive cost
            # avoids the pure-value/cost over-correction toward 1-ship targets.
            time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold
            score = value / (base_ships + d + 1.0)
            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions
