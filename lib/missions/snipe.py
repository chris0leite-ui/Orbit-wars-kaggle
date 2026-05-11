"""Snipe mission builder — capture enemy / neutral planets via straight ROI.

For every (our-planet, non-our-planet) pair, produce one Mission candidate
with score = `target.production / (dist + 1.0)` (the same ROI metric the
v1.2/roi / v2 strategies use, now exposed as a typed mission).

Filter: drop pairs where the WorldModel predicts the target will already
be ours with surplus garrison at our fleet's arrival step (the
don't-double-commit rule). Identical logic to the v2 strategy-level
filter — moved here so the planner sees only viable candidates.

Pure function of `(world, model)`. Both are constructed once per turn by
the agent entry point; this builder is O(my_planets * enemy_planets).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel


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
            score = t.production / (d + 1.0)
            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions
