"""v3.5.1 ablation: end-game ship burn at step >= 470.

At step 470+, snipe is overridden by a pure-greedy burn: every owned
planet with > 1 ship sends `src.ships - 1` ships at the nearest
non-owned planet. Ships in flight count toward final_score at step
500, so leaving garrisons home wastes them.

Before step 470, behaves identically to v3_snipe.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel

BURN_STEP = 470          # steps 470..499 are burn window
MIN_BURN_GARRISON = 2    # only burn sources with > this many ships


def propose_endgame_burn(world: World, model: WorldModel) -> list[Mission]:
    """At step >= BURN_STEP: every owned planet sends src.ships - 1 at
    its nearest non-owned planet, ignoring ROI."""
    if int(world.step) < BURN_STEP:
        return []
    my_planets = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.ships > MIN_BURN_GARRISON
    ]
    if not my_planets:
        return []
    targets = [p for p in world.planets_by_id.values() if p.owner != world.my_id]
    if not targets:
        return []

    missions: list[Mission] = []
    for src in my_planets:
        # Nearest non-owned target.
        best_t = None
        best_d = float("inf")
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            if d < best_d:
                best_d = d
                best_t = t
        if best_t is None:
            continue
        ships = max(1, int(src.ships) - 1)
        v = fleet_speed(ships)
        eta = int(math.ceil(best_d / max(v, 1e-6))) if v > 0 else 0
        # Score huge so settle_plan picks this over any snipe candidate
        # at the same source — the endgame override should dominate.
        missions.append(Mission(
            mission_class="endgame_burn",
            src_id=src.id,
            target_id=best_t.id,
            ships=ships,
            score=1e9,
            eta=eta,
        ))
    return missions


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
        + propose_endgame_burn(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
