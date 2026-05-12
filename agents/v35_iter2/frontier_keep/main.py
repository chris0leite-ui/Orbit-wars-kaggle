"""v3.5.1 ablation: frontier_keep defensive reserve (per @ykhnkf).

A snipe Mission from a frontier source is allowed only if AFTER the
launch, the source retains a minimum garrison sufficient to deter the
nearest enemy fleet.

Frontier source = an owned planet within FRONTIER_RADIUS of any
enemy-owned planet. Required keep =
    max(KEEP_FLOOR + production * KEEP_PROD_MULT,
        nearest_enemy_ships * KEEP_THREAT_FRAC)

If the proposed launch would drop src.ships below this keep value, the
intent is dropped (the source defers to defense). Non-frontier sources
are unaffected.

This is the defensive twin of v3.5's drain Mission — instead of forcing
a launch when a source is "safe surplus," this BLOCKS launches when a
frontier source can't afford to lose ships.
"""

from __future__ import annotations

import math

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel

FRONTIER_RADIUS = 30.0
KEEP_FLOOR = 6
KEEP_PROD_MULT = 4
KEEP_THREAT_FRAC = 0.30


def _frontier_filter(
    missions: list[Mission], world: World
) -> list[Mission]:
    """Drop snipe Missions that would leave a frontier source below
    its computed keep value."""
    if not missions:
        return missions
    # Compute the nearest enemy distance + ships for each owned planet.
    enemies = [p for p in world.planets_by_id.values()
                if p.owner != world.my_id and p.owner != -1]
    keep_by_src: dict[int, int] = {}
    for src in world.planets_by_id.values():
        if src.owner != world.my_id:
            continue
        nearest_enemy_d = float("inf")
        nearest_enemy_ships = 0
        for e in enemies:
            d = math.hypot(e.x - src.x, e.y - src.y)
            if d < nearest_enemy_d:
                nearest_enemy_d = d
                nearest_enemy_ships = int(e.ships)
        if nearest_enemy_d <= FRONTIER_RADIUS:
            keep = max(
                KEEP_FLOOR + int(src.production) * KEEP_PROD_MULT,
                int(nearest_enemy_ships * KEEP_THREAT_FRAC),
            )
            keep_by_src[src.id] = keep

    if not keep_by_src:
        return missions

    out = []
    for m in missions:
        keep = keep_by_src.get(m.src_id)
        if keep is None:
            out.append(m)
            continue
        src = world.planets_by_id.get(m.src_id)
        if src is None:
            out.append(m)
            continue
        # Drop only snipe-class missions; reinforce should never be
        # filtered out (reinforce IS the defensive response).
        if m.mission_class != "snipe":
            out.append(m)
            continue
        if int(src.ships) - int(m.ships) >= keep:
            out.append(m)
        # else: drop — source can't afford this launch + keep frontier
    return out


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
    )
    missions = _frontier_filter(missions, world)
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
