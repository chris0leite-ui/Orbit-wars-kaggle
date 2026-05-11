"""v3_snipe — Block E mission-framework MVP, snipe-only.

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> propose_snipe_missions(world, model)  [lib/missions/snipe.py]
        -> settle_plan(missions, world, model)    [lib/planner.py]
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

What this changes vs v2:
- v2's per-source greedy is replaced by a planner that picks at most one
  mission per source AND never sends two of our this-turn fleets at the
  same target. v2 over-commits when two sources independently rank the
  same target highest; v3 falls back to the source's second-best target.
- The Mission abstraction opens the door to v3.1+ classes (reinforce /
  recapture / gang_up), all running through the same settle_plan solver.

Mechanism stack: unchanged from v2 (Block A physics: validate ->
arrival_size -> lead_aim_v2 -> sun_avoid -> path_clears_other_planets ->
oob_guard). v3's lever is the planner, not the mechanism layer.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)
