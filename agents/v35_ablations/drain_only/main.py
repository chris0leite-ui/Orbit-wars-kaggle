"""v3.5 ablation: v3_snipe + drain Mission only."""
from __future__ import annotations
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.drain import propose_drain_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
        + propose_drain_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
