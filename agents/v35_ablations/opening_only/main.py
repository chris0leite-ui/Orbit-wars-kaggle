"""v3.5 ablation: v3_snipe + opening Mission only.

Used to isolate which v3.5 wave is responsible for the full-stack
regression. If this variant alone beats v3_snipe at ≥55% Wilson lo,
opening is a keeper.
"""
from __future__ import annotations
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.opening import propose_opening_missions
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
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
