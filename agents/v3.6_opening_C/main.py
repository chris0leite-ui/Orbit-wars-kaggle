"""v3.6_opening_C — v3.5.1 + drain-source opener with enemy targets.

Variant C from audit/2026-05-12-v3.6-opening-failed.md hypothesis.
Variant A (neutral-only drain-source) FAILED on canonical panel.

Hypothesis: bowwowforeach picks 42% enemy targets vs midpack 14%.
Variant A's neutral-only restriction may have left enemy-home raids
on the table, redirecting full-source launches at low-prod neutrals
instead of denying opponent economy.

This variant uses the same drain-to-2 sizing as A but allows enemy
targets. settle_plan's score function still favours nearby high-prod
targets, so enemy homes get picked only when geometrically favourable.

Pipeline:
    obs -> World -> WorldModel
        -> propose_opening_missions(allow_enemies=True)
        -> propose_snipe_missions(aggressive=True)
        -> propose_reinforce_missions
        -> settle_plan -> realize
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
        propose_opening_missions(
            world, model,
            window=5, min_garrison=8, reserve=2, allow_enemies=True,
        )
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
