"""v3.6_opening_B — v3.5.1 + timing-matched bowwowforeach opener.

Variant B from audit/2026-05-12-v3.6-opening-failed.md hypothesis.
Variant A (default-params) FAILED on canonical panel (41.7% mean WR).

Hypothesis: A regressed because it fired at step 0 with the starting
10-ship garrison, sending 8 and leaving 2 — too-thin defense for
opponents arriving steps 1-20. bowwowforeach's measured 7.7-ship
garrison-at-launch is taken at step 4 with built-up production:
home + 4×prod ≈ 22 ships → send 14, leave 7.7.

This variant waits until the source has built up beyond
`min_garrison=14` (typically step 2-3 depending on home production),
then sends src.ships - 7 ships (leaves bowwow's defensive floor).

Pipeline:
    obs -> World -> WorldModel
        -> propose_opening_missions(window=5, min_garrison=14, reserve=7)
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
            window=5, min_garrison=14, reserve=7, allow_enemies=False,
        )
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
