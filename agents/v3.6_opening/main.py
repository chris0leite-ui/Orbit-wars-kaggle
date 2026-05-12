"""v3.6_opening — v3.5.1 + bowwowforeach-style opening Mission.

Single surgical change vs v3.5.1: prepend `propose_opening_missions`
to the mission pipeline. Opening fires only for steps 0-5 from home
planets with > 8 ships, draining each source down to OPENING_RESERVE=2
ships per launch (top-10 bowwowforeach #1 archetype, garrison-at-launch
~7.7).

The previously-shipped opener in main (used by v3.5 stack) sent
`max(1, t.ships + 1)` ships — for a 0-ship neutral that's 1 ship,
identical to v3_snipe. The 2026-05-12 sizing fix in
`lib/missions/opening.py` changes that to drain-the-source. With
the firing window unchanged, only the launch SIZE changes.

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> propose_opening_missions(world, model)   [step ≤ 5; drains source]
        -> propose_snipe_missions(world, model, aggressive=True)
        -> propose_reinforce_missions(world, model)
        -> settle_plan(combined, world, model)
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

What's UNCHANGED from v3.5.1:
- snipe aggressive=True (the v3.5.1 midgame win condition).
- reinforce Mission class.
- DEFAULT_MECHANISMS pipeline.
- Per-source greedy settle_plan with same-turn arrival ledger.
- All trajectory-ray-cast guards.
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
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
