"""v3.5 — top-performer-aligned mission portfolio.

Translates the empirical findings of
`knowledge-base/concepts/top-performer-strategies.md` into Mission
classes that close specific behavioural gaps between v3_snipe and the
top-10 fingerprint. Stack:

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> propose_opening_missions(world, model)       [opening; step ≤ 5]
        -> propose_snipe_missions(world, model)         [snipe; rebalanced denominator]
        -> propose_reinforce_missions(world, model)     [reinforce; defensive]
        -> propose_drain_missions(world, model)         [drain; surplus-safe]
        -> propose_recapture_missions(world, model)     [recapture; 50-turn window]
        -> propose_gang_up_missions(world, model)       [gang_up; paired arrivals]
        -> settle_plan(all_missions, world, model)
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

What's NEW vs v3_snipe / v3.4:
- **Opening** Mission class fires at steps 0-5 to close the opening-tempo
  gap (top-10 first-launch step 4.1 vs midpack 10.5).
- **Drain** Mission class flushes safe high-garrison sources (top-10
  mean garrison-at-launch 11 vs midpack 22).
- **Gang_up** Mission class proposes paired sources at contested targets
  (games-analysis §3 in-flight volume gap).
- **Recapture** Mission class re-takes recently-lost planets within a
  50-turn window (closes the comeback gap from games-analysis §2).
- **Targeted off-by-one** in arrival_size for orbiting/comet only
  (avoids the v3.3 blanket-fix regression).
- **Rebalanced ROI denominator** in snipe (0.5 × ship-cost) — favours
  overwhelming-force commits without flat priority multipliers.

What's UNCHANGED from v3.4:
- LEADER_MULTIPLIER=1.5 in snipe (4P spoiler, in-bundle).
- DEFAULT_MECHANISMS (validate → arrival_size → lead_aim_v2 → sun_avoid
  → path_clears_other_planets → oob_guard).
- Per-source greedy settle_plan with same-turn arrival ledger.
- Trajectory-ray-cast fleet-fate guards.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.drain import propose_drain_missions
from lib.missions.gang_up import propose_gang_up_missions
from lib.missions.opening import propose_opening_missions
from lib.missions.recapture import propose_recapture_missions
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
        + propose_drain_missions(world, model)
        + propose_recapture_missions(world, model)
        + propose_gang_up_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
