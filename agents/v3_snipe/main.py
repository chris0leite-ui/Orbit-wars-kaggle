"""v3_snipe — mission-framework agent (snipe + reinforce as of 2026-05-11).

Despite the directory name "v3_snipe" (kept for bundle continuity), this
agent now exercises the full Block E mission-class portfolio:

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> propose_snipe_missions(world, model)       [snipe class]
        -> propose_reinforce_missions(world, model)   [reinforce class]
        -> settle_plan(combined, world, model)        [same-turn ledger]
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

Mission classes (2026-05-12):
- **snipe**: capture non-our planets. Cost-aware ROI + comet-lifetime
  correction (don't target departing comets).
- **reinforce**: defend our planets predicted to flip to an enemy this
  horizon. Detected via `WorldModel.owner_at` timeline scan; addresses
  the "no defence" gap in `docs/strategies/simple-roi.md` line 130.
- **recapture**: retake planets we lost in the last 50 turns. Carries
  a time-decaying 1.5×-peak bonus on top of standard snipe scoring,
  capped on garrisons > 50. Closes the comeback gap documented in
  `audit/2026-05-11-v3-snipe-games-analysis.md` §2 (wins recover to
  median 28 planets after home loss; losses peak at 6).

Solver (`lib/planner.settle_plan`): per-source greedy with a same-turn
arrival ledger. Each source picks its highest-score mission whose target
isn't already over-committed by earlier-this-turn picks. Addresses the
"no same-turn ledger" gap (`simple-roi.md` line 127).

Mechanism stack: unchanged Block A physics (validate -> arrival_size ->
lead_aim_v2 -> sun_avoid -> path_clears_other_planets -> oob_guard).
The 2026-05-11 trajectory-ray-cast fixes apply automatically.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
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
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
        + propose_recapture_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
