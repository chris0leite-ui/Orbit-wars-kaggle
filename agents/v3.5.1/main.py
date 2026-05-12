"""v3.5.1 — v3_snipe + aggressive snipe ship sizing.

Single surgical change vs v3_snipe: snipe now sends a larger fraction
of source garrison per launch (top-10 fingerprint-aligned per
`knowledge-base/concepts/top-performer-strategies.md`).

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> propose_snipe_missions(world, model, aggressive=True)
        -> propose_reinforce_missions(world, model)
        -> settle_plan(combined, world, model)
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

Gate results (audit/2026-05-12-iter2-ablation-results.md):
- 32-seed 2P vs v3_snipe: 44/64 = 68.8%, Wilson lo 56.6%  [PASS]
- 8-seed × 4-seat 4P FFA vs weak background: 31/32 = 96.9%  [PASS]
  (v3_snipe baseline in same panel: 93.8%)
- Parameter sweep (audit/tournaments/sizing-sweep-20260512T044157Z.json):
  fraction=0.7 dominates 0.6, 0.8, 0.9 both vs-baseline and head-to-head.

What's UNCHANGED from v3_snipe:
- DEFAULT_MECHANISMS (validate → arrival_size → lead_aim_v2 →
  sun_avoid → path_clears_other_planets → oob_guard).
- snipe scoring formula, denominator, LEADER_MULTIPLIER 4P spoiler.
- reinforce Mission class.
- Per-source-greedy settle_plan with same-turn arrival ledger.
- All trajectory-ray-cast guards.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
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
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
