"""v8_minimal — v7_0's drop-one chassis + σ-equiv + evaluate_value head.

The cheapest bet from the super-version analysis:
- Same drop-one candidate enumeration as v7_0 (the proven 79.2% winner).
- Same fast_sim engine (183× faster than env.clone).
- ADD σ-equivariance via the library (sym_hypot + planner _tb +
  SCORE_ROUND=6 — already in lib/ after the v7 stack work).
- SWAP scoring head from `delta_us_minus_them` to `evaluate_value`
  (production-share + denial + ships + survivor — ported from v4_planner).

Isolates: does v4_planner's edge over v7_0 come from σ-equiv + value-
head ALONE (no portfolios, no adaptive K)? If yes, this is the
simplest "super" agent — minimal new code, maximum leverage.
"""

from __future__ import annotations

from lib.lookahead_planner import evaluate_value
from lib.v7_search import choose_simple_2p, _infer_num_seats
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


def _v351_fallback(obs):
    """4P fallback — same as v7_0."""
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


def agent(obs, configuration=None):
    # 4P → v3.5.1 fallback (same as v7_0).
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if _infer_num_seats(world) != 2:
        return _v351_fallback(obs)
    # 2P: drop-one + fast_sim + evaluate_value + σ-equiv (lib-level).
    return choose_simple_2p(
        obs, configuration,
        K=10,
        wallclock_ms=700.0,
        include_recapture=False,
        value_fn=evaluate_value,
    )
