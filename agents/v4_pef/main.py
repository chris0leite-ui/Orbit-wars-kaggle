"""v4_pef — v4_planner with PEF (Position Evaluation Function) leaf head.

Identical pipeline to v4_planner (receding-horizon mission-portfolio
search with adaptive K), except the leaf value function gets three
additional positional terms:

  V(s) = prod_share + 0.4 * prod_denied + 0.05 * ships_share + 5.0 * lone
       + cluster_weight  * cluster_cohesion(planets, me)
       - frontier_weight * frontier_exposure(planets, me)
       + reach_weight    * reach(planets, me)

These three terms answer the PI's framing of "improve our current
position over the next few steps":

- cluster_cohesion: defensibility. My planets in dense friendly
  clusters score higher — resists counter-attack.
- frontier_exposure: vulnerability penalty. My planets surrounded by
  stronger enemies score lower — discourages indefensible captures.
- reach: offensive option-value. Fraction of non-mine production
  within striking distance — rewards positions with many follow-up
  targets.

Weight defaults are starting points; PEF unit tests in
tests/test_lookahead_planner.py verify each term in isolation.
Backward-compat with v4_planner is bit-exact when all three weights
are 0 (verified by test_evaluate_value_backward_compat_with_default_pef_weights).
"""

from __future__ import annotations

import time
from functools import partial

from lib.candidate_portfolios import generate_portfolios
from lib.intent import World, realize
from lib.lookahead import env_from_obs, score_action
from lib.lookahead_planner import adaptive_K, evaluate_value, truncate_K_to_comet_boundary
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


_HARD_DEADLINE_MS = 750.0
_START_WATCHDOG_MS = 550.0

# PEF weights — start at modest values so the position terms compose
# with the base prod_share (1.0) and denial (0.4) without swamping them.
# These are the parameters under test in the v4_pef vs v4_planner A/B.
_CLUSTER_WEIGHT = 0.3
_FRONTIER_WEIGHT = 0.3
_REACH_WEIGHT = 0.2

_pef_value_fn = partial(
    evaluate_value,
    cluster_weight=_CLUSTER_WEIGHT,
    frontier_weight=_FRONTIER_WEIGHT,
    reach_weight=_REACH_WEIGHT,
)


def _v351_action(obs):
    """v3.5.1 incumbent — used as 4P fallback AND as the rollout policy.

    Inlined to avoid an import cycle (agents/ is not on sys.path for the
    submission bundle; the bundle inlines all dependencies).
    """
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


def _num_distinct_players(obs) -> int:
    """Distinct player ids visible on owned planets + fleets."""
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    owners: set[int] = set()
    for p in planets:
        if int(p[1]) >= 0:
            owners.add(int(p[1]))
    for f in fleets:
        if int(f[1]) >= 0:
            owners.add(int(f[1]))
    return len(owners)


def agent(obs, configuration=None):
    t_start = time.perf_counter()

    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    if not raw_planets:
        return []

    # 4P fallback: env_from_obs reconstructs with num_agents=2 and the
    # value function is 2P-tuned. In 4P FFA, fall back to v3.5.1 unchanged.
    if _num_distinct_players(obs) > 2:
        return _v351_action(obs)

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)

    incumbent_missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    incumbent_intents = settle_plan(incumbent_missions, world, model)
    incumbent_action = realize(
        incumbent_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
    )

    portfolios = generate_portfolios(world, model, incumbent_missions)

    if len(portfolios) <= 2 and all(
        p.label in ("incumbent", "noop") for p in portfolios
    ):
        return incumbent_action

    K = adaptive_K(world)
    K = truncate_K_to_comet_boundary(K, world.step)

    try:
        sim_env = env_from_obs(obs, configuration)
    except Exception:
        return incumbent_action

    my_id = world.my_id
    best_action = incumbent_action
    best_v = float("-inf")

    hard_deadline = t_start + _HARD_DEADLINE_MS / 1000.0
    for portfolio in portfolios:
        elapsed = (time.perf_counter() - t_start) * 1000.0
        if elapsed > _START_WATCHDOG_MS:
            break
        try:
            if portfolio.label == "incumbent":
                action = incumbent_action
            else:
                intents = settle_plan(portfolio.missions, world, model)
                action = realize(
                    intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
                )
            v = score_action(
                sim_env,
                action,
                K=K,
                my_id=my_id,
                policy=_v351_action,
                value_fn=_pef_value_fn,
                deadline=hard_deadline,
            )
        except Exception:
            continue
        if v > best_v:
            best_v = v
            best_action = action

    return best_action
