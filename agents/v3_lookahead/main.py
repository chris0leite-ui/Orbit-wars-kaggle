"""v3_lookahead — Sim<K> scorer over drop-one-launch candidate set.

Pipeline per turn:

1. Compute the *incumbent action* via the v3_snipe pipeline (Mission
   builder + per-source greedy planner + DEFAULT_MECHANISMS). This is
   the same actions v2 / v3_snipe would emit — our baseline.
2. Build a `drop-one-launch` candidate set:
     [incumbent, incumbent - launch_0, incumbent - launch_1, ...]
   We never propose launches the incumbent didn't consider; v3.1's
   lever is "prune launches the scorer thinks are net-negative."
3. Reconstruct a steppable env from the agent-visible obs + cfg.
4. For each candidate, run a K-step forward sim where the candidate is
   our first move and both players play v3_snipe afterwards. Score by
   (us - opp) ship total at the rollout's final step.
5. Return the candidate with the highest projected score.

Budget envelope (Phase 2 measurements):
- K=20 forward sim ~110 ms median.
- Typical N=3-5 launches → 4-6 candidates → 440-660 ms total.
- 1000 ms actTimeout; safety margin via `_TIME_LIMIT_MS`.

If we run out of wallclock partway through, we return the best
candidate scored so far. We always score the incumbent first so the
fallback never regresses below v3_snipe.
"""

from __future__ import annotations

import time

from lib.intent import World, realize
from lib.lookahead import enumerate_drop_one_candidates, env_from_obs, score_action
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# Per-turn wallclock budget for the lookahead loop. Leaves margin for the
# v3_snipe baseline computation + env reconstruction + return overhead.
_TIME_LIMIT_MS = 400.0
_SIM_K_DEFAULT = 8


def _incumbent_action(obs):
    """The action v3_snipe / v2 would emit. Reused as the rollout policy."""
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)


def agent(obs, configuration=None):
    t_start = time.perf_counter()
    incumbent = _incumbent_action(obs)
    candidates = enumerate_drop_one_candidates(incumbent)
    # With 0 or 1 candidates there's nothing to compare against — just emit.
    if len(candidates) <= 1:
        return incumbent

    sim_env = env_from_obs(obs, configuration)
    my_id = int(obs.get("player", 0))

    best_action = incumbent
    best_score = float("-inf")
    for cand in candidates:
        if (time.perf_counter() - t_start) * 1000.0 > _TIME_LIMIT_MS:
            break
        score = score_action(
            sim_env, cand, K=_SIM_K_DEFAULT, my_id=my_id,
            policy=_incumbent_action,
        )
        if score > best_score:
            best_score = score
            best_action = cand
    return best_action
