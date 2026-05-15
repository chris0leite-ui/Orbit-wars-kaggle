"""v12 — v4_planner architecture parity-match (Commit 1 of the v12 PR).

This commit ports `submissions/v4_planner.py::agent` (the frozen
μ=1056 bundle) onto the in-tree libs. No modeling changes yet —
verifies that `agents/v12` produces bit-identical actions to the
bundle. Subsequent commits layer:

  C2 — CRN via pre-recorded opp_traj (variance reduction)
  C3 — STRATEGIC_HORIZON pv_horizon term in value head
  C4 — opp policy upgrade to top_tier_mirror_policy
  C5 — adaptive K bump if wallclock allows

The agent pipeline (each turn):

  1. Build incumbent missions (snipe aggressive + reinforce)
  2. settle_plan -> realize -> incumbent_action  (always emit if score loses)
  3. Generate ≤ 5 mission portfolios via lib.candidate_portfolios
  4. Adaptive K, comet-boundary-safe truncation
  5. Reconstruct env_from_obs (~105ms one-shot)
  6. For each portfolio: settle_plan + realize -> action; score_action via
     score_action(K=K, value_fn=evaluate_value) under _v351_action policy
     for both players in the rollout
  7. Return best-scoring portfolio's action; fallback to incumbent on
     exception or deadline

4P games fall back to v3.5.1 (env_from_obs is num_agents=2 and the value
head is 2P-tuned).
"""

from __future__ import annotations

import time
from typing import Any

from lib.candidate_portfolios import generate_portfolios
from lib.intent import World, realize
from lib.lookahead import env_from_obs, record_opp_traj, score_action_crn
from lib.lookahead_planner import (
    adaptive_K,
    evaluate_value,
    truncate_K_to_comet_boundary,
)
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# Per-turn hard deadline (ms from agent start). Local profiling on this
# box shows score_action ~24 ms per clone + ~10 ms per step (K=10 ~120 ms).
# 750 ms hard deadline + per-step abort inside score_action means even a
# slow rollout terminates inside the 1000 ms actTimeout.
_HARD_DEADLINE_MS = 750.0

# Start-watchdog: don't begin a new portfolio's rollout once the agent's
# elapsed time has consumed this much of the deadline. The remaining
# budget is reserved for the in-progress rollout to either finish or hit
# the hard deadline.
_START_WATCHDOG_MS = 550.0


def _v351_action(obs: Any) -> list:
    """v3.5.1 incumbent — used as 4P fallback AND as the rollout policy.

    Same structure as `submissions/v4_planner.py::_v351_action`. Kept
    inline (not a lib export) because the agent's `policy` parameter is
    called many times per rollout step and the indirection cost matters.
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


def _num_distinct_players(obs: Any) -> int:
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

    # Empty world (no planets visible at all) — no work to do.
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

    # Build incumbent missions once and reuse for the portfolio generator.
    incumbent_missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    incumbent_intents = settle_plan(incumbent_missions, world, model)
    incumbent_action = realize(
        incumbent_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
    )

    portfolios = generate_portfolios(world, model, incumbent_missions)

    # Short-circuit: if only the incumbent portfolio is non-trivial (e.g.
    # opening turn with one source, no alternatives generated), skip the
    # ~105 ms env_from_obs setup and just return the incumbent action.
    if len(portfolios) <= 2 and all(
        p.label in ("incumbent", "noop") for p in portfolios
    ):
        return incumbent_action

    # Adaptive K + comet-boundary truncation.
    K = adaptive_K(world)
    K = truncate_K_to_comet_boundary(K, world.step)

    # Reconstruct env once for cloning across candidates.
    try:
        sim_env = env_from_obs(obs, configuration)
    except Exception:
        # If reconstruction fails for any reason, fall back to incumbent.
        return incumbent_action

    my_id = world.my_id
    best_action = incumbent_action
    best_v = float("-inf")

    hard_deadline = t_start + _HARD_DEADLINE_MS / 1000.0

    # CRN: record opp's K-step trajectory once (under "me idle" baseline),
    # then replay it for every portfolio. Removes opp-policy variance
    # from the cross-portfolio argmax. ~60-100 ms one-shot cost.
    try:
        opp_traj = record_opp_traj(
            sim_env, K=K, my_id=my_id,
            policy=_v351_action, deadline=hard_deadline,
        )
    except Exception:
        # If trajectory recording fails, fall back to incumbent.
        return incumbent_action

    for portfolio in portfolios:
        elapsed = (time.perf_counter() - t_start) * 1000.0
        # Skip starting a new candidate once we're past the start watchdog.
        # The per-step deadline inside score_action handles a candidate
        # that bumps into the hard deadline mid-rollout.
        if elapsed > _START_WATCHDOG_MS:
            break
        # Build the action for this portfolio. Incumbent is precomputed;
        # other portfolios re-run settle_plan + realize.
        try:
            if portfolio.label == "incumbent":
                action = incumbent_action
            else:
                intents = settle_plan(portfolio.missions, world, model)
                action = realize(
                    intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
                )
            v = score_action_crn(
                sim_env,
                action,
                K=K,
                my_id=my_id,
                opp_traj=opp_traj,
                policy=_v351_action,
                value_fn=evaluate_value,
                deadline=hard_deadline,
            )
        except Exception:
            # Never crash the agent — skip this candidate.
            continue
        if v > best_v:
            best_v = v
            best_action = action

    return best_action
