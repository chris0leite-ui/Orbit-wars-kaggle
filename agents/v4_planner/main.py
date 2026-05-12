"""v4_planner — receding-horizon mission-portfolio search with adaptive K.

Per-turn pipeline:

1. **2P guard**: count distinct player ids visible on planets/fleets. If >2
   it's a 4P FFA game — the value function and rollout policy are 2P-tuned,
   so fall back to v3.5.1's stateless heuristic.
2. **Incumbent**: build v3.5.1's mission set (snipe aggressive=True +
   reinforce). Realized first so the lookahead never regresses below
   v3.5.1 — if the time budget expires before any portfolio is scored,
   the incumbent action is returned.
3. **Portfolios**: `generate_portfolios` enumerates up to 5 mission-list
   variants (incumbent / conservative / per_source_swap /
   drop_weakest_source / noop). Each is run through `settle_plan` and
   `realize` to produce a candidate action.
4. **Adaptive K**: `adaptive_K(world)` returns 8..30 based on entropy
   (in-flight fleets + neutral planet count). Truncated by
   `truncate_K_to_comet_boundary` to avoid the {50,150,250,350,450}
   spawn boundaries where `env_from_obs` diverges from the live env.
5. **Sim<K> scoring**: each candidate action is scored via
   `score_action(env, action, K, my_id, policy, value_fn=evaluate_value)`
   where the leaf value is the production-share / denial / ships /
   survivor goal-shaped V. Robustness: a `_TIME_LIMIT_MS` watchdog
   cuts the loop early, and per-candidate exceptions are swallowed
   (skip and continue rather than crash the agent).
6. **Argmax**: return the highest-V candidate. If every scored
   candidate raised an exception OR the budget aborted before any
   completed, fall back to the realized incumbent action.

The user's framing was "think N steps ahead → find the ideal state →
recursively compute the path." Symbolic backward induction is
intractable (continuous angle + huge state); the practical reduction
is **receding-horizon control** — at each turn, score candidate
portfolios by simulating to depth N and evaluating a goal-shaped
value function, pick the best, commit only this turn's action,
recompute next turn. The "ideal state" is encoded in V (max own
production share + denial); the "recursive path" is implicit in the
N-step rollout under a fixed-policy opponent model.
"""

from __future__ import annotations

import time

from lib.candidate_portfolios import generate_portfolios
from lib.intent import World, realize
from lib.lookahead import env_from_obs, score_action
from lib.lookahead_planner import adaptive_K, evaluate_value, truncate_K_to_comet_boundary
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# Per-turn hard deadline (ms from agent start). Local profiling on this
# box: env_from_obs ~105 ms (one-time), Sim<K=10> ~124 ms per candidate.
# 750 ms hard deadline + per-step abort inside score_action means even a
# slow rollout cannot exceed the 1 s actTimeout. Incumbent is always
# scored first so a partial-loop abort returns v3.5.1's action without
# regression.
_HARD_DEADLINE_MS = 750.0
# Pre-candidate-start watchdog: skip starting a new candidate sim once
# elapsed time has consumed this much of the deadline. The remaining
# budget (~200 ms) handles a single in-flight candidate that bumps into
# the hard deadline.
_START_WATCHDOG_MS = 550.0


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
            v = score_action(
                sim_env,
                action,
                K=K,
                my_id=my_id,
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
