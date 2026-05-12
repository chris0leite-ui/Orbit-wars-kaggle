"""v8_fastbrain — v7_0 chassis + v4_planner brain.

The biggest bet from the super-version analysis:
- fast_sim engine from v7_0 (183× faster than v4_planner's env.clone).
- 5 portfolio candidates from v4_planner (incumbent / conservative /
  per_source_swap / drop_weakest_source / noop) — different mission
  compositions per turn vs drop-one's subtractive-only set.
- Adaptive K (8-30) from v4_planner with comet-boundary truncation.
- evaluate_value scoring head (production-share + denial + ships).
- σ-equivariance via library (sym_hypot + planner _tb).
- 4P → v3.5.1 fallback (same as v7_0 / v4_planner).

Per-turn pipeline:
  1. Build incumbent missions (v3.5.1 snipe-aggressive + reinforce).
  2. Compute incumbent action (the parity floor).
  3. Generate 5 portfolios; realize each via settle_plan + realize.
  4. Build Snapshot once; clone per candidate.
  5. Score each via score_candidate(value_fn=evaluate_value, K=adaptive_K).
  6. Watchdog at 700 ms; fall back to incumbent if budget runs out.

Hypothesis: v4_planner's 84.4% vs v7_minimax comes from its richer
candidate set + value head. With our 183× faster sim, we can afford
BIGGER K and/or MORE candidates than v4_planner manages on env.clone.
"""

from __future__ import annotations

import time

from lib.fast_sim import from_obs as fs_from_obs
from lib.candidate_portfolios import generate_portfolios
from lib.intent import World, realize
from lib.lookahead_planner import adaptive_K, evaluate_value, truncate_K_to_comet_boundary
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.v7_search import _infer_num_seats, score_candidate
from lib.world_model import WorldModel


_HARD_DEADLINE_MS = 700.0     # match v7_0
_START_WATCHDOG_MS = 600.0    # start no new candidate past this
_K_MAX = 30                   # cap; with fast_sim ~0.12 ms/step we can go deeper


def _v351_action(obs):
    """4P fallback + 2P incumbent helper."""
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
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []

    # 4P → v3.5.1 fallback (evaluate_value head + lookahead are 2P-tuned).
    if _infer_num_seats(world) != 2:
        return _v351_action(obs)

    model = WorldModel.from_world(world)

    # Build incumbent missions once (used by both portfolio generator
    # and the parity-floor action).
    incumbent_missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    incumbent_intents = settle_plan(incumbent_missions, world, model)
    incumbent_action = realize(
        incumbent_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model,
    )

    # Generate 5 portfolios.
    portfolios = generate_portfolios(world, model, incumbent_missions)

    # Realize each portfolio into a candidate action.
    candidates: list[list] = []
    seen_keys: set[tuple] = set()
    # Incumbent is always first → parity floor.
    inc_key = tuple((int(m[0]), round(float(m[1]), 5), int(m[2])) for m in incumbent_action)
    candidates.append(incumbent_action)
    seen_keys.add(inc_key)
    for portfolio in portfolios:
        if portfolio.label == "incumbent":
            continue  # already added
        try:
            p_intents = settle_plan(portfolio.missions, world, model)
            p_action = realize(
                p_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model,
            )
        except Exception:
            continue
        k = tuple((int(m[0]), round(float(m[1]), 5), int(m[2])) for m in p_action)
        if k not in seen_keys:
            candidates.append(p_action)
            seen_keys.add(k)

    # If we got nothing new beyond the incumbent → return it directly,
    # avoiding the Snapshot setup.
    if len(candidates) <= 1:
        return incumbent_action

    # Adaptive K + comet-boundary truncation.
    K = adaptive_K(world)
    K = truncate_K_to_comet_boundary(K, world.step)
    K = min(K, _K_MAX)

    # Build Snapshot for fast_sim rollouts.
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    my_id = world.my_id

    best_action = incumbent_action
    best_v = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed = (time.perf_counter() - t_start) * 1000.0
        if elapsed > _START_WATCHDOG_MS:
            break
        try:
            v = score_candidate(
                snap, cand,
                my_id=my_id, K=K, opp_tier=1,
                value_fn=evaluate_value,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_v = v
            best_action = cand
            continue
        if v > best_v:
            best_v = v
            best_action = cand

    return best_action
