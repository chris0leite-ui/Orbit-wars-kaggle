"""sa_replay — pre-solve at module load via simulated annealing, then replay.

PI 2026-05-26: in solo this is just a scheduling problem; solve it with
SA starting from ROI's trajectory. This agent does exactly that — at
module load (BEFORE kaggle starts timing turns), it runs SA on a separate
env replica of the same seed and caches the optimised plan. Each agent()
call then returns the cached actions for the current turn.

ONLY makes sense vs a no-op opponent (the SA plan assumes the opponent
does nothing). For Kaggle live play the actTimeout would gate this out —
it's a bench / ceiling-diagnostic agent, not a Kaggle submission.

Env vars (set by scripts/solo_bench.py automatically):
  SA_SEED          — game seed (REQUIRED to trigger pre-solve)
  SA_EPISODE_STEPS — game length (REQUIRED to trigger pre-solve)
  SA_ITERATIONS    — default 500
  SA_T0            — default 500
  SA_COOLING       — default 0.99
  SA_RNG_SEED      — default 42
  SA_INITIAL_AGENT — default "agents/simple/roi.py"

Critical: do NOT use `__file__` at module top-level. kaggle_environments
loads this file via `exec(compile(source, path, "exec"), {})` with an
empty namespace; `__file__` is undefined and `NameError` gets swallowed
into the fallback no-op agent. Use sys.path-relative imports instead.
"""
from __future__ import annotations

import os
import random

# scripts.sa_solo_solver is reachable because the bench harness adds REPO
# to sys.path before exec'ing this file. We pull REPO from there too so
# we don't have to recompute it via __file__ (which kaggle's exec doesn't
# populate — see module docstring).
from scripts.sa_solo_solver import (
    REPO,
    _get_step,
    record_initial_plan,
    simulated_anneal,
)


_PLAN_BY_TURN: dict[int, list[list]] = {}
_PLAN_READY = False


def _solve(seed: int, steps: int) -> dict[int, list[list]]:
    n_iter = int(os.environ.get("SA_ITERATIONS", "500"))
    t0 = float(os.environ.get("SA_T0", "500"))
    cooling = float(os.environ.get("SA_COOLING", "0.99"))
    rng_seed = int(os.environ.get("SA_RNG_SEED", "42"))
    init_agent = os.environ.get("SA_INITIAL_AGENT", "agents/simple/roi.py")
    initial_emissions, _env_score, _n_steps, initial_planets = record_initial_plan(
        seed, steps, REPO / init_agent)
    best_plan, _best_score, _hist = simulated_anneal(
        initial_emissions, seed, steps, n_iter,
        t0, cooling, random.Random(rng_seed),
        initial_planets=initial_planets,
    )
    plan_by_turn: dict[int, list[list]] = {}
    for turn, action in best_plan:
        plan_by_turn.setdefault(int(turn), []).append(list(action))
    return plan_by_turn


def _maybe_solve_at_load():
    global _PLAN_BY_TURN, _PLAN_READY
    seed_env = os.environ.get("SA_SEED")
    steps_env = os.environ.get("SA_EPISODE_STEPS")
    if seed_env is None or steps_env is None:
        return
    try:
        seed = int(seed_env)
        steps = int(steps_env)
    except ValueError:
        return
    _PLAN_BY_TURN = _solve(seed, steps)
    _PLAN_READY = True


def agent(obs, configuration=None):
    t = _get_step(obs)
    if not _PLAN_READY:
        # Fallback path: bench harness didn't set SA_SEED / SA_EPISODE_STEPS.
        # Try configuration; this will exceed actTimeout under env.run but is
        # useful for direct-call testing.
        if configuration is not None:
            seed = int(getattr(configuration, "seed", 0)) if hasattr(configuration, "seed") else int((configuration or {}).get("seed", 0))
            steps = int(getattr(configuration, "episodeSteps", 200)) if hasattr(configuration, "episodeSteps") else int((configuration or {}).get("episodeSteps", 200))
            globals()["_PLAN_BY_TURN"] = _solve(seed, steps)
            globals()["_PLAN_READY"] = True
    return [list(a) for a in _PLAN_BY_TURN.get(int(t), [])]


# Solve at module load if the bench harness supplied SA_SEED / SA_EPISODE_STEPS.
_maybe_solve_at_load()
