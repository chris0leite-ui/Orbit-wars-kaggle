"""sa_online — receding-horizon SA vs a simple opponent model.

PI 2026-05-26: extend the solo SA solver to vs-opponent play via Model
Predictive Control. At every turn:
  1. Predict opponent moves with `simple/roi` as the model.
  2. Hot-start from the previous turn's cached plan (shifted forward).
  3. Run a small SA refinement.
  4. Emit the cached action for the current turn.

Module-load: when SA_SEED + SA_EPISODE_STEPS are set in env (bench
harness does this — see scripts/solo_bench.py), run a larger initial
SA solve at import time. Kaggle's actTimeout doesn't apply at module
load, so this is where the heavy lifting goes. See sa_replay/main.py
for the same trick.

Critical: NO `__file__` at module top-level. kaggle_environments loads
agents via `exec(compile(source), {})` with an empty namespace; any
`Path(__file__)` raises NameError that's silently swallowed into a
fallback no-op agent.

Env vars (all optional):
  SA_SEED            — required to trigger module-load solve
  SA_EPISODE_STEPS   — required to trigger module-load solve
  SA_ITER_INIT       — iterations for the turn-0 solve (default 200)
  SA_ITER_STEP       — iterations per turn after t=0 (default 20)
  SA_HORIZON         — receding-horizon length in turns (default 50)
  SA_T0              — initial annealing temp for the big solve (default 500)
  SA_T0_STEP         — per-turn refine temp (default 100)
  SA_COOLING         — cooling for the big solve (default 0.99)
  SA_COOLING_STEP    — cooling for refines (default 0.95)
  SA_RNG_SEED        — RNG seed for the big solve (default 42)
  SA_OPP_AGENT       — opponent model path, default agents/simple/roi.py
  SA_INITIAL_AGENT   — bootstrap-plan agent path, default agents/simple/roi.py
"""
from __future__ import annotations

import os
import random

# scripts.sa_solo_solver is reachable because the bench harness adds REPO
# to sys.path before exec'ing this file. We pull REPO from there too so
# we don't have to recompute it via __file__.
from scripts.sa_solo_solver import (
    REPO,
    _build_solo_snap0,
    _load_agent,
    record_initial_plan,
)
from lib.sa_core import (
    _get_step,
    simulated_anneal_online,
)
from lib.fast_sim import from_obs as fs_from_obs


_PLAN_BY_TURN: dict[int, list[list]] = {}
_OPP_POLICY = None
_INITIAL_PLANETS: list = []
_SETTINGS: dict = {}


def _load_opp_policy():
    """Load the opponent-model agent function; fall back to noop on error."""
    try:
        path = REPO / os.environ.get("SA_OPP_AGENT", "agents/simple/roi.py")
        return _load_agent(path)
    except Exception:
        return lambda obs: []


def _resolve_seed_and_steps_from_env() -> tuple[int, int] | None:
    seed_env = os.environ.get("SA_SEED")
    steps_env = os.environ.get("SA_EPISODE_STEPS")
    if seed_env is None or steps_env is None:
        return None
    try:
        return int(seed_env), int(steps_env)
    except ValueError:
        return None


def _resolve_seed_and_steps_from_config(obs, configuration) -> tuple[int, int]:
    """Fallback when env vars aren't set (e.g. direct env.run with config)."""
    if configuration is None:
        return 0, 200
    if hasattr(configuration, "seed"):
        seed = int(configuration.seed)
    elif isinstance(configuration, dict):
        seed = int(configuration.get("seed", 0))
    else:
        seed = 0
    if hasattr(configuration, "episodeSteps"):
        steps = int(configuration.episodeSteps)
    elif isinstance(configuration, dict):
        steps = int(configuration.get("episodeSteps", 200))
    else:
        steps = 200
    return seed, steps


def _plan_list_to_dict(plan_list):
    out: dict[int, list[list]] = {}
    for t, action in plan_list:
        out.setdefault(int(t), []).append(list(action))
    return out


def _initial_solve(seed: int, steps: int):
    """Big SA at turn 0: record ROI vs opp, then SA-refine."""
    global _INITIAL_PLANETS
    opp_path = REPO / os.environ.get("SA_OPP_AGENT", "agents/simple/roi.py")
    init_agent_path = REPO / os.environ.get(
        "SA_INITIAL_AGENT", "agents/simple/roi.py")
    initial_emissions, _env_score, _n_steps, initial_planets = record_initial_plan(
        seed, steps, init_agent_path, opp_path=opp_path)
    _INITIAL_PLANETS = initial_planets

    snap0 = _build_solo_snap0(seed, steps)
    best_plan, _best_score, _hist = simulated_anneal_online(
        initial_emissions, snap0, max_steps=steps,
        opp_policy=_OPP_POLICY,
        n_iter=int(os.environ.get("SA_ITER_INIT", "200")),
        t0=float(os.environ.get("SA_T0", "500")),
        cooling=float(os.environ.get("SA_COOLING", "0.99")),
        rng=random.Random(int(os.environ.get("SA_RNG_SEED", "42"))),
        start_step=0,
        initial_planets=initial_planets,
    )
    return _plan_list_to_dict(best_plan)


def _refine_step(obs, configuration, t: int):
    """Per-turn small SA: hot-start from cached plan, re-snap, refine."""
    seed = _SETTINGS["seed"]
    steps = _SETTINGS["steps"]
    snap_t = fs_from_obs(obs, configuration,
                          episode_seed=seed, num_seats=2)
    remaining_plan = [
        (tau, list(a))
        for tau, acts in _PLAN_BY_TURN.items()
        for a in acts if tau >= t
    ]
    horizon = min(steps - t, int(os.environ.get("SA_HORIZON", "50")))
    if horizon <= 0:
        return _PLAN_BY_TURN
    best_plan, _best_score, _hist = simulated_anneal_online(
        remaining_plan, snap_t, max_steps=horizon,
        opp_policy=_OPP_POLICY,
        n_iter=int(os.environ.get("SA_ITER_STEP", "20")),
        t0=float(os.environ.get("SA_T0_STEP", "100")),
        cooling=float(os.environ.get("SA_COOLING_STEP", "0.95")),
        rng=random.Random(t),  # per-turn seed for reproducibility
        start_step=t,
        initial_planets=_INITIAL_PLANETS,
    )
    # Merge refined plan back: replace turns >= t with the new plan's actions,
    # keep turns < t as-is (those are already executed).
    new_plan_dict: dict[int, list[list]] = {
        int(tau): acts for tau, acts in _PLAN_BY_TURN.items() if tau < t
    }
    for tau, action in best_plan:
        new_plan_dict.setdefault(int(tau), []).append(list(action))
    return new_plan_dict


def _maybe_solve_at_load():
    global _PLAN_BY_TURN, _OPP_POLICY, _SETTINGS
    sc = _resolve_seed_and_steps_from_env()
    if sc is None:
        return
    seed, steps = sc
    _OPP_POLICY = _load_opp_policy()
    _SETTINGS["seed"] = seed
    _SETTINGS["steps"] = steps
    _PLAN_BY_TURN = _initial_solve(seed, steps)


def agent(obs, configuration=None):
    global _PLAN_BY_TURN, _OPP_POLICY, _SETTINGS
    t = _get_step(obs)

    if not _PLAN_BY_TURN:
        # Module-load path didn't fire (env vars unset). Fall back to
        # solving on first call — will exceed actTimeout under env.run
        # but works for direct-call testing.
        if _OPP_POLICY is None:
            _OPP_POLICY = _load_opp_policy()
        seed, steps = _resolve_seed_and_steps_from_config(obs, configuration)
        _SETTINGS["seed"] = seed
        _SETTINGS["steps"] = steps
        _PLAN_BY_TURN = _initial_solve(seed, steps)
    elif t > 0:
        _PLAN_BY_TURN = _refine_step(obs, configuration, t)

    return [list(a) for a in _PLAN_BY_TURN.get(int(t), [])]


# Solve at module load if the bench harness supplied SA_SEED / SA_EPISODE_STEPS.
_maybe_solve_at_load()
