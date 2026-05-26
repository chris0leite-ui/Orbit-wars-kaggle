"""sa_online — receding-horizon SA with iterated-best-response co-evolution.

PI 2026-05-26: opp-agnostic search via co-evolved opp plans. Instead of
committing to a fixed opp model (which traps SA in pessimism when the
model is too strong), we evolve our_plan and opp_plan jointly at module
load: SA optimises our_plan against the current opp_plan, then opp_plan
against the new our_plan, alternating. For finite zero-sum games this is
fictitious play and converges asymptotically to Nash equilibrium.

Each per-turn refine uses the CACHED opp_plan as a plan-replay policy —
~14× faster than calling a live ROI agent. With a wallclock deadline
inside the SA loop, every turn fits inside kaggle's actTimeout
regardless of opp complexity.

Three additive fixes vs. the prior sa_online (commit 1d40dc0):
  (A) max_wall_s deadline inside simulated_anneal_online
  (B) differential score mode (ours - max_o theirs) — breaks pessimism
  (C) iterated best response: _co_evolve replaces _initial_solve

Module-load: bench harness sets SA_SEED + SA_EPISODE_STEPS. Co-evolution
runs before kaggle starts timing turns.

Critical: NO `__file__` at module top-level. kaggle_environments loads
agents via `exec(compile(source), {})` with an empty namespace; any
`Path(__file__)` raises NameError that's silently swallowed into a
fallback no-op agent. We pull REPO from scripts.sa_solo_solver.

Env vars (all optional):
  SA_SEED               — required to trigger module-load solve
  SA_EPISODE_STEPS      — required to trigger module-load solve
  SA_COEVOLVE_CYCLES    — fictitious-play alternations (default 3)
  SA_BUDGET_INIT_S      — wallclock per SA solve in co-evolution (default 30)
  SA_BUDGET_STEP_S      — wallclock per per-turn refine (default 0.8)
  SA_ITER_INIT          — iteration cap per co-evolution SA solve (default 300)
  SA_ITER_STEP          — iteration cap per refine (default 100)
  SA_HORIZON            — receding-horizon length in turns (default 30)
  SA_T0                 — SA initial temp for big solve (default 500)
  SA_T0_STEP            — SA initial temp for per-turn refine (default 100)
  SA_COOLING            — geometric cooling for big solve (default 0.99)
  SA_COOLING_STEP       — cooling for refine (default 0.95)
  SA_RNG_SEED           — base RNG seed for co-evolution (default 42)
  SA_BOOTSTRAP_AGENT    — focal for the bootstrap recording (default agents/simple/roi.py)
"""
from __future__ import annotations

import os
import random

# ---------------------------------------------------------------------------
# Kaggle ladder defaults — set BEFORE any other env-var read in this file.
# These are tuned for orbit_wars's actTimeout=1s + remainingOverageTime=60s.
# Each can be overridden by setting the env var BEFORE importing this module.
# PI 2026-05-26 PM observation submit: opp_policy = simple/nearest.
# ---------------------------------------------------------------------------
os.environ.setdefault("SA_REFINE_OPP_POLICY", "agents/simple/nearest.py")
os.environ.setdefault("SA_BUDGET_STEP_S", "0.5")         # leaves 500ms slack vs actTimeout=1s
os.environ.setdefault("SA_ITER_STEP", "30")
os.environ.setdefault("SA_HORIZON", "30")
os.environ.setdefault("SA_COEVOLVE_CYCLES", "1")          # first-turn budget: 2 SAs * 15s = 30s, fits 60s overage
os.environ.setdefault("SA_BUDGET_INIT_S", "15")
os.environ.setdefault("SA_ITER_INIT", "200")
os.environ.setdefault("SA_T0", "500")
os.environ.setdefault("SA_T0_STEP", "100")
os.environ.setdefault("SA_COOLING", "0.99")
os.environ.setdefault("SA_COOLING_STEP", "0.95")
os.environ.setdefault("SA_RNG_SEED", "42")
os.environ.setdefault("SA_BOOTSTRAP_AGENT", "agents/simple/roi.py")

# scripts.sa_solo_solver is reachable because the bench harness adds REPO
# to sys.path before exec'ing this file. We pull REPO from there too so
# we don't have to recompute it via __file__.
from pathlib import Path

# REPO discovery — kaggle's exec() doesn't set __file__, so do NOT use it
# at module top-level. Instead trust sys.path[0] which the bench harness
# (and live Kaggle) populates with the working dir.
import sys as _sys
REPO = Path(_sys.path[0]) if _sys.path[0] else Path(".")

from lib.sa_core import _get_step
from lib.sa_core import build_solo_snap0 as _build_solo_snap0
from lib.sa_core import load_agent as _load_agent
from lib.sa_core import record_initial_plan
from lib.sa_core import simulated_anneal_online
from lib.fast_sim import from_obs as fs_from_obs


_PLAN_BY_TURN: dict[int, list[list]] = {}
_OPP_PLAN_BY_TURN: dict[int, list[list]] = {}
_INITIAL_PLANETS: list = []
_SETTINGS: dict = {}
_INITIALIZED: bool = False  # set after first-call init regardless of co_evolve success


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


def _plan_dict_to_list(plan_dict):
    return [(int(tau), list(a))
            for tau, acts in plan_dict.items()
            for a in acts]


def _plan_replay_policy(plan_emissions):
    """Wrap a list[(turn, action)] as Callable(obs) -> list[actions]."""
    plan_dict = _plan_list_to_dict(plan_emissions)
    def policy(obs):
        t = _get_step(obs)
        return [list(a) for a in plan_dict.get(t, [])]
    return policy


def _co_evolve(seed: int, steps: int):
    """Iterated best response between our_plan and opp_plan.

    Returns (our_plan_dict, opp_plan_dict). For finite zero-sum games
    this is fictitious play and converges asymptotically to a Nash
    equilibrium (Robinson 1951). Even 2-3 cycles are enough to escape
    the MPC pessimism trap because opp is no longer a fixed counter —
    it's an evolving plan SA can find new attacks against.
    """
    global _INITIAL_PLANETS
    n_cycles = int(os.environ.get("SA_COEVOLVE_CYCLES", "3"))
    budget_s = float(os.environ.get("SA_BUDGET_INIT_S", "30"))
    iter_cap = int(os.environ.get("SA_ITER_INIT", "300"))
    t0 = float(os.environ.get("SA_T0", "500"))
    cooling = float(os.environ.get("SA_COOLING", "0.99"))
    base_rng = int(os.environ.get("SA_RNG_SEED", "42"))
    bootstrap_agent = REPO / os.environ.get(
        "SA_BOOTSTRAP_AGENT", "agents/simple/roi.py")

    # Bootstrap: record ROI vs ROI to get a starting plan for both sides.
    initial_emissions, _env_score, _n_steps, initial_planets = record_initial_plan(
        seed, steps, bootstrap_agent, opp_path=bootstrap_agent)
    _INITIAL_PLANETS = initial_planets

    our_plan = list(initial_emissions)
    opp_plan = list(initial_emissions)

    snap0 = _build_solo_snap0(seed, steps)

    for cycle in range(n_cycles):
        # Our SA: maximise (our_ships - opp_ships) holding opp_plan fixed.
        opp_policy = _plan_replay_policy(opp_plan)
        our_plan, _best, _hist = simulated_anneal_online(
            our_plan, snap0, max_steps=steps,
            opp_policy=opp_policy,
            n_iter=iter_cap, t0=t0, cooling=cooling,
            rng=random.Random(base_rng + cycle),
            start_step=0,
            initial_planets=initial_planets,
            max_wall_s=budget_s,
            me=0,
            score_mode="diff",
        )

        # Opp SA: from opp's POV (me=1), maximise (their_ships - our_ships)
        # holding our new our_plan fixed.
        our_policy = _plan_replay_policy(our_plan)
        opp_plan, _best, _hist = simulated_anneal_online(
            opp_plan, snap0, max_steps=steps,
            opp_policy=our_policy,
            n_iter=iter_cap, t0=t0, cooling=cooling,
            rng=random.Random(base_rng + 1000 + cycle),
            start_step=0,
            initial_planets=initial_planets,
            max_wall_s=budget_s,
            me=1,
            score_mode="diff",
        )

    return _plan_list_to_dict(our_plan), _plan_list_to_dict(opp_plan)


_RUNTIME_OPP_AGENT_CACHE: dict[str, object] = {}


def _resolve_runtime_opp_policy(path_spec: str):
    """Load an agent function from `path_spec` and wrap it as a noop-safe
    policy. Used when SA_REFINE_OPP_POLICY names an agent (e.g.
    'agents/simple/nearest.py') instead of 'noop' or 'coevolve'.

    Cached on first load so we don't re-import per turn.
    """
    cached = _RUNTIME_OPP_AGENT_CACHE.get(path_spec)
    if cached is not None:
        return cached
    try:
        agent_fn = _load_agent(REPO / path_spec)
    except Exception:
        agent_fn = lambda _obs: []  # fail-safe to noop
    _RUNTIME_OPP_AGENT_CACHE[path_spec] = agent_fn
    return agent_fn


def _refine_step(obs, configuration, t: int):
    """Per-turn refine; two modes selected by SA_REFINE_OPP_POLICY env var.

    Modes (default "noop"):

      "noop"     — opp_policy = noop in OUR rollout. PI 2026-05-26 PM:
                   observability replaces opp prediction. Snap encodes
                   opp's past + currently-in-flight actions (substrate
                   gate: test_inflight_opp_fleets_advance_under_noop).
                   We don't *predict* opp's future emissions — we'll see
                   them in next turn's obs and re-plan. Closed-loop MPC.
                   Full per-turn budget for OUR SA. opp_plan unchanged.

      "coevolve" — Conditional iterated best response: snap from current
                   obs, N_REFINE_CYCLES of alternating SA (our vs opp),
                   update BOTH cached plans. Heavier, more sophisticated;
                   produces Nash-against-itself plans that may diverge
                   from a heuristic opponent like simple/roi.

    Both modes share: snap-from-current-obs, deadline, diff score,
    hot-start from cached plans. Updates _OPP_PLAN_BY_TURN as a side
    effect (only in coevolve mode); always returns the new
    _PLAN_BY_TURN dict.
    """
    global _OPP_PLAN_BY_TURN

    seed = _SETTINGS["seed"]
    steps = _SETTINGS["steps"]
    snap_t = fs_from_obs(obs, configuration,
                          episode_seed=seed, num_seats=2)
    remaining_our = [
        (tau, list(a))
        for tau, acts in _PLAN_BY_TURN.items()
        for a in acts if tau >= t
    ]
    remaining_opp = [
        (tau, list(a))
        for tau, acts in _OPP_PLAN_BY_TURN.items()
        for a in acts if tau >= t
    ]
    horizon = min(steps - t, int(os.environ.get("SA_HORIZON", "30")))
    if horizon <= 0:
        return _PLAN_BY_TURN

    total_budget = float(os.environ.get("SA_BUDGET_STEP_S", "0.8"))
    iter_cap = int(os.environ.get("SA_ITER_STEP", "100"))
    t0_step = float(os.environ.get("SA_T0_STEP", "100"))
    cool = float(os.environ.get("SA_COOLING_STEP", "0.95"))
    mode_raw = os.environ.get("SA_REFINE_OPP_POLICY", "noop").strip()
    mode = mode_raw.lower()

    if mode == "coevolve":
        n_cycles = max(1, int(os.environ.get("SA_REFINE_CYCLES", "1")))
        budget_per_side = total_budget / (2 * n_cycles)
        for cycle in range(n_cycles):
            # OUR best response to current opp_plan.
            opp_policy = _plan_replay_policy(remaining_opp)
            remaining_our, _b, _h = simulated_anneal_online(
                remaining_our, snap_t, max_steps=horizon,
                opp_policy=opp_policy,
                n_iter=iter_cap, t0=t0_step, cooling=cool,
                rng=random.Random(t * 2 + cycle),
                start_step=t,
                initial_planets=_INITIAL_PLANETS,
                max_wall_s=budget_per_side,
                me=0,
                score_mode="diff",
            )
            # OPP best response to our updated plan (search from opp's POV).
            our_policy = _plan_replay_policy(remaining_our)
            remaining_opp, _b, _h = simulated_anneal_online(
                remaining_opp, snap_t, max_steps=horizon,
                opp_policy=our_policy,
                n_iter=iter_cap, t0=t0_step, cooling=cool,
                rng=random.Random(t * 2 + cycle + 10_000),
                start_step=t,
                initial_planets=_INITIAL_PLANETS,
                max_wall_s=budget_per_side,
                me=1,
                score_mode="diff",
            )
    else:
        # mode is either "noop" (default) or a live-agent path
        # (e.g. "agents/simple/nearest.py"). Live-agent paths use the
        # original (non-lowercased) string so case-sensitive paths work.
        # Skip the opp-side SA in both cases — full per-turn budget for
        # OUR SA. opp_plan stays unchanged.
        if mode == "noop":
            opp_policy = lambda _obs: []
        else:
            opp_policy = _resolve_runtime_opp_policy(mode_raw)
        remaining_our, _b, _h = simulated_anneal_online(
            remaining_our, snap_t, max_steps=horizon,
            opp_policy=opp_policy,
            n_iter=iter_cap, t0=t0_step, cooling=cool,
            rng=random.Random(t),
            start_step=t,
            initial_planets=_INITIAL_PLANETS,
            max_wall_s=total_budget,
            me=0,
            score_mode="diff",
        )

    # Merge: past tau < t kept as-is; future replaced with refined.
    new_our: dict[int, list[list]] = {
        int(tau): acts
        for tau, acts in _PLAN_BY_TURN.items() if tau < t
    }
    for tau, action in remaining_our:
        new_our.setdefault(int(tau), []).append(list(action))

    if mode == "coevolve":
        new_opp: dict[int, list[list]] = {
            int(tau): acts
            for tau, acts in _OPP_PLAN_BY_TURN.items() if tau < t
        }
        for tau, action in remaining_opp:
            new_opp.setdefault(int(tau), []).append(list(action))
        _OPP_PLAN_BY_TURN = new_opp

    return new_our


def _maybe_solve_at_load():
    global _PLAN_BY_TURN, _OPP_PLAN_BY_TURN, _SETTINGS
    sc = _resolve_seed_and_steps_from_env()
    if sc is None:
        return
    seed, steps = sc
    _SETTINGS["seed"] = seed
    _SETTINGS["steps"] = steps
    _PLAN_BY_TURN, _OPP_PLAN_BY_TURN = _co_evolve(seed, steps)


def _safe_co_evolve(seed: int, steps: int):
    """Try _co_evolve; on failure (e.g. Kaggle sandbox blocks recursive
    make() or external agent files missing in the bundle), fall back to
    empty plans so per-turn refine can build from scratch."""
    try:
        return _co_evolve(seed, steps)
    except Exception:
        return {}, {}


def _capture_initial_planets_from_obs(obs):
    global _INITIAL_PLANETS
    if _INITIAL_PLANETS:
        return
    try:
        obs_d = obs if isinstance(obs, dict) else dict(obs)
        _INITIAL_PLANETS = [list(p) for p in (obs_d.get("planets") or [])]
    except Exception:
        _INITIAL_PLANETS = []


def agent(obs, configuration=None):
    global _PLAN_BY_TURN, _OPP_PLAN_BY_TURN, _SETTINGS, _INITIALIZED
    t = _get_step(obs)

    if not _INITIALIZED:
        seed, steps = _resolve_seed_and_steps_from_config(obs, configuration)
        _SETTINGS["seed"] = seed
        _SETTINGS["steps"] = steps
        _capture_initial_planets_from_obs(obs)
        _PLAN_BY_TURN, _OPP_PLAN_BY_TURN = _safe_co_evolve(seed, steps)
        _INITIALIZED = True
    elif t > 0:
        try:
            _PLAN_BY_TURN = _refine_step(obs, configuration, t)
        except Exception:
            # Per-turn refine failed (unexpected); keep previous plan
            # so the agent at least returns its last cached action.
            pass

    return [list(a) for a in _PLAN_BY_TURN.get(int(t), [])]


# Solve at module load if the bench harness supplied SA_SEED / SA_EPISODE_STEPS.
# Wrap in try/except so the agent module always imports cleanly even when the
# load-time co-evolve fails (Kaggle sandbox, missing external files, etc.).
try:
    _maybe_solve_at_load()
    if _PLAN_BY_TURN:
        _INITIALIZED = True
except Exception:
    pass
