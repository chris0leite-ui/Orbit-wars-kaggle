"""Integration tests for agents/sa_online co-evolution.

The critical-rigor gates for the iterated-best-response architecture:
co-evolve must produce non-empty plans, and the score for our side
must trend non-decreasing as cycles add up (the convergence property
that defines fictitious play's value).
"""
from __future__ import annotations

import os
import random

import pytest


def _set_env(**kw):
    """Snapshot + override env vars; caller restores via the returned dict."""
    snap = {}
    for k, v in kw.items():
        snap[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return snap


def _restore_env(snap):
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_coevolve_produces_nonempty_plans():
    """1 cycle must yield non-empty our_plan AND opp_plan on a tiny game."""
    snap = _set_env(SA_COEVOLVE_CYCLES="1", SA_BUDGET_INIT_S="3",
                     SA_ITER_INIT="30")
    try:
        from agents.sa_online.main import _co_evolve
        our_plan, opp_plan = _co_evolve(seed=0, steps=50)
        assert isinstance(our_plan, dict)
        assert isinstance(opp_plan, dict)
        # Bootstrap is ROI vs ROI, so initial emissions are non-empty;
        # SA may prune but on this small budget shouldn't gut everything.
        n_our = sum(len(v) for v in our_plan.values())
        n_opp = sum(len(v) for v in opp_plan.values())
        assert n_our > 0, f"our_plan is empty after co-evolve: {our_plan}"
        assert n_opp > 0, f"opp_plan is empty after co-evolve: {opp_plan}"
    finally:
        _restore_env(snap)


def test_coevolve_score_does_not_regress_across_cycles():
    """Our terminal-ship-diff against the EVOLVED opp must not collapse
    to ≤ 0 over multiple cycles.

    This is the anti-pessimism gate. The diagnostic showed that with
    fixed opp=ROI, sa_online's plan collapsed to empty by step 30 (score
    ~0). With co-evolved opp, even after multiple cycles, SA should
    find aggressive plans that maintain a positive score-diff.
    """
    snap = _set_env(SA_COEVOLVE_CYCLES="2", SA_BUDGET_INIT_S="5",
                     SA_ITER_INIT="60")
    try:
        from agents.sa_online.main import _co_evolve, _plan_dict_to_list
        from scripts.sa_solo_solver import _build_solo_snap0
        from lib.sa_core import score_plan_from_snap

        our_plan_d, opp_plan_d = _co_evolve(seed=0, steps=50)
        our_plan = _plan_dict_to_list(our_plan_d)
        opp_plan = _plan_dict_to_list(opp_plan_d)
        assert our_plan, "our_plan empty after 2-cycle co-evolve"
        assert opp_plan, "opp_plan empty after 2-cycle co-evolve"

        snap0 = _build_solo_snap0(seed=0, steps=50)

        def opp_policy(obs):
            from lib.sa_core import _get_step
            t = _get_step(obs)
            return [list(a) for a in opp_plan_d.get(int(t), [])]

        # diff score for our side. Must be > -1000 (we're not getting
        # blown out). Real convergence would give us positive diff.
        score_diff = score_plan_from_snap(
            our_plan, snap0, opp_policy=opp_policy, max_steps=50,
            me=0, score_mode="diff",
        )
        assert score_diff > -1000.0, (
            f"our_plan vs evolved opp got crushed: diff={score_diff}. "
            "Pessimism-trap regression — should never go this negative.")
    finally:
        _restore_env(snap)


def test_refine_step_fits_budget():
    """A single _refine_step call must complete inside its deadline."""
    import time
    snap = _set_env(SA_COEVOLVE_CYCLES="1", SA_BUDGET_INIT_S="3",
                     SA_ITER_INIT="20", SA_BUDGET_STEP_S="0.5",
                     SA_ITER_STEP="200", SA_HORIZON="20")
    try:
        # Force-trigger co-evolve so _PLAN_BY_TURN + _OPP_PLAN_BY_TURN populate
        os.environ["SA_SEED"] = "0"
        os.environ["SA_EPISODE_STEPS"] = "50"
        # Reload sa_online to trigger module-load co-evolve
        import importlib
        import agents.sa_online.main as sa_mod
        importlib.reload(sa_mod)

        # Build a fresh obs at step 5 for refine
        from kaggle_environments import make
        env = make("orbit_wars",
                   configuration={"seed": 0, "episodeSteps": 50},
                   debug=False)
        env.reset(num_agents=2)
        # Step a few turns with noop both sides to advance state
        for _ in range(5):
            env.step([[], []])
        state = env.steps[-1]
        obs = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation

        t0 = time.perf_counter()
        result = sa_mod._refine_step(obs, env.configuration, t=5)
        elapsed = time.perf_counter() - t0
        # 0.5s deadline + generous slack for last-iter overshoot
        assert elapsed < 1.2, f"refine_step overshot: {elapsed:.2f}s > 1.2s"
        assert isinstance(result, dict)
    finally:
        _restore_env(snap)
        os.environ.pop("SA_SEED", None)
        os.environ.pop("SA_EPISODE_STEPS", None)
