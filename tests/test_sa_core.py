"""Unit + parity tests for lib/sa_core.py.

The critical-rigor gate of the sa_online architecture. Every claim
made by the SA layer has a test here. If any of these fail, the SA
agent is broken — don't ship.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from lib.sa_core import (  # noqa: E402
    _noop_policy,
    perturb,
    score_plan_from_snap,
    simulated_anneal_online,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_snap0(seed: int = 0, steps: int = 50):
    """Build a turn-0 snapshot for seed/steps — used by parity tests."""
    from kaggle_environments import make
    from lib.fast_sim import from_obs as fs_from_obs

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    return fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)


def _initial_planets(seed: int = 0, steps: int = 50):
    """Pull initial_planets list from a fresh env at the same seed."""
    from kaggle_environments import make

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    od = obs0 if isinstance(obs0, dict) else dict(obs0)
    return [list(p) for p in (od.get("planets") or [])]


# ---------------------------------------------------------------------------
# score_plan_from_snap — determinism + the critical solo-vs-online parity
# ---------------------------------------------------------------------------


def test_score_deterministic():
    """Same inputs → same output. No hidden state."""
    snap = _build_snap0(seed=0, steps=50)
    emissions = []  # empty plan
    s1 = score_plan_from_snap(emissions, snap, opp_policy=None, max_steps=50)
    s2 = score_plan_from_snap(emissions, snap, opp_policy=None, max_steps=50)
    assert s1 == s2, f"non-deterministic: {s1} vs {s2}"
    assert s1 > 0, f"empty-plan score should be > 0 (own home produces): {s1}"


def test_score_solo_parity():
    """score_plan_from_snap(plan, snap0, noop, T) == old solo score_plan."""
    from scripts.sa_solo_solver import score_plan
    snap = _build_snap0(seed=0, steps=50)
    emissions = []
    online_score = score_plan_from_snap(emissions, snap,
                                         opp_policy=None, max_steps=50)
    solo_score = score_plan(emissions, seed=0, steps=50)
    # Both build the same snap and roll the same empty plan against noop.
    # Result must be identical.
    assert online_score == solo_score, \
        f"parity break: online={online_score} solo={solo_score}"


def test_score_snap_unchanged_after_call():
    """score_plan_from_snap must NOT mutate the input snap."""
    from lib.fast_sim import ship_totals
    snap = _build_snap0(seed=0, steps=50)
    ships_before = ship_totals(snap).get(0, 0.0)
    _ = score_plan_from_snap([], snap, opp_policy=None, max_steps=50)
    ships_after = ship_totals(snap).get(0, 0.0)
    assert ships_before == ships_after, \
        f"snap mutated: {ships_before} -> {ships_after}"


# ---------------------------------------------------------------------------
# perturb — shape + range guarantees per op
# ---------------------------------------------------------------------------


def test_perturb_remove_shrinks():
    """remove op decreases plan length by 1."""
    plan = [(5, [0, 0.0, 10]), (10, [1, 0.5, 20])]
    rng = random.Random(0)
    # Disable add-op (initial_planets=None) so we only get remove/ships/shift/angle.
    # Find a remove draw deterministically by trying a few rng seeds.
    for trial in range(50):
        rng = random.Random(trial)
        new = perturb(list(plan), rng,
                       initial_planets=None, t_start=0, t_end=20)
        if len(new) == len(plan) - 1:
            return  # found a remove
    pytest.fail("perturb never produced 'remove' op over 50 trials")


def test_perturb_add_inserts():
    """add op increases plan length by 1, new emission has valid shape."""
    plan = [(5, [0, 0.0, 10])]
    initial_planets = _initial_planets(seed=0, steps=50)
    for trial in range(100):
        rng = random.Random(trial)
        new = perturb(list(plan), rng,
                       initial_planets=initial_planets,
                       t_start=0, t_end=50)
        if len(new) == len(plan) + 1:
            # Validate new emission shape
            added = new[-1]
            assert isinstance(added, tuple)
            turn, action = added
            assert 0 <= int(turn) < 50, f"turn {turn} out of range [0, 50)"
            assert len(action) == 3, f"action wrong shape: {action}"
            return
    pytest.fail("perturb never produced 'add' op over 100 trials")


def test_perturb_add_respects_t_range():
    """add op only generates turns in [t_start, t_end)."""
    initial_planets = _initial_planets(seed=0, steps=50)
    t_start_test = 30
    t_end_test = 50
    n_adds = 0
    for trial in range(200):
        rng = random.Random(trial)
        new = perturb([], rng,
                       initial_planets=initial_planets,
                       t_start=t_start_test, t_end=t_end_test)
        if len(new) == 1:  # add fired
            turn, _ = new[0]
            assert t_start_test <= int(turn) < t_end_test, \
                f"add violated range: turn={turn} not in [{t_start_test}, {t_end_test})"
            n_adds += 1
    assert n_adds >= 5, f"expected many add draws over 200 trials, got {n_adds}"


# ---------------------------------------------------------------------------
# simulated_anneal_online — monotone-best + parity with solo wrapper
# ---------------------------------------------------------------------------


def test_sa_online_monotone_best():
    """best_score never decreases across iterations (history rows)."""
    snap = _build_snap0(seed=0, steps=50)
    initial_planets = _initial_planets(seed=0, steps=50)
    _, _, history = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=30, t0=100.0, cooling=0.95,
        rng=random.Random(42),
        start_step=0, initial_planets=initial_planets,
    )
    best_so_far = -float("inf")
    for (_, _, best) in history:
        assert best >= best_so_far, \
            f"monotone violation: best dropped from {best_so_far} to {best}"
        best_so_far = best


def test_sa_online_vs_solo_parity():
    """simulated_anneal_online(... noop opp ...) ≡ scripts/sa_solo_solver.simulated_anneal.

    Both should yield the same trajectory given identical RNG (same seed
    drawn from same Random instance). This is THE critical parity test —
    if it fails, the refactor changed the SA behaviour and the existing
    solo results no longer reproduce.
    """
    from scripts.sa_solo_solver import simulated_anneal
    initial_planets = _initial_planets(seed=0, steps=50)
    n_iter = 20

    # Online path
    snap = _build_snap0(seed=0, steps=50)
    rng_online = random.Random(42)
    _, best_online, _ = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=n_iter, t0=100.0, cooling=0.95,
        rng=rng_online, start_step=0, initial_planets=initial_planets,
    )

    # Solo wrapper (which delegates internally to sa_online)
    rng_solo = random.Random(42)
    _, best_solo, _ = simulated_anneal(
        initial_plan=[], seed=0, steps=50, n_iterations=n_iter,
        t0=100.0, cooling=0.95, rng=rng_solo, initial_planets=initial_planets,
    )

    assert best_online == best_solo, \
        f"SA parity break: online={best_online} solo={best_solo}"
