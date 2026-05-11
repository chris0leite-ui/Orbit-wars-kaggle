"""Tests for lib/lookahead — Sim<K> scorer + candidate enumerator."""

from __future__ import annotations

import math

import pytest

from lib.lookahead import (
    enumerate_drop_one_candidates,
    env_from_obs,
    score_action,
    _ship_total_by_owner,
)


# ---------------------------------------------------------------------------
# enumerate_drop_one_candidates — pure list manipulation
# ---------------------------------------------------------------------------


def test_drop_one_candidates_empty_input_yields_single_empty():
    """No launches → one candidate (the empty action), no scoring needed."""
    assert enumerate_drop_one_candidates([]) == [[]]


def test_drop_one_candidates_single_launch_yields_keep_and_drop():
    """One launch → two candidates: keep it, drop it."""
    action = [[0, 1.57, 10]]
    cands = enumerate_drop_one_candidates(action)
    assert len(cands) == 2
    assert cands[0] == action
    assert cands[1] == []


def test_drop_one_candidates_three_launches_yields_four():
    """N launches → N+1 candidates: incumbent + N drop-one variants."""
    action = [[0, 0.0, 5], [1, 1.0, 10], [2, 2.0, 15]]
    cands = enumerate_drop_one_candidates(action)
    assert len(cands) == 4
    assert cands[0] == action
    # Each drop-one candidate omits exactly one launch
    for i in range(3):
        omitted = action[i]
        assert omitted not in cands[i + 1]
        assert len(cands[i + 1]) == 2


def test_drop_one_candidates_does_not_mutate_input():
    action = [[0, 0.0, 5], [1, 1.0, 10]]
    original = [list(m) for m in action]
    enumerate_drop_one_candidates(action)
    assert action == original


# ---------------------------------------------------------------------------
# _ship_total_by_owner — scoring helper
# ---------------------------------------------------------------------------


def test_ship_total_excludes_neutral_planets():
    """Neutral (-1) planets shouldn't contribute to any player's total."""
    obs = {
        "planets": [
            (0, 0, 10.0, 10.0, 1.5, 50, 2),   # P0 owned, 50 ships
            (1, -1, 90.0, 90.0, 1.5, 100, 3),  # neutral, ignored
            (2, 1, 50.0, 50.0, 1.5, 30, 2),    # P1 owned, 30 ships
        ],
        "fleets": [],
    }
    totals = _ship_total_by_owner(obs)
    assert totals == {0: 50.0, 1: 30.0}


def test_ship_total_sums_planets_and_fleets():
    obs = {
        "planets": [(0, 0, 10.0, 10.0, 1.5, 50, 2)],
        "fleets": [(100, 0, 5.0, 5.0, 0.0, 0, 20)],
    }
    totals = _ship_total_by_owner(obs)
    assert totals == {0: 70.0}


# ---------------------------------------------------------------------------
# env_from_obs — reconstruction parity (slow; uses kaggle_environments)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_env_from_obs_matches_real_env_after_one_step():
    """Reconstructed env stepped one tick = real env stepped one tick.

    Validates the assumption that the agent-visible obs + configuration
    is enough to rebuild a steppable mirror without the seed.
    """
    import sys, importlib.util
    from kaggle_environments import make

    spec = importlib.util.spec_from_file_location(
        "_v2_under_test", "agents/v2/main.py"
    )
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)

    real = make("orbit_wars", configuration={"seed": 42}, debug=False)
    real.reset(num_agents=2)
    for _ in range(20):
        a0 = v2.agent(real.state[0].observation)
        a1 = v2.agent(real.state[1].observation)
        real.step([a0, a1])

    obs = real.state[0].observation
    cfg = real.configuration
    recon = env_from_obs(obs, cfg)

    # Step both with identical agents
    a0r = v2.agent(real.state[0].observation)
    a1r = v2.agent(real.state[1].observation)
    a0c = v2.agent(recon.state[0].observation)
    a1c = v2.agent(recon.state[1].observation)
    real.step([a0r, a1r])
    recon.step([a0c, a1c])

    r_planets = sorted(tuple(p) for p in real.state[0].observation["planets"])
    c_planets = sorted(tuple(p) for p in recon.state[0].observation["planets"])
    r_fleets = sorted(tuple(f) for f in real.state[0].observation["fleets"])
    c_fleets = sorted(tuple(f) for f in recon.state[0].observation["fleets"])
    assert r_planets == c_planets
    assert r_fleets == c_fleets


# ---------------------------------------------------------------------------
# score_action — full forward-sim integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_score_action_runs_K_steps_and_returns_finite_scalar():
    import sys, importlib.util
    from kaggle_environments import make

    spec = importlib.util.spec_from_file_location(
        "_v2_under_test_2", "agents/v2/main.py"
    )
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)

    real = make("orbit_wars", configuration={"seed": 42}, debug=False)
    real.reset(num_agents=2)
    for _ in range(20):
        a0 = v2.agent(real.state[0].observation)
        a1 = v2.agent(real.state[1].observation)
        real.step([a0, a1])

    obs = real.state[0].observation
    cfg = real.configuration
    recon = env_from_obs(obs, cfg)
    incumbent = v2.agent(obs)

    score_incumbent = score_action(
        recon, incumbent, K=10, my_id=0, policy=v2.agent
    )
    score_empty = score_action(recon, [], K=10, my_id=0, policy=v2.agent)
    # Both should be finite and on the same scale (typical mid-game total
    # ship counts are O(100); deltas O(10-50)).
    assert math.isfinite(score_incumbent)
    assert math.isfinite(score_empty)
    # The clone is independent of the source env — the real env's state
    # mustn't have advanced.
    assert real.state[0].observation["step"] == 20
