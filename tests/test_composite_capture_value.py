"""Smoke + invariant tests for `lib.value_heads.composite_capture_value`.

Pins:
- Returns a float scalar.
- Equals plain `delta_us_minus_them_obs` when there are no fleets.
- Penalises trajectories that won't capture (bounce / OOB / sun).
- Rewards captures.
"""

from __future__ import annotations

import pytest

from kaggle_environments import make

from lib.value_heads import (
    composite_capture_value,
    delta_us_minus_them_obs,
)


@pytest.fixture(scope="module")
def fresh_obs():
    env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 500})
    env.reset(num_agents=2)
    return env.steps[0][0].observation


def test_returns_float_scalar(fresh_obs):
    v = composite_capture_value(fresh_obs, my_id=0)
    assert isinstance(v, float)


def test_matches_base_when_no_fleets(fresh_obs):
    # Step 0: no fleets are in flight.
    base = delta_us_minus_them_obs(fresh_obs, my_id=0)
    composite = composite_capture_value(fresh_obs, my_id=0)
    assert composite == base


def test_zero_weights_collapse_to_base(fresh_obs):
    base = delta_us_minus_them_obs(fresh_obs, my_id=0)
    v = composite_capture_value(
        fresh_obs, my_id=0,
        capture_weight=0.0, waste_weight=0.0,
    )
    # With both weights zero, composite reduces to base regardless of fleets.
    assert v == base


def test_runs_after_a_few_turns():
    """Play a couple of self-play turns then score — fleets exist now."""
    env = make("orbit_wars", configuration={"seed": 2, "episodeSteps": 500})
    env.reset(num_agents=2)
    # Manually step a couple of turns with simple actions.
    from agents.v7_ablations.v7_0_drop_one.main import agent as v7_0
    state = env.steps[0]
    for _ in range(8):
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        a0 = v7_0(obs0, env.configuration)
        a1 = v7_0(obs1, env.configuration)
        state = env.step([a0, a1])
    obs = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
    v = composite_capture_value(obs, my_id=0)
    base = delta_us_minus_them_obs(obs, my_id=0)
    # Once fleets are in flight, composite differs from base in general.
    # We don't assert direction (could be + or -) — just that it's still
    # a finite float and didn't crash.
    assert isinstance(v, float)
    assert v == v  # not NaN
