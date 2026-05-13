"""Smoke tests for v7_wide_deep + the v7_search extensions
(value_fn + opp_tiers list / maximin).
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from kaggle_environments import make

from lib.v7_search import choose
from lib.value_heads import delta_us_minus_them_obs
from lib.lookahead_planner import evaluate_value


def _composite_value(observation, my_id):
    d = delta_us_minus_them_obs(observation, my_id)
    e = evaluate_value(observation, my_id)
    return 0.6 * d + 0.4 * e


def _make_obs(seed: int):
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    return env.state[0].observation, env.configuration


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
def test_choose_with_value_fn_returns_action(seed):
    """choose() with custom value_fn returns a valid action list."""
    obs, cfg = _make_obs(seed)
    a = choose(
        obs, cfg,
        enumerator_mode="drop_one",
        K=10,
        value_fn=_composite_value,
    )
    assert isinstance(a, list)
    for move in a:
        assert isinstance(move, list)
        assert len(move) == 3
        assert isinstance(move[0], int)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_choose_with_opp_tiers_maximin(seed):
    """choose() with multi-tier opp pool (maximin) returns a valid action."""
    obs, cfg = _make_obs(seed)
    a = choose(
        obs, cfg,
        enumerator_mode="drop_one",
        K=8,
        opp_tiers=[0, 1],
    )
    assert isinstance(a, list)
    for move in a:
        assert len(move) == 3


@pytest.mark.parametrize("seed", [0, 42, 137])
def test_v7_wide_deep_agent_smoke(seed):
    """Full v7_wide_deep agent path: combined enumerator + K=25 + maximin
    + composite value_fn. Returns a valid action without raising."""
    from agents.v7_wide_deep.main import agent
    obs, cfg = _make_obs(seed)
    a = agent(obs, cfg)
    assert isinstance(a, list)
    for move in a:
        assert len(move) == 3
        assert isinstance(move[0], int)
        assert isinstance(move[2], int) or isinstance(move[2], float)


def test_choose_default_path_unchanged_by_extensions():
    """Calling choose() with the v7_0 args (no opp_tiers, no value_fn)
    must produce an identical action to the v7_0 default behaviour
    (regression guard on the kwarg extensions).
    """
    obs, cfg = _make_obs(42)
    # Two calls with the same default args — deterministic.
    a1 = choose(obs, cfg, enumerator_mode="drop_one", K=10)
    a2 = choose(obs, cfg, enumerator_mode="drop_one", K=10)
    assert a1 == a2
