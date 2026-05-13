"""v7.4 4P-aware rollout smoke tests."""

from __future__ import annotations

import random
import time

import pytest
from kaggle_environments import make

from lib import fast_sim
from lib.v7_search import (
    choose_4p,
    choose_with_4p,
    score_candidate_4p,
)


def _warmed_4p_env(seed: int = 42, warmup: int = 15):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=4)
    rng = random.Random(seed)
    for _ in range(warmup):
        obs = env.state[0].observation
        acts = []
        for p in range(4):
            ll = [
                [pl[0], rng.uniform(0, 6.283), int(pl[5] // 3)]
                for pl in obs["planets"]
                if pl[1] == p and pl[5] > 6 and rng.random() < 0.3
            ]
            acts.append(ll)
        env.step(acts)
    return env


def test_score_candidate_4p_requires_num_seats_4():
    """Wrong num_seats raises a clear error."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    snap = fast_sim.from_obs(env.state[0].observation, env.configuration,
                              episode_seed=env.info["seed"], num_seats=2)
    with pytest.raises(ValueError):
        score_candidate_4p(snap, [], my_id=0, K=3)


def test_score_candidate_4p_runs_and_returns_float():
    env = _warmed_4p_env()
    snap = fast_sim.from_obs(env.state[0].observation, env.configuration,
                              episode_seed=env.info["seed"], num_seats=4)
    score = score_candidate_4p(snap, [], my_id=0, K=3)
    assert isinstance(score, float)


def test_choose_4p_finishes_in_budget():
    """In a warmed 4P state, choose_4p completes within ~700 ms."""
    env = _warmed_4p_env()
    t0 = time.perf_counter()
    action = choose_4p(env.state[0].observation, env.configuration,
                       K=6, wallclock_ms=700.0)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert isinstance(action, list)
    assert dt_ms < 1500, f"4P choose took {dt_ms:.0f} ms"


def test_choose_with_4p_routes_2p():
    """In 2P, choose_with_4p must invoke the 2P maximin path
    (not crash on 4P-only code)."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    # Warm a little
    rng = random.Random(0)
    for _ in range(10):
        obs = env.state[0].observation
        a = [[p[0], rng.uniform(0, 6.28), int(p[5]//2)]
             for p in obs["planets"] if p[1] == 0 and p[5] > 5 and rng.random() < 0.3]
        b = [[p[0], rng.uniform(0, 6.28), int(p[5]//2)]
             for p in obs["planets"] if p[1] == 1 and p[5] > 5 and rng.random() < 0.3]
        env.step([a, b])
    action = choose_with_4p(env.state[0].observation, env.configuration,
                             K_2p=6, K_4p=4, wallclock_ms=700.0)
    assert isinstance(action, list)


def test_choose_with_4p_routes_4p():
    """In 4P, choose_with_4p must invoke the 4P drop-one path."""
    env = _warmed_4p_env()
    action = choose_with_4p(env.state[0].observation, env.configuration,
                             K_2p=6, K_4p=4, wallclock_ms=700.0)
    assert isinstance(action, list)
