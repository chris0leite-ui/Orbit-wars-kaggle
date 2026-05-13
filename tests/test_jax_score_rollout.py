"""Sub-phase 5: end-to-end JAX rollout sanity tests.

These tests verify the composed pipeline (state → score matrices →
settle_plan → mechanism → jax_step) executes without errors and
produces reasonable values. Strict parity vs scalar `score_candidate`
is sub-phase 7 (it requires the full lead_aim_v2 / sun_avoid stack).
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import pytest

from kaggle_environments import make

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_score import (
    policy_step_jax,
    rollout_step_jax,
    score_candidate_jax,
    value_delta_ships,
)


def _light_play(env, n_steps=25, rng_seed=7, num_agents=2):
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for ps in range(num_agents):
            mv = []
            for p in env.state[ps].observation.planets:
                if p[1] == ps and p[5] > 5 and rng.random() < 0.2:
                    mv.append([p[0], rng.uniform(0, 2 * math.pi),
                               max(1, int(p[5] * rng.uniform(0.1, 0.3)))])
            actions.append(mv)
        env.step(actions)


@pytest.mark.parametrize("seed", [3, 42])
def test_value_delta_ships_matches_scalar(seed):
    """value_delta_ships on initial JAX state matches scalar
    (planet ships + alive fleet ships per side)."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    obs = env.state[0].observation
    my_id = 0
    my_total = 0
    opp_total = 0
    for p in obs.planets:
        owner = p[1]
        ships = p[5]
        if owner == my_id:
            my_total += int(ships)
        elif owner != -1:
            opp_total += int(ships)
    for f in obs.fleets:
        owner = f[1]
        ships = f[6]
        if owner == my_id:
            my_total += int(ships)
        elif owner != -1:
            opp_total += int(ships)
    expected = my_total - opp_total

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_val = int(value_delta_ships(gs, my_id=my_id))
    assert jax_val == expected, (
        f"seed {seed}: scalar={expected} jax={jax_val}"
    )


def test_policy_step_emits_reasonable_actions():
    """policy_step_jax produces a non-negative-length action list and
    none of the emitted intents over-spend their source."""
    seed = 42
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    emit, _ = policy_step_jax(gs, my_id=0)

    import numpy as np
    planets_id = np.asarray(gs.planets_id)
    planets_ships = np.asarray(gs.planets_ships)
    planets_owner = np.asarray(gs.planets_owner)
    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(planets_id)
        if pid >= 0
    }
    src_used = {}
    for e in emit:
        src_slot = pid_to_slot[e["src_pid"]]
        assert int(planets_owner[src_slot]) == 0, "emitted from non-owned planet"
        src_used.setdefault(src_slot, 0)
        src_used[src_slot] += int(e["ships"])
        assert src_used[src_slot] <= int(planets_ships[src_slot]), (
            "per-source ship budget exceeded"
        )


def test_rollout_step_advances_state():
    """rollout_step_jax actually advances the env step by 1 and returns
    a state with consistent shapes."""
    env = make("orbit_wars", configuration={"seed": 11})
    env.reset(num_agents=2)
    _light_play(env, n_steps=15, rng_seed=99)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    initial_step = int(gs.step)
    new_s = rollout_step_jax(gs, my_id=0)
    assert int(new_s.step) == initial_step + 1
    assert new_s.planets_x.shape == gs.planets_x.shape
    assert new_s.fleets_x.shape == gs.fleets_x.shape


def test_score_candidate_returns_finite_value():
    """K=5 rollout finishes and returns a finite ship-delta value."""
    env = make("orbit_wars", configuration={"seed": 42})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=42 * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    emit, _ = policy_step_jax(gs, my_id=0)
    val = score_candidate_jax(gs, emit, K=5, my_id=0)
    assert math.isfinite(val), f"score_candidate returned non-finite: {val}"
