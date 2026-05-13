"""Sub-phase 1 (JAX engine port) scaffolding tests.

Validates:
1. GameState Pytree construction shapes match the SHAPES dict.
2. `scalar_to_jax` builds a GameState from a freshly-initialised scalar
   game without crashing.
3. `jax_to_scalar_obs` round-trips planets/fleets back to a usable
   scalar-style obs.
4. The pre-computed comet schedule has the expected structure
   (5 spawns, mostly valid).

This is foundation-only; per-step JAX physics arrives in sub-phases 2+.
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv
from lib.game.interpreter import interpreter as scalar_interpreter, COMET_SPAWN_STEPS
from lib.game.jax import (
    GameState,
    MAX_PLANETS,
    MAX_FLEETS,
    NUM_COMET_SPAWNS,
    scalar_to_jax,
    jax_to_scalar,
)
from lib.game.jax.jax_types import SHAPES


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
def test_scalar_to_jax_shapes(seed):
    """Every field of GameState matches its declared shape."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    gs = scalar_to_jax(env.state, env.info["seed"])

    for field, expected_shape in SHAPES.items():
        arr = getattr(gs, field)
        assert hasattr(arr, "shape"), f"{field} not a JAX/numpy array"
        assert arr.shape == expected_shape, (
            f"{field}: got {arr.shape}, expected {expected_shape}"
        )


@pytest.mark.parametrize("seed", [0, 42, 137])
def test_scalar_to_jax_planet_count(seed):
    """alive-mask count matches the scalar planet count."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    gs = scalar_to_jax(env.state, env.info["seed"])

    scalar_count = len(env.state[0].observation.planets)
    alive_count = int(np.sum(np.asarray(gs.planets_alive)))
    assert alive_count == scalar_count


@pytest.mark.parametrize("seed", [0, 42, 137])
def test_comet_schedule_populated(seed):
    """All 5 comet spawn slots have a step number; most should be valid."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    gs = scalar_to_jax(env.state, env.info["seed"])

    comet_steps = np.asarray(gs.comet_step)
    assert list(comet_steps) == COMET_SPAWN_STEPS, (
        f"comet_step mismatch: got {comet_steps}, expected {COMET_SPAWN_STEPS}"
    )

    # At least 3 of 5 spawns should produce valid paths in a typical
    # game; 2026-05-12 audit replays show 4-5 valid is common.
    valid = np.asarray(gs.comet_valid)
    assert int(valid.sum()) >= 3, f"only {valid.sum()} valid spawn(s)"

    # Each valid spawn has 4 paths with non-zero lengths.
    paths_len = np.asarray(gs.comet_paths_len)
    for k in range(NUM_COMET_SPAWNS):
        if valid[k]:
            for j in range(4):
                assert paths_len[k, j] > 0, f"spawn {k} path {j} empty"
                assert paths_len[k, j] <= 40, f"spawn {k} path {j} too long"
        # All ship counts in [1, 99] when valid (per orbit_wars rules).
        if valid[k]:
            assert 1 <= int(gs.comet_ships[k]) <= 99


@pytest.mark.parametrize("seed", [0, 42])
def test_roundtrip_preserves_planet_fleet_counts(seed):
    """scalar → jax → scalar preserves planet & fleet membership."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    gs = scalar_to_jax(env.state, env.info["seed"])
    obs_back = jax_to_scalar(gs)

    assert len(obs_back["planets"]) == len(env.state[0].observation.planets)
    assert len(obs_back["fleets"]) == len(env.state[0].observation.fleets)
    assert obs_back["step"] == int(env.state[0].observation.get("step", 0))
    assert obs_back["angular_velocity"] == pytest.approx(
        env.state[0].observation.angular_velocity
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_roundtrip_planet_positions(seed):
    """Planet (x, y, owner, ships, prod, radius) round-trip values are
    preserved to float32 precision (which is what JAX storage uses).
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    gs = scalar_to_jax(env.state, env.info["seed"])
    obs_back = jax_to_scalar(gs)

    src_planets = env.state[0].observation.planets
    # Same count; positions should match within float32 tolerance.
    for i, (p_src, p_jax) in enumerate(zip(src_planets, obs_back["planets"])):
        assert p_jax[1] == p_src[1], f"planet[{i}] owner"
        assert p_jax[2] == pytest.approx(p_src[2], rel=1e-5, abs=1e-5), (
            f"planet[{i}] x"
        )
        assert p_jax[3] == pytest.approx(p_src[3], rel=1e-5, abs=1e-5), (
            f"planet[{i}] y"
        )
        assert p_jax[4] == pytest.approx(p_src[4], rel=1e-5, abs=1e-5), (
            f"planet[{i}] radius"
        )
        assert p_jax[5] == p_src[5], f"planet[{i}] ships"
        assert p_jax[6] == p_src[6], f"planet[{i}] prod"
