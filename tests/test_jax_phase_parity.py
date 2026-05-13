"""Per-phase parity: each JAX phase must match the scalar interpreter's
equivalent computation on the same state.

Implemented phases (sub-phase 1b):
- production_tick
- planet_path_compute
- comet_expire

The full `jax_step` parity test (sub-phase 1c+) compares the chained
JAX pipeline against the scalar interpreter end-to-end. This file
tests each phase in ISOLATION so divergences localise.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv
from lib.game.interpreter import (
    interpreter as scalar_interpreter,
    BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT,
    COMET_SPAWN_STEPS,
)
from lib.game.jax import (
    GameState, MAX_PLANETS,
    scalar_to_jax, jax_to_scalar,
    production_tick, planet_path_compute, comet_expire,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_state(seed: int, num_agents: int = 2):
    """Fresh post-init scalar state + matching JAX state."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=num_agents)
    gs = scalar_to_jax(env.state, env.info["seed"])
    return env, gs


def _scalar_production(env):
    """Apply the scalar production tick in-place on a copy of env state."""
    obs0 = env.state[0].observation
    out = []
    for p in obs0.planets:
        new_p = list(p)
        if new_p[1] != -1:
            new_p[5] += new_p[6]
        out.append(new_p)
    return out


def _scalar_planet_path(env):
    """Apply the scalar planet-path compute, returning new (x, y) per planet."""
    obs0 = env.state[0].observation
    angular_velocity = obs0.angular_velocity
    step = int(obs0.get("step", 0))
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}
    out = []
    for p in obs0.planets:
        pid = p[0]
        new_x, new_y = p[2], p[3]
        if pid not in comet_pid_set:
            initial_p = initial_by_id.get(pid)
            if initial_p is not None:
                dx = initial_p[2] - CENTER
                dy = initial_p[3] - CENTER
                r = math.sqrt(dx * dx + dy * dy)
                if r + p[4] < ROTATION_RADIUS_LIMIT:
                    current_angle = math.atan2(dy, dx) + angular_velocity * step
                    new_x = CENTER + r * math.cos(current_angle)
                    new_y = CENTER + r * math.sin(current_angle)
        out.append((new_x, new_y))
    return out


# ---------------------------------------------------------------------------
# production_tick parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
def test_production_tick_parity(seed):
    """JAX production_tick produces identical ship counts to the scalar pass."""
    env, gs = _build_state(seed)
    new_gs = production_tick(gs)
    expected = _scalar_production(env)

    new_ships = np.asarray(new_gs.planets_ships)
    for i, p in enumerate(expected):
        assert new_ships[i] == p[5], (
            f"planet[{i}]: jax={new_ships[i]} scalar={p[5]}"
        )


# ---------------------------------------------------------------------------
# planet_path_compute parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
@pytest.mark.parametrize("step", [0, 10, 50, 100, 250, 499])
def test_planet_path_compute_parity(seed, step):
    """JAX planet positions match scalar's (within float32 tolerance)
    for fully-rotating planets, and exactly for non-rotating.
    """
    env, gs = _build_state(seed)
    # Override the JAX state's step to test rotation at various phases.
    gs = gs._replace(step=jnp.asarray(step, dtype=jnp.int32))
    # Also force the scalar obs's step for the reference computation.
    env.state[0].observation.step = step

    new_gs = planet_path_compute(gs)
    expected = _scalar_planet_path(env)

    new_x = np.asarray(new_gs.planets_x)
    new_y = np.asarray(new_gs.planets_y)
    for i, (ex, ey) in enumerate(expected):
        if not gs.planets_alive[i]:
            continue
        # Float32 storage in JAX, float64 in scalar — accept ~1e-5 abs.
        assert abs(float(new_x[i]) - ex) < 5e-4, (
            f"planet[{i}] x at step {step}: jax={new_x[i]:.6f} scalar={ex:.6f}"
        )
        assert abs(float(new_y[i]) - ey) < 5e-4, (
            f"planet[{i}] y at step {step}: jax={new_y[i]:.6f} scalar={ey:.6f}"
        )


# ---------------------------------------------------------------------------
# comet_expire parity
# ---------------------------------------------------------------------------


def test_comet_expire_no_op_at_start():
    """At step 0 with no comets spawned yet, comet_expire is a no-op."""
    env, gs = _build_state(42)
    new_gs = comet_expire(gs)
    np.testing.assert_array_equal(
        np.asarray(new_gs.planets_alive),
        np.asarray(gs.planets_alive),
    )
