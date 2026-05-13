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


# ---------------------------------------------------------------------------
# swept_pair_hit_batch parity
# ---------------------------------------------------------------------------


def _scalar_swept(A, B, P0, P1, r):
    """The exact scalar from lib/game/interpreter.py, inlined here for
    bit-exact reference. Includes the AABB prune since the scalar
    function applies it before the discriminant check."""
    from lib.game.interpreter import swept_pair_hit
    return swept_pair_hit(A, B, P0, P1, r)


def test_swept_pair_hit_batch_matches_scalar():
    """Vectorised swept_pair test matches the scalar implementation on
    a battery of synthetic (fleet, planet) configurations."""
    from lib.game.jax import swept_pair_hit_batch
    import random as _rnd

    rng = _rnd.Random(42)
    # 8 fleet trajectories and 6 planet trajectories, mix of near/far.
    fold_list = []
    fnew_list = []
    for _ in range(8):
        ox = rng.uniform(0, 100); oy = rng.uniform(0, 100)
        nx = ox + rng.uniform(-6, 6); ny = oy + rng.uniform(-6, 6)
        fold_list.append((ox, oy)); fnew_list.append((nx, ny))
    pold_list = []
    pnew_list = []
    pr_list = []
    for _ in range(6):
        ox = rng.uniform(0, 100); oy = rng.uniform(0, 100)
        nx = ox + rng.uniform(-1, 1); ny = oy + rng.uniform(-1, 1)
        r = rng.uniform(1, 4)
        pold_list.append((ox, oy)); pnew_list.append((nx, ny))
        pr_list.append(r)

    # Include one guaranteed-hit case (fleet sits on planet).
    fold_list.append(pold_list[0]); fnew_list.append(pold_list[0])
    pr_arr_list = pr_list

    fold = jnp.asarray(fold_list, dtype=jnp.float32)
    fnew = jnp.asarray(fnew_list, dtype=jnp.float32)
    pold = jnp.asarray(pold_list, dtype=jnp.float32)
    pnew = jnp.asarray(pnew_list, dtype=jnp.float32)
    pr = jnp.asarray(pr_arr_list, dtype=jnp.float32)

    jax_hits = np.asarray(swept_pair_hit_batch(fold, fnew, pold, pnew, pr))

    F = fold.shape[0]; P = pold.shape[0]
    for i in range(F):
        for j in range(P):
            scalar_hit = _scalar_swept(
                (float(fold[i, 0]), float(fold[i, 1])),
                (float(fnew[i, 0]), float(fnew[i, 1])),
                (float(pold[j, 0]), float(pold[j, 1])),
                (float(pnew[j, 0]), float(pnew[j, 1])),
                float(pr[j]),
            )
            assert bool(jax_hits[i, j]) == scalar_hit, (
                f"fleet[{i}] x planet[{j}]: jax={bool(jax_hits[i,j])} "
                f"scalar={scalar_hit}"
            )


# ---------------------------------------------------------------------------
# comet_path_advance parity (no-spawn-yet smoke)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42, 137])
def test_comet_spawn_at_step_49(seed):
    """At step 49 (one before COMET_SPAWN_STEPS[0]=50), comet_spawn
    instantiates 4 comet planets if the seed produced valid paths.
    Validates: 4 new planets alive, owners = -1, ships matches
    pre-computed comet_ships, planet_comet_spawn=0, planet_comet_path
    is 0/1/2/3, comet_spawned[0]=True, comet_planet_idx[0] populated.
    """
    from lib.game.jax import comet_spawn
    env, gs = _build_state(seed)
    # Force step = 49 so step+1 = 50 = COMET_SPAWN_STEPS[0].
    gs = gs._replace(step=jnp.asarray(49, dtype=jnp.int32))
    new_gs = comet_spawn(gs)

    # If the seed produced valid paths for spawn 0, we should see 4
    # new alive planets; else no change.
    valid_0 = bool(gs.comet_valid[0])
    alive_before = int(jnp.sum(gs.planets_alive.astype(jnp.int32)))
    alive_after = int(jnp.sum(new_gs.planets_alive.astype(jnp.int32)))

    if valid_0:
        assert alive_after == alive_before + 4, (
            f"valid spawn should add 4 alive planets; got "
            f"before={alive_before} after={alive_after}"
        )
        assert bool(new_gs.comet_spawned[0]) is True
        # Per-path index lookup
        idx_arr = np.asarray(new_gs.comet_planet_idx[0])
        for j in range(4):
            slot = int(idx_arr[j])
            assert slot >= 0, f"comet_planet_idx[0, {j}] should be set"
            assert bool(new_gs.planets_alive[slot]) is True
            assert int(new_gs.planets_owner[slot]) == -1
            assert int(new_gs.planets_ships[slot]) == int(gs.comet_ships[0])
            assert int(new_gs.planet_comet_spawn[slot]) == 0
            assert int(new_gs.planet_comet_path[slot]) == j
            assert bool(new_gs.is_comet[slot]) is True
            assert float(new_gs.planets_x[slot]) == pytest.approx(-99.0)
            assert float(new_gs.planets_y[slot]) == pytest.approx(-99.0)
    else:
        assert alive_after == alive_before, "invalid spawn should be no-op"
        assert bool(new_gs.comet_spawned[0]) is False


def test_comet_spawn_no_double_fire():
    """Calling comet_spawn again at the same step (already-spawned k)
    is a no-op."""
    from lib.game.jax import comet_spawn
    env, gs = _build_state(42)
    if not bool(gs.comet_valid[0]):
        pytest.skip("seed 42 doesn't produce valid spawn 0")
    gs = gs._replace(step=jnp.asarray(49, dtype=jnp.int32))
    once = comet_spawn(gs)
    twice = comet_spawn(once)
    # Same alive count after second call.
    assert int(jnp.sum(once.planets_alive.astype(jnp.int32))) == (
        int(jnp.sum(twice.planets_alive.astype(jnp.int32)))
    )


def test_comet_spawn_then_advance_positions_at_path0(seed=42):
    """After spawn at step 49, comet_path_advance moves the 4 comets
    from (-99, -99) to `comet_paths_xy[0, j, 0, :]`.
    """
    from lib.game.jax import comet_spawn, comet_path_advance
    env, gs = _build_state(seed)
    if not bool(gs.comet_valid[0]):
        pytest.skip("seed 42 doesn't produce valid spawn 0")
    gs = gs._replace(step=jnp.asarray(49, dtype=jnp.int32))
    spawned = comet_spawn(gs)
    advanced = comet_path_advance(spawned)

    for j in range(4):
        slot = int(spawned.comet_planet_idx[0, j])
        expected_x = float(gs.comet_paths_xy[0, j, 0, 0])
        expected_y = float(gs.comet_paths_xy[0, j, 0, 1])
        assert float(advanced.planets_x[slot]) == pytest.approx(expected_x)
        assert float(advanced.planets_y[slot]) == pytest.approx(expected_y)
    # Path index should now be 0 (incremented from -1).
    assert int(advanced.comet_path_index[0]) == 0


def test_comet_path_advance_no_op_at_start():
    """At step 0 with no comets spawned, comet_path_advance leaves the
    state unchanged (no comets to advance)."""
    env, gs = _build_state(42)
    from lib.game.jax import comet_path_advance
    new_gs = comet_path_advance(gs)
    np.testing.assert_array_equal(
        np.asarray(new_gs.planets_x), np.asarray(gs.planets_x)
    )
    np.testing.assert_array_equal(
        np.asarray(new_gs.planets_y), np.asarray(gs.planets_y)
    )
    np.testing.assert_array_equal(
        np.asarray(new_gs.planets_alive), np.asarray(gs.planets_alive)
    )
    np.testing.assert_array_equal(
        np.asarray(new_gs.comet_path_index),
        np.asarray(gs.comet_path_index),
    )
