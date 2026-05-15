"""Step 5 — JAX engine wrapper: random-action parity vs scalar.

Confirms that wrapping `jax_step` in `lib.foundation.jax_engine`'s
entry points (`step`, `step_from_action_tensor`, `step_batched`,
`rollout_python`) doesn't introduce any drift. Reuses the
infrastructure in `test_jax_full_step_parity.py` (paired-state
construction, planet / fleet comparison) to keep the test focused
on the wrapper, not the underlying physics.

Tolerances (matching the existing JAX parity bar):
- combat / ownership / ship counts / fleet list: exact (int / bool)
- planet & fleet positions: float32 tol ≤ 1e-3
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter as scalar_interpreter

from lib.foundation.actions import ActionSpec, specs_to_tensor
from lib.foundation.jax_engine import (
    clone_state,
    rollout_python,
    step,
    step_batched,
    step_from_action_tensor,
)
from lib.game.jax.conversions import actions_to_jax, scalar_to_jax
from tests.test_jax_full_step_parity import (
    _compare_fleets,
    _compare_planets,
    _make_paired_states,
    _rand_actions_for_scalar,
)


def _bookkeep_scalar_step(env, num_agents: int):
    """Mirror of the bookkeeping in `test_jax_full_step_parity` —
    bump the scalar obs step counter that env.step doesn't touch when
    we call the interpreter directly."""
    obs0 = env.state[0].observation
    new_step = int(obs0.get("step", 0)) + 1
    obs0.step = new_step
    for i in range(1, num_agents):
        env.state[i].observation.step = new_step


# ---------------------------------------------------------------------------
# step() — single state, single turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_step_matches_scalar_30_random(seed):
    """`lib.foundation.jax_engine.step` matches the scalar interpreter
    over 30 random-action turns."""
    num_agents = 2
    env, gs = _make_paired_states(seed, num_agents)

    diff = _compare_planets(env.state[0].observation, gs, -1, seed)
    assert not diff, f"init: {diff}"

    rng = random.Random(seed * 7919 + 11)
    for tick in range(30):
        if env.state[0].status != "ACTIVE":
            break
        actions = _rand_actions_for_scalar(env.state, rng, num_agents)

        # Scalar step.
        for i, a in enumerate(actions):
            env.state[i].action = a
        scalar_interpreter(env.state, env)
        _bookkeep_scalar_step(env, num_agents)

        # JAX step via the foundation wrapper.
        pids, angles, ships = actions_to_jax(actions, num_agents)
        gs = step(gs, pids, angles, ships)

        diff_p = _compare_planets(env.state[0].observation, gs, tick, seed)
        assert not diff_p, diff_p
        diff_f = _compare_fleets(env.state[0].observation, gs, tick, seed)
        assert not diff_f, diff_f


# ---------------------------------------------------------------------------
# step_from_action_tensor() — convenience wrapper
# ---------------------------------------------------------------------------


def test_step_from_action_tensor_single_launch():
    """Round-trip an ActionTensor through the wrapper. With a single
    launch in (candidate=0, turn=0), the result equals a direct
    step() with the same action."""
    num_agents = 2
    env, gs0 = _make_paired_states(seed=42, num_agents=num_agents)

    # Pick an owned planet for seat 0 with ships > 0.
    owned = [p for p in env.state[0].observation.planets if p[1] == 0 and p[5] > 0]
    if not owned:
        pytest.skip("no owned planets at init; pick another seed")
    src = owned[0]
    spec = ActionSpec(
        from_planet_id=int(src[0]),
        dir_angle=0.5,
        ships=max(1, int(src[5] // 2)),
        launch_turn=0,
        agent_id=0,
    )

    # Path A: via ActionTensor.
    tensor = specs_to_tensor([[spec]], horizon=1, num_agents=num_agents)
    gs_a = step_from_action_tensor(gs0, tensor, candidate=0, turn=0)

    # Path B: direct step() with action arrays.
    pids = jnp.asarray(tensor.pids[0, 0])
    angles = jnp.asarray(tensor.angles[0, 0])
    ships = jnp.asarray(tensor.ships[0, 0])
    gs_b = step(gs0, pids, angles, ships)

    # The two paths must produce identical results.
    assert np.array_equal(np.asarray(gs_a.planets_owner), np.asarray(gs_b.planets_owner))
    assert np.array_equal(np.asarray(gs_a.planets_ships), np.asarray(gs_b.planets_ships))
    assert np.array_equal(np.asarray(gs_a.fleets_alive), np.asarray(gs_b.fleets_alive))
    assert np.array_equal(np.asarray(gs_a.fleets_ships), np.asarray(gs_b.fleets_ships))


# ---------------------------------------------------------------------------
# step_batched() — vmap over candidates
# ---------------------------------------------------------------------------


def test_step_batched_matches_scalar_per_candidate():
    """`step_batched` over a batch of identical states with different
    actions must produce per-candidate results equal to running
    `step` independently on each."""
    num_agents = 2
    env, gs0 = _make_paired_states(seed=7, num_agents=num_agents)

    rng = random.Random(7919)
    # Build 3 different action sets from the same gs0.
    action_sets = [
        _rand_actions_for_scalar(env.state, rng, num_agents) for _ in range(3)
    ]
    action_jax = [actions_to_jax(a, num_agents) for a in action_sets]
    pids_b = jnp.stack([a[0] for a in action_jax], axis=0)   # (3, A, L)
    angles_b = jnp.stack([a[1] for a in action_jax], axis=0)
    ships_b = jnp.stack([a[2] for a in action_jax], axis=0)

    # Replicate gs0 along a leading axis of 3 — we use jax.tree.map.
    import jax
    gs_batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), gs0)

    gs_after = step_batched(gs_batch, pids_b, angles_b, ships_b)

    # Compare each batch element against an independent step().
    for i in range(3):
        gs_single = step(gs0, action_jax[i][0], action_jax[i][1], action_jax[i][2])
        gs_batch_i = jax.tree.map(lambda x, idx=i: x[idx], gs_after)

        assert np.array_equal(
            np.asarray(gs_batch_i.planets_ships),
            np.asarray(gs_single.planets_ships),
        ), f"batch {i}: planets_ships mismatch"
        assert np.array_equal(
            np.asarray(gs_batch_i.planets_owner),
            np.asarray(gs_single.planets_owner),
        ), f"batch {i}: planets_owner mismatch"
        assert np.array_equal(
            np.asarray(gs_batch_i.fleets_alive),
            np.asarray(gs_single.fleets_alive),
        ), f"batch {i}: fleets_alive mismatch"


# ---------------------------------------------------------------------------
# rollout_python() — Python loop over turns
# ---------------------------------------------------------------------------


def test_rollout_python_matches_loop_of_step():
    """`rollout_python` for H turns must equal calling step() H times
    manually."""
    num_agents = 2
    env, gs0 = _make_paired_states(seed=42, num_agents=num_agents)
    rng_a = random.Random(7919)
    rng_b = random.Random(7919)  # SAME seed → same action sequence

    H = 12

    # Path A: manual loop.
    gs_a = gs0
    for _ in range(H):
        actions = _rand_actions_for_scalar(env.state, rng_a, num_agents)
        pids, angles, ships = actions_to_jax(actions, num_agents)
        gs_a = step(gs_a, pids, angles, ships)

    # Path B: rollout_python with an action_fn that draws from the
    # same RNG. Reset env state for fair RNG reuse.
    env_b, gs_b0 = _make_paired_states(seed=42, num_agents=num_agents)

    def action_fn(state, t, memory):
        actions = _rand_actions_for_scalar(env_b.state, rng_b, num_agents)
        pids, angles, ships = actions_to_jax(actions, num_agents)
        return pids, angles, ships, memory

    gs_b_final, _ = rollout_python(gs_b0, action_fn, horizon=H, memory=None)

    # Paths must agree — same RNG, same actions, same physics.
    assert np.array_equal(
        np.asarray(gs_a.planets_owner),
        np.asarray(gs_b_final.planets_owner),
    )
    assert np.array_equal(
        np.asarray(gs_a.planets_ships),
        np.asarray(gs_b_final.planets_ships),
    )


def test_rollout_python_skips_done_state():
    """If `state.done` is True at the top of the loop, no action_fn
    calls fire. Contract: rollout_python checks done BEFORE running
    the turn — otherwise `jax_step.terminate` would re-evaluate the
    done condition and could overwrite a manually-set flag."""
    num_agents = 2
    env, gs0 = _make_paired_states(seed=42, num_agents=num_agents)
    gs0 = gs0._replace(done=jnp.array(True))

    call_count = {"n": 0}

    def action_fn(state, t, memory):
        call_count["n"] += 1
        pids, angles, ships = actions_to_jax([[]] * num_agents, num_agents)
        return pids, angles, ships, memory

    rollout_python(gs0, action_fn, horizon=10, memory=None)
    assert call_count["n"] == 0, (
        f"rollout_python should make 0 calls when started on a done state, "
        f"got {call_count['n']}"
    )


def test_rollout_python_runs_full_horizon_on_active_state():
    """Sanity: on an ACTIVE (non-done) state, rollout_python runs
    exactly `horizon` turns and calls `action_fn` once per turn."""
    num_agents = 2
    env, gs0 = _make_paired_states(seed=42, num_agents=num_agents)

    call_count = {"n": 0}

    def action_fn(state, t, memory):
        call_count["n"] += 1
        pids, angles, ships = actions_to_jax([[]] * num_agents, num_agents)
        return pids, angles, ships, memory

    rollout_python(gs0, action_fn, horizon=5, memory=None)
    assert call_count["n"] == 5


# ---------------------------------------------------------------------------
# clone_state() — documented identity
# ---------------------------------------------------------------------------


def test_clone_state_is_identity():
    """`clone_state` returns the input unchanged (Pytrees immutable).
    Documented contract; protects against future API drift."""
    _, gs0 = _make_paired_states(seed=42, num_agents=2)
    gs1 = clone_state(gs0)
    assert gs1 is gs0, "clone_state must be identity for immutable Pytrees"
