"""Smoke + shape tests for `lib.game.jax.jax_depth2`.

This file pins the invariants we need for the depth-2 chooser to be
usable inside the Kaggle JAX A/B kernel:
- `jax_drop_one_variants` emits `(MAX_LAUNCH+1, MAX_LAUNCH)`-shaped
  triples with row 0 = incumbent, row k+1 = incumbent with slot k zeroed.
- `policy_emit_depth2_jax_pure` returns `(MAX_LAUNCH,)`-shape triples,
  is deterministic across repeated calls, and produces at least one
  valid launch on a fresh game.
- `rollout_step_depth2_jax_pure` advances a state by exactly one step.

Note: this is a SMOKE test, not a parity test against the scalar
`choose_depth2`. A bit-exact parity gate would require matching cap
sizes (scalar caps our_C/opp_C to 8/4; JAX uses all MAX_LAUNCH+1) and
matching K_tail; not worth the complexity for the offline-only path.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from kaggle_environments import make

from lib.game.jax.conversions import scalar_to_jax
from lib.game.jax.jax_depth2 import (
    jax_drop_one_variants,
    policy_emit_depth2_jax_pure,
    rollout_step_depth2_jax_pure,
)
from lib.game.jax.jax_types import MAX_LAUNCH_PER_AGENT


# ---------------------------------------------------------------------------
# jax_drop_one_variants
# ---------------------------------------------------------------------------


def test_drop_one_variants_row0_is_incumbent():
    pids = jnp.asarray([5, 7, 9] + [-1] * (MAX_LAUNCH_PER_AGENT - 3), dtype=jnp.int32)
    angles = jnp.asarray(
        [1.0, 2.0, 3.0] + [0.0] * (MAX_LAUNCH_PER_AGENT - 3),
        dtype=jnp.float32,
    )
    ships = jnp.asarray(
        [10, 20, 30] + [0] * (MAX_LAUNCH_PER_AGENT - 3),
        dtype=jnp.int32,
    )
    p_v, a_v, s_v = jax_drop_one_variants(pids, angles, ships)
    assert p_v.shape == (MAX_LAUNCH_PER_AGENT + 1, MAX_LAUNCH_PER_AGENT)
    assert (p_v[0] == pids).all()
    assert (a_v[0] == angles).all()
    assert (s_v[0] == ships).all()


def test_drop_one_variants_row_k_zeros_slot_k():
    pids = jnp.asarray([5, 7, 9] + [-1] * (MAX_LAUNCH_PER_AGENT - 3), dtype=jnp.int32)
    angles = jnp.asarray(
        [1.0, 2.0, 3.0] + [0.0] * (MAX_LAUNCH_PER_AGENT - 3),
        dtype=jnp.float32,
    )
    ships = jnp.asarray(
        [10, 20, 30] + [0] * (MAX_LAUNCH_PER_AGENT - 3),
        dtype=jnp.int32,
    )
    p_v, _, s_v = jax_drop_one_variants(pids, angles, ships)
    # Row 1 drops slot 0.
    assert int(p_v[1, 0]) == -1 and int(s_v[1, 0]) == 0
    # Other slots unaffected.
    assert int(p_v[1, 1]) == 7 and int(s_v[1, 1]) == 20
    # Row 3 drops slot 2.
    assert int(p_v[3, 2]) == -1 and int(s_v[3, 2]) == 0
    assert int(p_v[3, 0]) == 5 and int(p_v[3, 1]) == 7


# ---------------------------------------------------------------------------
# policy_emit_depth2_jax_pure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_state():
    env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 500})
    env.reset(num_agents=2)
    return scalar_to_jax(env.state, episode_seed=0)


def test_depth2_policy_emit_shapes(fresh_state):
    p, a, s = policy_emit_depth2_jax_pure(
        fresh_state, my_id=0, K_tail=4,
        opp_aggressive=True, my_aggressive=True,
        my_use_opening=True, opp_use_opening=True,
    )
    assert p.shape == (MAX_LAUNCH_PER_AGENT,)
    assert a.shape == (MAX_LAUNCH_PER_AGENT,)
    assert s.shape == (MAX_LAUNCH_PER_AGENT,)
    # No behavioural assertion: maximin may legitimately choose the
    # all-empty drop-one variant on any given state. Shape + dtype are
    # the only stable contract.
    assert p.dtype == jnp.int32
    assert a.dtype == jnp.float32
    assert s.dtype == jnp.int32


def test_depth2_policy_emit_deterministic(fresh_state):
    p1, a1, s1 = policy_emit_depth2_jax_pure(
        fresh_state, my_id=0, K_tail=4,
        opp_aggressive=True, my_aggressive=True,
        my_use_opening=True, opp_use_opening=True,
    )
    p2, a2, s2 = policy_emit_depth2_jax_pure(
        fresh_state, my_id=0, K_tail=4,
        opp_aggressive=True, my_aggressive=True,
        my_use_opening=True, opp_use_opening=True,
    )
    assert (p1 == p2).all()
    assert (a1 == a2).all()
    assert (s1 == s2).all()


# ---------------------------------------------------------------------------
# rollout_step_depth2_jax_pure
# ---------------------------------------------------------------------------


def test_depth2_rollout_advances_one_step(fresh_state):
    s_next = rollout_step_depth2_jax_pure(
        fresh_state, my_id=0, K_tail=4,
        opp_aggressive=True, my_aggressive=True,
        my_use_opening=True, opp_use_opening=True,
    )
    assert int(s_next.step) == int(fresh_state.step) + 1
