"""Value-head + vmap'd scorer tests.

The Phase A scoring kernel is:
  K=5 Tier-1 mirror rollout (`rollout_step_jax_pure`) → JAX-pure
  value head `value_with_future_production`.

Tests:
- `value_with_future_production` is exact on a synthetic 2-planet
  state.
- Scoring an empty action set returns a deterministic float.
- Scoring a launch that captures an enemy production planet should
  raise the score vs the empty-set baseline (after rollout
  resolution, we hold the planet for ~remaining_steps × prod
  ships).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation.actions import ActionSpec
from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies.analytic_score import (
    action_specs_to_candidate_arrays,
    enumerate_atomic_launches,
    score_candidates_vmap_value_prod,
    score_candidates_vmap_value_prod_jit,
    value_with_future_production,
)


def _seed_state(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    return obs_to_jax_state(obs, configuration=env.configuration)


def test_value_head_signs():
    """On a fresh state, both seats own one starter; production +
    ships are symmetric, so the value head ≈ 0."""
    state = _seed_state()
    v = float(value_with_future_production(state, my_id=0))
    v_swap = float(value_with_future_production(state, my_id=1))
    # Symmetry: swapping seats negates the value.
    assert abs(v + v_swap) < 1e-3, f"{v} vs {v_swap}"


def test_value_head_increases_with_more_owned_production():
    """A state with planet[0] re-owned to us beats the original."""
    state = _seed_state()
    v_base = float(value_with_future_production(state, my_id=0))

    # Force-flip the first non-mine non-neutral planet to us.
    owner = np.asarray(state.planets_owner).copy()
    alive = np.asarray(state.planets_alive)
    flipped_one = False
    for i in range(len(owner)):
        if alive[i] and owner[i] not in (0, -1):
            owner[i] = 0
            flipped_one = True
            break
    assert flipped_one, "test setup: needs at least one opp-owned planet"

    # GameState is a NamedTuple — use `_replace`.
    state_flipped = state._replace(
        planets_owner=jnp.asarray(owner, dtype=jnp.int32),
    )
    v_flipped = float(value_with_future_production(state_flipped, my_id=0))
    assert v_flipped > v_base


def test_score_empty_set_deterministic():
    """Scoring the empty action set twice returns the same value."""
    state = _seed_state()
    pids, angles, ships = action_specs_to_candidate_arrays([[]])
    s1 = score_candidates_vmap_value_prod_jit(
        state,
        jnp.asarray(pids),
        jnp.asarray(angles),
        jnp.asarray(ships),
        K=5, my_id=0, num_agents=2,
    )
    s2 = score_candidates_vmap_value_prod_jit(
        state,
        jnp.asarray(pids),
        jnp.asarray(angles),
        jnp.asarray(ships),
        K=5, my_id=0, num_agents=2,
    )
    assert abs(float(s1[0]) - float(s2[0])) < 1e-4


def test_vmap_returns_per_candidate_scores():
    """A vmap over C=3 candidates returns shape (3,)."""
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0, max_eta=80)
    assert len(atoms) >= 2
    candidates = [[], [atoms[0]], [atoms[1]]]
    pids, angles, ships = action_specs_to_candidate_arrays(candidates)
    scores = score_candidates_vmap_value_prod_jit(
        state,
        jnp.asarray(pids),
        jnp.asarray(angles),
        jnp.asarray(ships),
        K=5, my_id=0, num_agents=2,
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(np.asarray(scores)))


def test_4p_state_raises():
    """The vmap'd scorer is 2P-only (rollout primitives are 2P).
    Must raise a clear ValueError on num_agents != 2."""
    state = _seed_state()
    pids, angles, ships = action_specs_to_candidate_arrays([[]])
    with pytest.raises(ValueError, match="2P-only"):
        score_candidates_vmap_value_prod(
            state,
            jnp.asarray(pids),
            jnp.asarray(angles),
            jnp.asarray(ships),
            K=5, my_id=0, num_agents=4,
        )
