"""Beam-search mechanics tests.

Confirms the structural behaviour of `beam_search`:
- Empty atomic list returns the empty action set (no launches).
- Returned action set never exceeds `depth`.
- The one-launch-per-source filter is enforced.
- A beam-1-depth-1 search reduces to a single launch (greedy-1).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies.analytic_score import enumerate_atomic_launches
from lib.foundation.strategies.beam_search import beam_search, _filter_compatible


def _seed_state(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    return obs_to_jax_state(obs, configuration=env.configuration)


def test_empty_atomics_returns_empty():
    state = _seed_state()
    result = beam_search(
        state, atomic_launches=[], my_id=0,
        width=4, depth=4, K=5, num_agents=2, budget_ms=200.0,
    )
    assert result == []


def test_returned_set_size_bounded_by_depth():
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    result = beam_search(
        state, atoms, my_id=0,
        width=2, depth=3, K=5, num_agents=2, budget_ms=2000.0,
    )
    assert len(result) <= 3


def test_one_launch_per_source_constraint_holds():
    """Returned action set has at most one launch per source planet."""
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    result = beam_search(
        state, atoms, my_id=0,
        width=4, depth=4, K=5, num_agents=2, budget_ms=2000.0,
    )
    sources = [spec.from_planet_id for spec in result]
    assert len(sources) == len(set(sources)), (
        f"Source planet used twice in beam result: {sources}"
    )


def test_beam_1_depth_1_reduces_to_best_single_launch():
    """Width=1, depth=1 should pick the single best-scoring atom (or
    the empty set if no atom beats baseline)."""
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    assert len(atoms) > 0
    result = beam_search(
        state, atoms, my_id=0,
        width=1, depth=1, K=5, num_agents=2, budget_ms=2000.0,
    )
    assert len(result) <= 1


def test_filter_compatible_drops_used_sources():
    """`_filter_compatible` strips atoms whose source is already
    represented in `current_set`."""
    from lib.foundation.actions import ActionSpec

    pool = [
        ActionSpec(from_planet_id=0, dir_angle=0.0, ships=10,
                   launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=0, dir_angle=1.0, ships=20,
                   launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=1, dir_angle=0.0, ships=10,
                   launch_turn=0, agent_id=0),
    ]
    current = [ActionSpec(from_planet_id=0, dir_angle=0.5, ships=5,
                          launch_turn=0, agent_id=0)]
    filtered = _filter_compatible(current, pool)
    assert {f.from_planet_id for f in filtered} == {1}
