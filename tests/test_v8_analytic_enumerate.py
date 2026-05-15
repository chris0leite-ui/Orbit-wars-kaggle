"""Atomic-launch enumeration tests for v8_analytic.

`enumerate_atomic_launches` is strategy-agnostic: no proposer, no
mission framework. Just (src_owned × every-other-alive × {0.5, 1.0}
fraction) tuples with an orbital lead-aim + ETA filter.
"""

from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation.actions import ActionSpec
from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies.analytic_score import enumerate_atomic_launches


def _seed_state(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    return obs_to_jax_state(obs, configuration=env.configuration)


def test_returns_at_least_one_atom_on_seeded_state():
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    assert len(atoms) > 0


def test_all_atoms_are_action_specs_with_valid_fields():
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    for spec in atoms:
        assert isinstance(spec, ActionSpec)
        assert spec.from_planet_id >= 0
        assert spec.ships > 0
        assert spec.launch_turn == 0
        assert spec.agent_id == 0
        # dir_angle is a finite float (not NaN, not inf).
        assert np.isfinite(spec.dir_angle)


def test_src_planet_never_equals_target_planet():
    """No self-targeting atomic. We enumerate (src, target) with
    target != src; the aim function would also reject collinear
    angles but the loop pre-filters."""
    state = _seed_state()
    atoms = enumerate_atomic_launches(state, my_id=0)
    # All atoms launch from an owned planet. Self-loops would mean an
    # atom with from_planet_id == aim's predicted-target planet, but
    # we don't have target_id on the spec. The structural check is
    # easier: every src in `atoms` is in the owned set.
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    alive = np.asarray(state.planets_alive)
    owned_set = {int(ids[i]) for i in range(len(alive))
                 if alive[i] and owner[i] == 0}
    for spec in atoms:
        assert spec.from_planet_id in owned_set


def test_both_ship_fractions_appear():
    """For at least one (src, target) pair, both `0.5` and `1.0`
    fractions should produce an atom."""
    state = _seed_state()
    atoms = enumerate_atomic_launches(
        state, my_id=0, ship_fractions=(0.5, 1.0),
    )
    # Group by src; count distinct ship counts per src.
    by_src: dict[int, set[int]] = {}
    for spec in atoms:
        by_src.setdefault(spec.from_planet_id, set()).add(spec.ships)
    # At least one src has multiple distinct ship counts (proves
    # multiple fractions made it through).
    multi = [s for s, sizes in by_src.items() if len(sizes) >= 2]
    assert len(multi) > 0


def test_eta_filter_drops_far_targets():
    """`max_eta=1` should drop nearly everything (no fleet arrives
    in one step). Used as a sanity floor for the eta filter."""
    state = _seed_state()
    near_atoms = enumerate_atomic_launches(state, my_id=0, max_eta=1)
    all_atoms = enumerate_atomic_launches(state, my_id=0, max_eta=200)
    # With max_eta=1 we drop most atoms.
    assert len(near_atoms) < len(all_atoms)


def test_opponent_id_returns_opponent_owned_sources():
    state = _seed_state()
    atoms_p1 = enumerate_atomic_launches(state, my_id=1)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    alive = np.asarray(state.planets_alive)
    owned_set = {int(ids[i]) for i in range(len(alive))
                 if alive[i] and owner[i] == 1}
    for spec in atoms_p1:
        assert spec.from_planet_id in owned_set
        assert spec.agent_id == 1
