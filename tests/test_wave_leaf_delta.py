"""Unit tests for the V3 wave's value-head leaf-Δ helper.

`_wave_leaf_delta` runs two K-step rollouts (with vs without the wave's
extras) sharing the same opp action substrate, scores both leaves with
`agents.baseline.value.favor`, and returns Δfavor. Caller fires only
if Δ > epsilon AND wall stays under the budget.
"""

from __future__ import annotations

import math

import pytest
from kaggle_environments import make

import agents.baseline.main as bm
from lib.fast_sim import from_obs as fs_from_obs


def _snap_from_seed(seed: int = 42, num_seats: int = 2):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_seats)
    obs = env.steps[0][0].observation
    return obs, fs_from_obs(obs, num_seats=num_seats)


def test_leaf_delta_noop_is_zero():
    """Same moves on both sides → Δ must be 0 (CRN identity)."""
    _obs, snap = _snap_from_seed(seed=42)
    delta = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=None,
        moves_without=[],
        moves_with=[],
    )
    assert delta == 0.0


def test_leaf_delta_returns_finite_number():
    """Differing moves → Δ must be a finite float (any sign)."""
    _obs, snap = _snap_from_seed(seed=42)
    # Pick the first owned planet and aim east; ships value tuned to be
    # within the planet's available count. We're testing plumbing not
    # mechanics, so the action doesn't need to be strategically sound.
    obs0 = snap.state[0].observation
    my_planets = [p for p in obs0["planets"] if int(p[1]) == 0]
    assert my_planets, "seed 42 must have a player-0 planet at step 0"
    src_id = int(my_planets[0][0])
    ships = max(1, int(my_planets[0][5]) // 2)
    move = [src_id, 0.0, ships]
    delta = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=None,
        moves_without=[],
        moves_with=[move],
    )
    assert math.isfinite(delta)


def test_leaf_delta_crn_with_opp_traj():
    """When opp_traj is provided, the same script is replayed in both
    rollouts. Δ should still be a finite float. Verifies the opp-traj
    code path doesn't crash and is consistent on identical inputs."""
    _obs, snap = _snap_from_seed(seed=7)
    opp_traj = [[[], []] for _ in range(8)]  # all-idle opp script
    delta_a = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=opp_traj,
        moves_without=[],
        moves_with=[],
    )
    delta_b = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=opp_traj,
        moves_without=[],
        moves_with=[],
    )
    # Same inputs (and explicit opp script) → identical outputs.
    assert delta_a == delta_b == 0.0


def test_leaf_delta_consistent_on_repeated_call():
    """Determinism check: same snap + same inputs → identical Δ across
    two calls. Critical for the V3 gate to be reproducible."""
    _obs, snap = _snap_from_seed(seed=3)
    obs0 = snap.state[0].observation
    my_planets = [p for p in obs0["planets"] if int(p[1]) == 0]
    src_id = int(my_planets[0][0])
    ships = max(1, int(my_planets[0][5]) // 3)
    move = [src_id, 1.0, ships]
    d1 = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=None,
        moves_without=[],
        moves_with=[move],
    )
    d2 = bm._wave_leaf_delta(
        snap, me=0, num_seats=2, gamma=0.99,
        horizon=5, opp_traj=None,
        moves_without=[],
        moves_with=[move],
    )
    assert d1 == d2
