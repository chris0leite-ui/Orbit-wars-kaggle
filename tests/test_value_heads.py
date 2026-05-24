"""Sanity tests for the v9 value heads."""

from __future__ import annotations

import random

import pytest

from kaggle_environments import make

from lib import fast_sim
from lib.value_heads import (
    INFLIGHT_EXTRA_HORIZON,
    INFLIGHT_WEIGHT,
    delta_us_minus_them_obs,
    inflight_hhi_bonus,
    inflight_value,
    stockpile_pressure_penalty,
)


def _warmed_snap(seed: int = 42, warmup: int = 15):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed)
    for _ in range(warmup):
        obs = env.state[0].observation
        a = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
             for p in obs["planets"] if p[1] == 0 and p[5] > 5 and rng.random() < 0.3]
        b = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
             for p in obs["planets"] if p[1] == 1 and p[5] > 5 and rng.random() < 0.3]
        env.step([a, b])
    snap = fast_sim.from_obs(
        env.state[0].observation, env.configuration,
        episode_seed=env.info["seed"], num_seats=2,
    )
    return snap


# ---------------------------------------------------------------------------
# delta_us_minus_them_obs
# ---------------------------------------------------------------------------


def test_delta_us_minus_them_basic():
    """Total ships per side; us minus them."""
    snap = _warmed_snap()
    d0 = delta_us_minus_them_obs(snap.state[0].observation, my_id=0)
    d1 = delta_us_minus_them_obs(snap.state[1].observation, my_id=1)
    # Antisymmetric: from seat 0 it's us-them; from seat 1 it's them-us.
    # The MAGNITUDES match (one observation, two POVs).
    assert d0 == -d1


def test_delta_us_minus_them_empty_obs():
    """Empty world → 0.0."""
    obs = {"planets": [], "fleets": []}
    assert delta_us_minus_them_obs(obs, my_id=0) == 0.0


# ---------------------------------------------------------------------------
# inflight_value
# ---------------------------------------------------------------------------


def test_inflight_value_reduces_to_base_when_no_inflight_captures():
    """If no in-flight fleets will flip ownership, inflight credit = 0
    and the head returns the base ship-delta."""
    obs = {"planets": [], "fleets": [], "player": 0, "angular_velocity": 0.0,
           "initial_planets": [], "comet_planet_ids": [], "comets": [],
           "step": 0, "next_fleet_id": 0}
    # Empty world → base 0, bonus 0.
    assert inflight_value(obs, my_id=0) == 0.0


def test_inflight_value_credits_predicted_capture():
    """A fleet en route to an enemy planet that will flip to us
    within extra_horizon adds production × weight to the score."""
    snap = _warmed_snap(seed=7, warmup=20)
    obs = snap.state[0].observation

    base = delta_us_minus_them_obs(obs, my_id=0)
    composite = inflight_value(obs, my_id=0)

    # The composite is base + bonus ≥ 0 (bonus is non-negative).
    assert composite >= base


def test_inflight_value_weight_calibration():
    """Setting weight=0 reduces inflight_value to base ship-delta."""
    snap = _warmed_snap()
    obs = snap.state[0].observation
    base = delta_us_minus_them_obs(obs, my_id=0)
    zero_bonus = inflight_value(obs, my_id=0, weight=0.0)
    assert base == zero_bonus


def test_inflight_value_extra_horizon_affects_score():
    """Larger extra_horizon should see more (or equal) captures."""
    snap = _warmed_snap(seed=42, warmup=25)
    obs = snap.state[0].observation
    short = inflight_value(obs, my_id=0, extra_horizon=10)
    long_ = inflight_value(obs, my_id=0, extra_horizon=50)
    # More predicted captures with longer horizon → score >= short.
    # Equal allowed for boards where the arrival ledger is empty.
    assert long_ >= short - 1e-9


def test_inflight_value_works_as_value_fn_in_score_candidate():
    """The signature `value_fn(obs, my_id)` is what score_candidate
    expects. inflight_value must accept that."""
    from lib.v7_search import score_candidate
    snap = _warmed_snap()
    # Score the incumbent's empty action.
    score = score_candidate(snap, [], my_id=0, K=5, opp_tier=1,
                             value_fn=inflight_value)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Wave-incentive terms — pure-math tests (baseline_wave 2026-05-24)
# ---------------------------------------------------------------------------


def _obs_with(fleets=None, planets=None):
    """Build a minimal obs dict for the unit-math tests.

    Fleet tuple shape (kaggle): (id, owner, x, y, angle, from_planet, ships).
    Planet tuple shape (kaggle): (id, owner, x, y, radius, ships, production).
    """
    return {
        "fleets": list(fleets or []),
        "planets": list(planets or []),
    }


def test_hhi_zero_no_fleets():
    obs = _obs_with(fleets=[])
    assert inflight_hhi_bonus(obs, my_id=0, delta=0.1) == 0.0


def test_hhi_zero_below_noise_floor():
    """Total inflight 1 ship → below MIN_FLEET_SIZE → 0."""
    obs = _obs_with(fleets=[(0, 0, 0, 0, 0, 0, 1)])
    assert inflight_hhi_bonus(obs, my_id=0, delta=0.1) == 0.0


def test_hhi_single_fleet_max():
    """One friendly fleet of 100 ships → HHI = 1.0 → bonus = δ·1·100."""
    obs = _obs_with(fleets=[(0, 0, 0, 0, 0, 0, 100)])
    got = inflight_hhi_bonus(obs, my_id=0, delta=0.1)
    assert abs(got - 0.1 * 1.0 * 100.0) < 1e-9


def test_hhi_equal_fleets_one_over_n():
    """N equal-size friendly fleets → HHI = 1/n, scaled by Σs."""
    n = 4
    size = 25
    fleets = [(i, 0, 0, 0, 0, 0, size) for i in range(n)]
    obs = _obs_with(fleets=fleets)
    got = inflight_hhi_bonus(obs, my_id=0, delta=0.1)
    expected = 0.1 * (1.0 / n) * (n * size)  # = 0.1 * 25 = 2.5
    assert abs(got - expected) < 1e-9


def test_hhi_ignores_enemy_fleets():
    """Friendly filter: enemy ships must not enter HHI calculation."""
    obs = _obs_with(fleets=[
        (0, 0, 0, 0, 0, 0, 50),   # ours
        (1, 1, 0, 0, 0, 0, 999),  # enemy — must be ignored
    ])
    got = inflight_hhi_bonus(obs, my_id=0, delta=0.1)
    assert abs(got - 0.1 * 1.0 * 50.0) < 1e-9


def test_hhi_concentrated_beats_spread_at_equal_total():
    """Pareto check: 100 ships in 1 fleet should score higher than
    100 ships in 4 fleets of 25."""
    one = _obs_with(fleets=[(0, 0, 0, 0, 0, 0, 100)])
    spread = _obs_with(fleets=[(i, 0, 0, 0, 0, 0, 25) for i in range(4)])
    v_one = inflight_hhi_bonus(one, my_id=0, delta=0.1)
    v_spread = inflight_hhi_bonus(spread, my_id=0, delta=0.1)
    assert v_one > v_spread, (v_one, v_spread)


def test_stockpile_penalty_zero_at_target():
    """Excess ≤ 0 → penalty 0."""
    obs = _obs_with(planets=[(0, 0, 0, 0, 1.0, 50, 2)])
    assert stockpile_pressure_penalty(obs, my_id=0, eps=0.005, target=50.0) == 0.0


def test_stockpile_penalty_ignores_enemy_planets():
    obs = _obs_with(planets=[(0, 1, 0, 0, 1.0, 250, 2)])  # enemy planet
    assert stockpile_pressure_penalty(obs, my_id=0, eps=0.005, target=50.0) == 0.0


def test_stockpile_penalty_quadratic():
    """penalty(excess=50) / penalty(excess=25) ≈ 4."""
    obs_50 = _obs_with(planets=[(0, 0, 0, 0, 1.0, 100, 2)])  # excess=50
    obs_25 = _obs_with(planets=[(0, 0, 0, 0, 1.0, 75, 2)])   # excess=25
    p50 = stockpile_pressure_penalty(obs_50, my_id=0, eps=0.005, target=50.0)
    p25 = stockpile_pressure_penalty(obs_25, my_id=0, eps=0.005, target=50.0)
    assert abs(p50 / p25 - 4.0) < 1e-6


def test_stockpile_penalty_sums_across_planets():
    """Two planets with excess=50 each → penalty = 2 · ε · 50²."""
    obs = _obs_with(planets=[
        (0, 0, 0, 0, 1.0, 100, 2),
        (1, 0, 10, 0, 1.0, 100, 2),
    ])
    got = stockpile_pressure_penalty(obs, my_id=0, eps=0.005, target=50.0)
    expected = 2 * 0.005 * 2500.0
    assert abs(got - expected) < 1e-9


def test_select_favor_fn_layers_wave_terms():
    """End-to-end: with HHI+stockpile env vars set, select_favor_fn returns
    a wrapper that applies both on top of the base head."""
    from agents.baseline.value import select_favor_fn, favor
    obs = _obs_with(
        planets=[(0, 0, 0, 0, 1.0, 100, 2), (1, 1, 10, 0, 1.0, 10, 2)],
        fleets=[(0, 0, 0, 0, 0, 0, 50)],
    )
    import os as _os
    saved = {k: _os.environ.get(k) for k in
             ("BASELINE_HHI_BONUS", "BASELINE_STOCKPILE_PENALTY",
              "BASELINE_HHI_DELTA", "BASELINE_STOCKPILE_EPS",
              "BASELINE_STOCKPILE_TARGET")}
    try:
        # Clear → wrapper not installed.
        for k in saved:
            _os.environ.pop(k, None)
        bare = select_favor_fn()
        base_v = bare(obs, 0, 2, 0.99)
        assert bare is favor  # unwrapped

        _os.environ["BASELINE_HHI_BONUS"] = "1"
        _os.environ["BASELINE_STOCKPILE_PENALTY"] = "1"
        _os.environ["BASELINE_HHI_DELTA"] = "0.1"
        _os.environ["BASELINE_STOCKPILE_EPS"] = "0.005"
        _os.environ["BASELINE_STOCKPILE_TARGET"] = "50"
        wrapped = select_favor_fn()
        assert wrapped is not favor
        wrapped_v = wrapped(obs, 0, 2, 0.99)

        expect_delta = (
            inflight_hhi_bonus(obs, 0, delta=0.1)
            - stockpile_pressure_penalty(obs, 0, eps=0.005, target=50.0)
        )
        assert abs((wrapped_v - base_v) - expect_delta) < 1e-9
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
