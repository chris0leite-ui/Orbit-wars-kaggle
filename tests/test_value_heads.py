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
    inflight_value,
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
# composite_capture_value — Phase 3b in-flight fate check
# ---------------------------------------------------------------------------


def test_composite_fleet_survival_check_off_by_default():
    """Phase 3b, 2026-05-22: COMPOSITE_FLEET_SURVIVAL_CHECK is OFF by
    default. With the env var unset, composite_capture_value returns
    the same value as the prior implementation. End-to-end smoke on
    a warmed snapshot: head returns a float without raising.
    """
    import os
    from lib.value_heads import composite_capture_value
    saved = os.environ.pop("COMPOSITE_FLEET_SURVIVAL_CHECK", None)
    try:
        snap = _warmed_snap(seed=42, warmup=15)
        obs = snap.state[0].observation
        v = composite_capture_value(obs, my_id=0)
        assert isinstance(v, float)
    finally:
        if saved is not None:
            os.environ["COMPOSITE_FLEET_SURVIVAL_CHECK"] = saved


def test_composite_fleet_survival_check_lowers_or_equal_to_baseline():
    """Phase 3b, 2026-05-22: with the check ON, the value head can only
    DECREASE (or stay equal to) the no-check baseline — it only adds
    waste penalties for fleets that fail the trajectory ray-cast,
    never adds positive credit. Run on a warmed mid-game snapshot
    where some fleets are in flight.

    Reload `value_heads` to pick up the env var (module-level constant).
    """
    import importlib
    import os
    saved = os.environ.get("COMPOSITE_FLEET_SURVIVAL_CHECK")
    snap = _warmed_snap(seed=7, warmup=25)
    obs = snap.state[0].observation
    try:
        # OFF baseline.
        if saved is not None:
            os.environ.pop("COMPOSITE_FLEET_SURVIVAL_CHECK")
        import lib.value_heads as vh
        importlib.reload(vh)
        v_off = vh.composite_capture_value(obs, my_id=0)

        # ON: same obs, check fires for any in-flight fleet that would
        # have been credited.
        os.environ["COMPOSITE_FLEET_SURVIVAL_CHECK"] = "1"
        importlib.reload(vh)
        v_on = vh.composite_capture_value(obs, my_id=0)
        # Monotone: check never INCREASES value.
        assert v_on <= v_off + 1e-9
    finally:
        if saved is None:
            os.environ.pop("COMPOSITE_FLEET_SURVIVAL_CHECK", None)
        else:
            os.environ["COMPOSITE_FLEET_SURVIVAL_CHECK"] = saved
        import lib.value_heads as vh
        importlib.reload(vh)
