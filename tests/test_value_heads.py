"""Sanity tests for the v9 value heads."""

from __future__ import annotations

import math
import random

import pytest

from kaggle_environments import make

from lib import fast_sim
from lib.value_heads import (
    CAPTURE_REWARD_WEIGHT,
    INFLIGHT_EXTRA_HORIZON,
    INFLIGHT_WEIGHT,
    composite_capture_value,
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
# composite_capture_value — comet lifetime cap (2026-05-17)
# ---------------------------------------------------------------------------


def _obs_with_comet_target(path_len: int, path_index: int = 0):
    """Synthetic obs where our fleet is one tick from hitting a comet
    of known remaining-lifetime = path_len - path_index.

    The comet sits at (50, 50). Our fleet is at (48, 50) moving +x at
    angle 0. Our home planet at (10, 50) (so the fleet originated from
    it). One enemy planet at (90, 50) so the world isn't trivially
    empty. Fleet ships = 10 (low speed); comet has 0 garrison so we'll
    "capture" on arrival.

    `path_len` = total comet path length; `path_index` = where it is
    now. `comet_remaining_lifetime` returns `path_len - path_index`.
    """
    comet_id = 99
    return {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 1,
        "initial_planets": [],
        # Planet tuple: (id, owner, x, y, radius, ships, production)
        "planets": [
            (0, 0, 10.0, 50.0, 2.0, 50, 1),    # our home
            (comet_id, -1, 50.0, 50.0, 1.0, 0, 2),  # comet target
            (1, 1, 90.0, 50.0, 2.0, 50, 1),    # enemy
        ],
        # Fleet tuple: (id, owner, x, y, angle, from_planet_id, ships)
        "fleets": [(0, 0, 48.0, 50.0, 0.0, 0, 10)],
        "comet_planet_ids": [comet_id],
        "comets": [{
            "planet_ids": [comet_id],
            # Path stays put around (50, 50) — the test cares about
            # lifetime, not the comet's motion.
            "paths": [[[50.0, 50.0]] * path_len],
            "path_index": path_index,
        }],
    }


def test_composite_penalises_doomed_comet_target():
    """A fleet aimed at a comet that will expire BEFORE arrival hits
    empty space — composite must waste-penalise the launch. The
    WorldModel's per-planet timeline doesn't model comet expiry, so
    without the comet-lifetime gate the launch would slip through.
    Direct test of the 2026-05-17 PI direction: 'use comets only if
    really worth the risk and short lifetime'."""
    # Fleet at (48, 50) heading toward comet at (50, 50). Distance ~2,
    # speed for 10 ships ~1.96 → eta ≈ 1 tick.
    # path_len=1, path_index=0 → remaining lifetime = 1 tick at start.
    # comet_remaining_lifetime returns 1, eta=1, gate fires
    # (lifetime <= eta).
    doomed = _obs_with_comet_target(path_len=1, path_index=0)
    v_doomed = composite_capture_value(doomed, my_id=0)
    # Base (ship-delta): us=50+10=60, them=50 → 10. Waste penalty:
    # waste_weight (=0.5) * 10 ships = -5. Net: 10 - 5 = 5.
    # Without the gate composite would return ≥10 (no penalty).
    assert v_doomed < 10.0, (
        f"comet-lifetime-gate did not fire: v_doomed={v_doomed}, "
        f"expected < base ship-delta (10) due to waste penalty"
    )


def test_composite_allows_long_lived_comet_target():
    """A comet with plenty of life left is a legitimate target — the
    gate must NOT fire. composite returns base ship-delta (no penalty)
    since the pred_owner==me skip prevents capture credit anyway in
    this single-fleet scenario."""
    alive = _obs_with_comet_target(path_len=200, path_index=0)
    v_alive = composite_capture_value(alive, my_id=0)
    assert math.isclose(v_alive, 10.0, abs_tol=1e-6), (
        f"long-lived comet target should pass gate cleanly; got {v_alive}"
    )


def test_composite_gate_distinguishes_short_vs_long_comet():
    """Direct A/B: same fleet, same target, only the comet's remaining
    lifetime differs. Short-life is penalised; long-life is not."""
    short = composite_capture_value(_obs_with_comet_target(path_len=1, path_index=0), my_id=0)
    long_ = composite_capture_value(_obs_with_comet_target(path_len=200, path_index=0), my_id=0)
    assert short < long_, (
        f"short-life comet value ({short}) should be < long-life "
        f"({long_}) — gate must penalise the doomed launch"
    )
