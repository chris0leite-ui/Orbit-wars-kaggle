"""Smoke + invariant tests for `lib.value_features.extract_features`.

What these guarantee:
  - Output shape `(40,)`, dtype `float32`.
  - No NaN/Inf on degenerate boards (no planets, no fleets).
  - Determinism (same obs -> same vector, bit-exact).
  - Seat-symmetry: swapping seat ownership swaps the per-seat blocks.

What these do NOT guarantee:
  - Feature semantic correctness (caught downstream by the parity test
    that runs both `favor` and `favor_learned` on the same snap, plus
    by the A/B Wilson gate).
"""

from __future__ import annotations

import numpy as np

from lib.value_features import FEATURE_DIM, PER_SEAT_FEATURES, extract_features


def _mk_obs(planets: list, fleets: list, step: int = 50) -> dict:
    return {"planets": planets, "fleets": fleets, "step": step}


def _planet(pid: int, owner: int, x: float, y: float, ships: float,
            prod: float, radius: float = 2.0) -> tuple:
    return (pid, owner, x, y, radius, ships, prod)


def _fleet(fid: int, owner: int, x: float, y: float, ships: float,
           src: int = 0, angle: float = 0.0) -> tuple:
    return (fid, owner, x, y, angle, src, ships)


def test_shape_and_dtype():
    obs = _mk_obs([_planet(0, 0, 20, 20, 10, 2.0)], [])
    v = extract_features(obs, me=0, num_seats=2)
    assert v.shape == (FEATURE_DIM,)
    assert v.dtype == np.float32


def test_no_nan_on_empty_board():
    v = extract_features(_mk_obs([], []), me=0, num_seats=2)
    assert np.isfinite(v).all()
    assert v.shape == (FEATURE_DIM,)


def test_no_nan_on_one_sided_board():
    # All planets belong to me; no opp.
    planets = [_planet(0, 0, 20, 20, 10, 2.0), _planet(1, 0, 80, 80, 5, 1.0)]
    v = extract_features(_mk_obs(planets, []), me=0, num_seats=2)
    assert np.isfinite(v).all()


def test_determinism():
    planets = [
        _planet(0, 0, 20, 20, 10, 2.0),
        _planet(1, 1, 80, 80, 8, 1.5),
        _planet(2, -1, 50, 30, 3, 0.5),
    ]
    fleets = [_fleet(0, 0, 40, 40, 5)]
    obs = _mk_obs(planets, fleets)
    v1 = extract_features(obs, me=0, num_seats=2)
    v2 = extract_features(obs, me=0, num_seats=2)
    np.testing.assert_array_equal(v1, v2)


def test_seat_symmetry_2p():
    # Build a board, then run extract for me=0 and me=1. The per-seat
    # blocks should swap, the global block must stay the same.
    planets = [
        _planet(0, 0, 20, 20, 10, 2.0),
        _planet(1, 1, 80, 80, 8, 1.5),
        _planet(2, 0, 30, 70, 5, 1.0),
        _planet(3, 1, 70, 30, 6, 1.2),
    ]
    fleets = [_fleet(0, 0, 40, 40, 5), _fleet(1, 1, 60, 60, 4)]
    obs = _mk_obs(planets, fleets, step=100)

    v_me0 = extract_features(obs, me=0, num_seats=2)
    v_me1 = extract_features(obs, me=1, num_seats=2)

    # Block 0 of v_me0 (me) should equal block 1 of v_me1 (opp).
    np.testing.assert_array_equal(
        v_me0[:PER_SEAT_FEATURES],
        v_me1[PER_SEAT_FEATURES:2 * PER_SEAT_FEATURES],
    )
    # And vice versa.
    np.testing.assert_array_equal(
        v_me0[PER_SEAT_FEATURES:2 * PER_SEAT_FEATURES],
        v_me1[:PER_SEAT_FEATURES],
    )
    # Global block must be invariant.
    np.testing.assert_array_equal(
        v_me0[2 * PER_SEAT_FEATURES:],
        v_me1[2 * PER_SEAT_FEATURES:],
    )


def test_global_block_step_normalised():
    v = extract_features(_mk_obs([], [], step=250), me=0, num_seats=2)
    g = 2 * PER_SEAT_FEATURES
    assert v[g + 0] == np.float32(0.5)  # step / 500


def test_ship_total_includes_in_flight():
    planets = [_planet(0, 0, 20, 20, 10, 2.0)]
    fleets = [_fleet(0, 0, 40, 40, 7)]
    v = extract_features(_mk_obs(planets, fleets), me=0, num_seats=2)
    # Block 0, feature 0 = ship_total
    assert v[0] == np.float32(17)
    # Block 0, feature 4 = in_flight_ship_total
    assert v[4] == np.float32(7)
    # Block 0, feature 7 = planet_ships_total
    assert v[7] == np.float32(10)


def test_4p_aggregation_sums_opps():
    planets = [
        _planet(0, 0, 20, 20, 10, 2.0),
        _planet(1, 1, 60, 60, 5, 1.0),
        _planet(2, 2, 80, 20, 4, 0.8),
        _planet(3, 3, 20, 80, 3, 0.5),
    ]
    v = extract_features(_mk_obs(planets, []), me=0, num_seats=4)
    # Opp block (1) — feature 2 = planet_count summed over 3 opps = 3.
    assert v[PER_SEAT_FEATURES + 2] == np.float32(3)
    # Opp block feature 7 = sum of opp planet ships = 5+4+3 = 12.
    assert v[PER_SEAT_FEATURES + 7] == np.float32(12)


def test_obs_object_attr_path():
    """`extract_features` must accept BOTH dict and attr-style obs."""

    class _ObsObj:
        def __init__(self, planets, fleets, step):
            self.planets = planets
            self.fleets = fleets
            self.step = step

    planets = [_planet(0, 0, 20, 20, 10, 2.0)]
    obs_obj = _ObsObj(planets, [], 50)
    obs_dict = _mk_obs(planets, [], 50)

    v_obj = extract_features(obs_obj, me=0, num_seats=2)
    v_dict = extract_features(obs_dict, me=0, num_seats=2)
    np.testing.assert_array_equal(v_obj, v_dict)
