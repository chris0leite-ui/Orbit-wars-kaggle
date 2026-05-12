"""Tests for lib/geometry.py — pure-numeric primitives.

Specs are taken from `data/README.md::Board Layout` and `Configuration`.
"""

from __future__ import annotations

import math

import pytest

import agent as G


def test_constants_match_readme():
    assert G.BOARD_SIZE == 100.0
    assert G.CENTER == 50.0
    assert G.SUN_RADIUS == 10.0
    assert G.ROTATION_RADIUS_LIMIT == 50.0


def test_dist_3_4_5_triangle():
    assert G.dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_dist_zero_when_points_coincide():
    assert G.dist((42.0, 42.0), (42.0, 42.0)) == 0.0


def test_point_to_segment_distance_perpendicular_drop():
    # Drop a perpendicular from (5, 5) onto the x-axis from (0,0) to (10,0).
    assert G.point_to_segment_distance((5.0, 5.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(5.0)


def test_point_to_segment_distance_outside_segment_clamps_to_endpoint():
    # Point past the b end — distance is to b, not to the infinite line.
    d = G.point_to_segment_distance((20.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    assert d == pytest.approx(10.0)


def test_point_to_segment_distance_zero_length_segment_is_point_distance():
    d = G.point_to_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0))
    assert d == pytest.approx(5.0)


def test_path_clears_sun_diagonal_through_sun_is_blocked():
    # Corner to corner runs straight through the sun at (50, 50).
    assert G.path_clears_sun((0.0, 0.0), (100.0, 100.0)) is False


def test_path_clears_sun_along_left_edge_is_clear():
    # Left edge is 50 units from sun centre, well outside the 10-radius.
    assert G.path_clears_sun((0.0, 0.0), (0.0, 100.0)) is True


def test_path_clears_sun_safety_margin_tightens_check():
    # Path that grazes the sun radius is clear at safety=0 but blocked at safety=5.
    src = (40.0, 40.0)
    dst = (60.0, 60.0)
    assert G.path_clears_sun(src, dst, safety=0.0) is False  # well inside
    # Path tangent to sun on its boundary at exactly SUN_RADIUS away:
    src = (60.0, 0.0)  # x=60, sun centre x=50 → perpendicular distance 10.0 exactly
    dst = (60.0, 100.0)
    # Strict > comparison: distance == sun_radius means NOT clear.
    assert G.path_clears_sun(src, dst, safety=0.0) is False
    # Adding a 5-unit safety margin further blocks it:
    assert G.path_clears_sun(src, dst, safety=5.0) is False
    # Pull back the path by 6 units → now > 10 + 5
    src = (66.0, 0.0)
    dst = (66.0, 100.0)
    assert G.path_clears_sun(src, dst, safety=5.0) is True
