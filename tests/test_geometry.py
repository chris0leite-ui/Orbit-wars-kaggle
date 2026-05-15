"""Tests for lib/geometry.py — pure-numeric primitives.

Specs are taken from `data/README.md::Board Layout` and `Configuration`.
"""

from __future__ import annotations

import math

import pytest

from lib import geometry as G


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


# ---------------------------------------------------------------------------
# danger_3nn — 3-NN allegiance count (H17 / TID 699003)
# ---------------------------------------------------------------------------


class _P:
    """Minimal duck-typed planet for danger_3nn tests."""
    __slots__ = ("id", "x", "y", "owner")
    def __init__(self, pid, x, y, owner):
        self.id = pid
        self.x = x
        self.y = y
        self.owner = owner


def test_danger_3nn_excludes_the_target_itself():
    # Target at origin; the only planets are at large distances. Target's
    # own owner field must not contribute (we exclude by id).
    target = _P(0, 0.0, 0.0, owner=0)
    others = [_P(1, 5.0, 0.0, owner=0), _P(2, 10.0, 0.0, owner=1)]
    score = G.danger_3nn((target.x, target.y), target.id, [target, *others], my_id=0)
    assert score == 0  # +1 ally, -1 enemy = 0


def test_danger_3nn_all_three_neighbours_are_allies():
    target = _P(0, 0.0, 0.0, owner=-1)
    allies = [_P(i, i + 1.0, 0.0, owner=0) for i in (1, 2, 3)]
    far_enemy = _P(99, 50.0, 50.0, owner=1)
    score = G.danger_3nn((0.0, 0.0), 0, [target, *allies, far_enemy], my_id=0)
    assert score == 3


def test_danger_3nn_all_three_neighbours_are_enemies():
    target = _P(0, 0.0, 0.0, owner=-1)
    enemies = [_P(i, i + 1.0, 0.0, owner=1) for i in (1, 2, 3)]
    far_ally = _P(99, 50.0, 50.0, owner=0)
    score = G.danger_3nn((0.0, 0.0), 0, [target, *enemies, far_ally], my_id=0)
    assert score == -3


def test_danger_3nn_neutrals_contribute_zero():
    target = _P(0, 0.0, 0.0, owner=-1)
    near = [
        _P(1, 1.0, 0.0, owner=-1),    # neutral → +0
        _P(2, 2.0, 0.0, owner=0),     # ally → +1
        _P(3, 3.0, 0.0, owner=1),     # enemy → -1
    ]
    score = G.danger_3nn((0.0, 0.0), 0, [target, *near], my_id=0)
    assert score == 0


def test_danger_3nn_uses_only_the_three_nearest():
    target = _P(0, 0.0, 0.0, owner=-1)
    near_allies = [_P(i, i + 1.0, 0.0, owner=0) for i in (1, 2, 3)]
    # 4th and 5th planet are enemies but FARTHER — they should be ignored.
    far_enemies = [_P(i, i + 10.0, 0.0, owner=1) for i in (4, 5)]
    score = G.danger_3nn(
        (target.x, target.y), target.id,
        [target, *near_allies, *far_enemies], my_id=0,
    )
    assert score == 3  # only the 3 near allies count


def test_danger_3nn_empty_or_single_planet_returns_zero():
    only_target = _P(0, 0.0, 0.0, owner=0)
    assert G.danger_3nn((0.0, 0.0), 0, [only_target], my_id=0) == 0
    assert G.danger_3nn((0.0, 0.0), 0, [], my_id=0) == 0


def test_danger_3nn_my_id_matches_perspective():
    # Same board, different "me": +1 becomes -1.
    target = _P(0, 0.0, 0.0, owner=-1)
    p0_planet = _P(1, 1.0, 0.0, owner=0)
    p1_planet = _P(2, 2.0, 0.0, owner=1)
    planets = [target, p0_planet, p1_planet]
    assert G.danger_3nn((0.0, 0.0), 0, planets, my_id=0) == 0
    # Flip perspective: same planets, but we are P1 now.
    # Wait — we only have 2 non-target neighbours. p0 was ally for my_id=0;
    # for my_id=1 the ally is p1 and enemy is p0 → still net 0.
    assert G.danger_3nn((0.0, 0.0), 0, planets, my_id=1) == 0
    # Add an extra: a third planet owned by P0. From P0's POV: 2 allies + 1
    # enemy = +1. From P1's POV: 1 ally + 2 enemies = -1.
    p0b = _P(3, 3.0, 0.0, owner=0)
    planets.append(p0b)
    assert G.danger_3nn((0.0, 0.0), 0, planets, my_id=0) == 1
    assert G.danger_3nn((0.0, 0.0), 0, planets, my_id=1) == -1
