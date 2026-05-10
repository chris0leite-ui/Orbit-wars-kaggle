"""Unit tests for lib.scoring — projected-arrival helpers used by ROI variants."""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib import scoring


def _planet(pid, owner, x, y, ships, production):
    return Planet(pid, owner, x, y, 1.0, ships, production)


def test_eta_proxy_zero_distance_is_zero():
    p = _planet(0, 0, 50, 50, 100, 1)
    assert scoring.eta_proxy(p, p) == 0


def test_eta_proxy_positive_for_typical_pair():
    src = _planet(0, 0, 10, 10, 100, 1)
    tgt = _planet(1, -1, 50, 50, 30, 2)
    assert scoring.eta_proxy(src, tgt) > 0


def test_projected_garrison_neutral_does_not_grow():
    tgt = _planet(1, -1, 50, 50, 20, 5)
    assert scoring.projected_garrison(tgt, eta=10) == 20


def test_projected_garrison_enemy_grows_by_production_times_eta():
    tgt = _planet(1, 1, 50, 50, 20, 3)
    assert scoring.projected_garrison(tgt, eta=10) == 20 + 30


def test_s_needed_strict_win():
    tgt = _planet(1, 1, 50, 50, 10, 2)
    assert scoring.s_needed(tgt, eta=5) == 10 + 10 + 1


def test_horizon_clamps_to_zero_late_game():
    assert scoring.horizon(step=500, eta=10) == 0
    assert scoring.horizon(step=499, eta=10) == 0
    assert scoring.horizon(step=400, eta=10) == 90


def test_margin_multiplier_enemy_is_2_neutral_is_1_self_is_0():
    enemy = _planet(1, 1, 0, 0, 0, 0)
    neutral = _planet(2, -1, 0, 0, 0, 0)
    own = _planet(3, 0, 0, 0, 0, 0)
    assert scoring.margin_multiplier(enemy, my_id=0) == 2
    assert scoring.margin_multiplier(neutral, my_id=0) == 1
    assert scoring.margin_multiplier(own, my_id=0) == 0
