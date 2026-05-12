"""Tests for the aggressive ship-sizing path in propose_snipe_missions.

base_ships = max(target_min, min(int(src.ships * 0.7), src.ships - 5)) when
src has more than 12 ships; otherwise target_min (= t.ships + 1). The
fraction/reserve/min-garrison constants come from the v3.5.1 top-10
fingerprint calibration.
"""

from __future__ import annotations

from agent import (
    AGGRESSIVE_FRACTION,
    AGGRESSIVE_MIN_GARRISON,
    AGGRESSIVE_RESERVE,
    World,
    WorldModel,
    propose_snipe_missions,
)


def _world(planets, *, my_id=0, step=10):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": [],
    }
    return World.from_obs(obs)


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0):
    return [pid, owner, x, y, 1.0, ships, prod]


def test_aggressive_sends_fraction_when_garrison_large():
    """src.ships > 12: send ~70% of garrison."""
    planets = [
        _planet(0, owner=0, ships=100, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w))
    assert len(out) == 1
    # min(int(100*0.7)=70, 100-5=95) = 70, max with target_min=6 → 70
    assert out[0].ships == 70


def test_aggressive_reserves_minimum_at_source():
    """Never send more than src.ships - AGGRESSIVE_RESERVE."""
    planets = [
        _planet(0, owner=0, ships=20, x=10.0, y=10.0),
        _planet(1, owner=1, ships=2, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w))
    assert len(out) == 1
    # fraction_size = 14, cap = 15. min(14, 15) = 14. target_min = 3.
    assert out[0].ships == 14
    assert 20 - out[0].ships >= AGGRESSIVE_RESERVE


def test_aggressive_falls_back_to_minimum_for_small_sources():
    """src.ships <= AGGRESSIVE_MIN_GARRISON: use minimum-viable formula."""
    planets = [
        _planet(0, owner=0, ships=AGGRESSIVE_MIN_GARRISON, x=10.0, y=10.0),
        _planet(1, owner=1, ships=2, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w))
    assert len(out) == 1
    assert out[0].ships == 3  # target_min, NOT the fraction


def test_aggressive_respects_target_minimum_on_high_garrison_target():
    """Never send less than what's needed to capture."""
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=40, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w))
    assert len(out) == 1
    # fraction = int(50*0.7)=35; cap = 45; target_min = 41. max(41, 35) = 41.
    assert out[0].ships == 41


def test_aggressive_constants():
    assert AGGRESSIVE_FRACTION == 0.7
    assert AGGRESSIVE_RESERVE == 5
    assert AGGRESSIVE_MIN_GARRISON == 12
