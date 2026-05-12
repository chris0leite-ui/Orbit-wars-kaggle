"""Tests for `propose_snipe_missions(aggressive=True)` — the v3.5.1
ship-sizing change.

The `aggressive` flag bumps `base_ships` from `t.ships + 1` (minimum
viable) to `min(src.ships * 0.7, src.ships - 5)` when src has more
than 12 ships. Default (`aggressive=False`) is unchanged and
parity-gated by the test_replay_parity test.
"""

from __future__ import annotations

import math

from lib.intent import World
from lib.missions.snipe import (
    AGGRESSIVE_FRACTION,
    AGGRESSIVE_MIN_GARRISON,
    AGGRESSIVE_RESERVE,
    propose_snipe_missions,
)
from lib.world_model import WorldModel


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


# ---------------------------------------------------------------------------
# Default (aggressive=False) unchanged
# ---------------------------------------------------------------------------


def test_default_sizing_minimum_viable():
    """aggressive=False: base_ships = t.ships + 1 (unchanged from v3.4)."""
    planets = [
        _planet(0, owner=0, ships=100, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w))
    assert len(out) == 1
    assert out[0].ships == 6  # target_min = 5 + 1


# ---------------------------------------------------------------------------
# Aggressive sizing
# ---------------------------------------------------------------------------


def test_aggressive_sends_fraction_when_garrison_large():
    """aggressive=True with src.ships > 12: send ~70% of garrison."""
    planets = [
        _planet(0, owner=0, ships=100, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w), aggressive=True)
    assert len(out) == 1
    # min(int(100*0.7)=70, 100-5=95) = 70, max with target_min=6 → 70
    assert out[0].ships == 70


def test_aggressive_reserves_minimum_at_source():
    """aggressive=True: never send more than src.ships - 5."""
    # Source very small relative to fraction cap
    planets = [
        _planet(0, owner=0, ships=20, x=10.0, y=10.0),
        _planet(1, owner=1, ships=2, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w), aggressive=True)
    assert len(out) == 1
    # fraction_size = 14, cap = 15. min(14, 15) = 14. target_min = 3.
    # max(3, 14) = 14.
    assert out[0].ships == 14
    assert 20 - out[0].ships >= AGGRESSIVE_RESERVE


def test_aggressive_falls_back_to_minimum_for_small_sources():
    """aggressive=True but src.ships <= AGGRESSIVE_MIN_GARRISON:
    use minimum-viable formula (don't strand small sources)."""
    planets = [
        _planet(0, owner=0, ships=AGGRESSIVE_MIN_GARRISON, x=10.0, y=10.0),
        _planet(1, owner=1, ships=2, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w), aggressive=True)
    assert len(out) == 1
    assert out[0].ships == 3  # target_min, NOT the fraction


def test_aggressive_respects_target_minimum_on_high_garrison_target():
    """aggressive=True: never send less than what's needed to capture."""
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=40, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_snipe_missions(w, WorldModel.from_world(w), aggressive=True)
    assert len(out) == 1
    # fraction = int(50*0.7)=35; cap = 45; target_min = 41.
    # max(41, min(35, 45)) = max(41, 35) = 41 (target_min wins).
    assert out[0].ships == 41


def test_aggressive_constants():
    """Sanity check on the calibrated constants."""
    assert AGGRESSIVE_FRACTION == 0.7
    assert AGGRESSIVE_RESERVE == 5
    assert AGGRESSIVE_MIN_GARRISON == 12
