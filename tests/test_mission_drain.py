"""Source-drain Mission class — empties safe high-garrison planets.

Top-10 fingerprint: mean garrison-at-launch 11 (midpack 22). Drain is
the safety-gated version of "send the surplus."
"""

from __future__ import annotations

import math

import pytest

import lib.missions.drain as _drain_mod
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.missions.drain import (
    DRAIN_BONUS,
    MIN_DRAIN_SHIPS,
    RESERVE_KEEP,
    SAFE_ETA_BUFFER,
    propose_drain_missions,
)
from lib.world_model import WorldModel


@pytest.fixture(autouse=True)
def _enable_drain_mission():
    saved = _drain_mod.USE_DRAIN_MISSION
    _drain_mod.USE_DRAIN_MISSION = 1
    try:
        yield
    finally:
        _drain_mod.USE_DRAIN_MISSION = saved


def _world(planets, *, my_id=0, step=20, fleets=()):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": list(fleets),
    }
    return World.from_obs(obs)


def _model(world):
    return WorldModel.from_world(world)


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0, radius=1.0):
    return [pid, owner, x, y, radius, ships, prod]


# Fleet schema: [id, owner, x, y, angle, from_planet_id, ships]
def _fleet(fid, owner, x, y, angle, from_pid, ships):
    return [fid, owner, x, y, angle, from_pid, ships]


# ---------------------------------------------------------------------------
# Firing conditions
# ---------------------------------------------------------------------------


def test_skips_source_below_min_garrison():
    planets = [
        _planet(0, owner=0, ships=MIN_DRAIN_SHIPS, x=10.0, y=10.0),  # boundary
        _planet(1, owner=-1, ships=5, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_drain_missions(w, _model(w))
    assert out == []


def test_fires_for_surplus_source_with_safe_target():
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),  # 50 > MIN_DRAIN_SHIPS
        _planet(1, owner=-1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_drain_missions(w, _model(w))
    assert len(out) == 1
    m = out[0]
    assert m.mission_class == "drain"
    assert m.ships == 50 - RESERVE_KEEP   # drain_ships = src.ships - RESERVE_KEEP


def test_skips_when_enemy_inbound_too_soon():
    """An enemy fleet 2 turns from our source aborts the drain regardless
    of how good the target looks."""
    # Source at (10,10); enemy fleet at (12, 10) moving toward us — eta ~ 1
    planets = [
        _planet(0, owner=0, ships=80, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    fleets = [_fleet(0, owner=1, x=12.0, y=10.0, angle=math.pi, from_pid=2, ships=30)]
    w = _world(planets, fleets=fleets)
    # WorldModel.incoming_enemy_eta should detect this.
    model = _model(w)
    eta = model.incoming_enemy_eta(0, my_id=0)
    assert eta is not None and eta <= 3
    out = propose_drain_missions(w, model)
    assert out == []


def test_skips_target_we_cannot_capture():
    """50-ship surplus can't take a target with 200 ships."""
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=200, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_drain_missions(w, _model(w))
    assert out == []


def test_score_uses_drain_bonus_and_rebalanced_denominator():
    """score = DRAIN_BONUS * value / (0.5 * drain_ships + d + 1)."""
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets, step=20)
    out = propose_drain_missions(w, _model(w))
    assert len(out) == 1
    m = out[0]
    drain_ships = 50 - RESERVE_KEEP   # 42
    d = 60.0
    v = fleet_speed(drain_ships)
    eta = int(math.ceil(d / v))
    remaining = max(1, 500 - 20 - eta)
    value = 3.0 * remaining
    expected = DRAIN_BONUS * value / (0.5 * drain_ships + d + 1.0)
    assert abs(m.score - expected) < 1e-6


def test_skips_target_already_predicted_ours():
    """If our existing in-flight fleets will own the target, drain skips."""
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    # Friendly fleet 100 ships nearly at target → flips to ours within eta
    fleets = [_fleet(0, owner=0, x=65.0, y=10.0, angle=0.0, from_pid=2, ships=100)]
    w = _world(planets, fleets=fleets)
    out = propose_drain_missions(w, _model(w))
    # Drain may still propose against other (no other targets here) — empty.
    assert out == []


def test_does_not_strand_source_below_reserve():
    """Output mission's `ships` must leave RESERVE_KEEP ships at source."""
    planets = [
        _planet(0, owner=0, ships=100, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_drain_missions(w, _model(w))
    assert len(out) == 1
    assert 100 - out[0].ships >= RESERVE_KEEP
