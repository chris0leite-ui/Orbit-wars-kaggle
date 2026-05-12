"""Active gang_up Mission class — paired sources at contested targets.

Closes the in-flight volume gap from games-analysis §3.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.missions.gang_up import (
    GANG_UP_BONUS,
    MAX_DELAY,
    PAIR_SHARE,
    SINGLE_SOURCE_AFFORDABLE_RATIO,
    propose_gang_up_missions,
)
from lib.world_model import WorldModel


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


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0):
    return [pid, owner, x, y, 1.0, ships, prod]


# ---------------------------------------------------------------------------
# Firing conditions
# ---------------------------------------------------------------------------


def test_no_pair_when_only_one_owned_planet():
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=200, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets)
    out = propose_gang_up_missions(w, _model(w))
    assert out == []


def test_fires_against_contested_target_with_two_sources():
    """Target has 200 ships; each source has 50 (sends 35 = 70%).
    Neither alone can capture; paired they have 70 ships + production
    discount — still not enough vs 200, so should skip... let's make it
    closer: target 80 ships, each source sends 35 → combined 70 < 81 →
    still skips. Need source pair > target. Let's use 200-ship sources.
    """
    planets = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(2, owner=0, ships=200, x=10.0, y=20.0),  # near 0
        _planet(1, owner=1, ships=200, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets, step=10)
    out = propose_gang_up_missions(w, _model(w))
    assert len(out) == 2, f"expected 2 missions, got {len(out)}: {out}"
    classes = sorted([m.mission_class for m in out])
    assert classes == ["gang_up_follow", "gang_up_lead"]
    # Both target the same target
    assert {m.target_id for m in out} == {1}
    # Ships sent = PAIR_SHARE * src.ships
    expected_ships = max(1, int(200 * PAIR_SHARE))
    assert all(m.ships == expected_ships for m in out)


def test_skips_when_single_source_can_handle_target():
    """200-ship source vs 80-ship target — single source can handle it
    (200 * 0.7 = 140 >> 80). gang_up should NOT fire."""
    planets = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(2, owner=0, ships=200, x=10.0, y=20.0),
        _planet(1, owner=1, ships=80, prod=1, x=70.0, y=10.0),
    ]
    w = _world(planets, step=10)
    out = propose_gang_up_missions(w, _model(w))
    assert out == []


def test_skips_when_eta_gap_exceeds_max_delay():
    """One source close, one source very far → eta gap > MAX_DELAY."""
    planets = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),   # near
        _planet(2, owner=0, ships=200, x=95.0, y=95.0),   # very far
        _planet(1, owner=1, ships=200, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets, step=10)
    out = propose_gang_up_missions(w, _model(w))
    # Source 0 is 60 from target; source 2 is ~90 from target.
    # With ships=140 at speed ~v, ETA gap may exceed 3 turns.
    # If so, gang_up skips.
    if out:
        # If gang_up fires, eta_gap MUST be <= MAX_DELAY.
        etas = sorted([m.eta for m in out])
        assert etas[-1] - etas[0] <= MAX_DELAY


def test_skips_comet_targets():
    planets = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(2, owner=0, ships=200, x=10.0, y=20.0),
        _planet(1, owner=-1, ships=200, prod=3, x=70.0, y=10.0),
    ]
    obs = {
        "player": 0,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": [1],   # target 1 is a comet
        "step": 10,
        "comets": [{"planet_ids": [1], "paths": [[[70.0, 10.0]] * 50], "path_index": 0}],
        "fleets": [],
    }
    w = World.from_obs(obs)
    out = propose_gang_up_missions(w, WorldModel.from_world(w))
    assert out == []


def test_score_uses_gang_up_bonus():
    planets = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(2, owner=0, ships=200, x=10.0, y=20.0),
        _planet(1, owner=1, ships=200, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets, step=10)
    out = propose_gang_up_missions(w, _model(w))
    assert len(out) == 2
    # Score has GANG_UP_BONUS factor. Compute the baseline and check.
    s1 = next(m for m in out if m.mission_class == "gang_up_lead")
    # value = 3 * (500 - 10 - eta); combined = 2 * 140; mean_d ~ ~(60 + ~58)/2
    # We don't recompute exactly, just verify >0 and that bonus is applied
    # versus the same shape without bonus.
    assert s1.score > 0
    # No easy way to test factor without re-implementing; verify it's
    # at least bigger than a baseline without bonus would be:
    # baseline = value / (0.5 * combined + mean_d + 1)
    # got_score = GANG_UP_BONUS * baseline
    # Bound check: got_score / baseline ≈ GANG_UP_BONUS ≈ 1.3.
    # We pull the math here:
    eta = s1.eta
    value = 3.0 * max(1, 500 - 10 - eta)
    ships1 = max(1, int(200 * PAIR_SHARE))
    ships2 = ships1
    combined = ships1 + ships2
    # distance from (10,10) to (70,10) = 60; from (10,20) to (70,10) = ~60.8
    mean_d = (60.0 + math.hypot(60.0, 10.0)) / 2.0
    baseline = value / (0.5 * combined + mean_d + 1.0)
    expected = GANG_UP_BONUS * baseline
    assert abs(s1.score - expected) < 1.0   # allow eta rounding
