"""Recapture Mission — retake recently-lost planets within a 50-turn window.

Closes the comeback gap from games-analysis §2: in wins recover to
median 28 planets after home loss; in losses peak at 6.
"""

from __future__ import annotations

import pytest

from lib.intent import World
from lib.missions.recapture import (
    RECAPTURE_BONUS_PEAK,
    RECAPTURE_WINDOW,
    RECENTLY_LOST_GARRISON_MAX,
    _reset_state_for_tests,
    propose_recapture_missions,
)
from lib.world_model import WorldModel


@pytest.fixture(autouse=True)
def _isolate_state():
    """Each test gets a fresh module-level state."""
    _reset_state_for_tests()
    yield
    _reset_state_for_tests()


def _world(planets, *, my_id=0, step=0):
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


def _model(world):
    return WorldModel.from_world(world)


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0):
    return [pid, owner, x, y, 1.0, ships, prod]


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


def test_no_recapture_on_first_turn():
    """At step 0 there's no history; no losses to act on."""
    planets = [_planet(0, owner=0, ships=10), _planet(1, owner=-1, ships=5)]
    w = _world(planets, step=0)
    out = propose_recapture_missions(w, _model(w))
    assert out == []


def test_detects_loss_and_proposes_recapture_next_turn():
    """Turn 1: planet 1 is ours. Turn 2: planet 1 is enemy. Turn 2 should
    fire a recapture mission."""
    planets_t1 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=2, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=2, x=70.0, y=10.0),  # now enemy
    ]
    w1 = _world(planets_t1, step=20)
    propose_recapture_missions(w1, _model(w1))   # seed state
    w2 = _world(planets_t2, step=21)
    out = propose_recapture_missions(w2, _model(w2))
    assert len(out) == 1
    m = out[0]
    assert m.mission_class == "recapture"
    assert m.target_id == 1


def test_evicts_stale_lost_records_after_window():
    """A loss at step 20 should be evicted by step 71 (window=50)."""
    planets_t1 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    propose_recapture_missions(_world(planets_t1, step=20),
                                _model(_world(planets_t1, step=20)))
    propose_recapture_missions(_world(planets_t2, step=21),
                                _model(_world(planets_t2, step=21)))
    # 51 turns later — stale
    w_stale = _world(planets_t2, step=21 + RECAPTURE_WINDOW + 1)
    out = propose_recapture_missions(w_stale, _model(w_stale))
    assert out == []


def test_drops_record_when_we_retake_planet():
    """If we recapture (planet flips back to us), the lost record clears."""
    planets_t1 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    planets_t3 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=70.0, y=10.0),   # back to us
    ]
    propose_recapture_missions(_world(planets_t1, step=20),
                                _model(_world(planets_t1, step=20)))
    propose_recapture_missions(_world(planets_t2, step=21),
                                _model(_world(planets_t2, step=21)))
    propose_recapture_missions(_world(planets_t3, step=22),
                                _model(_world(planets_t3, step=22)))
    # Now lose it AGAIN at step 23 — fresh window starts.
    planets_t4 = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=1, x=70.0, y=10.0),
    ]
    out = propose_recapture_missions(_world(planets_t4, step=23),
                                       _model(_world(planets_t4, step=23)))
    assert len(out) == 1


def test_state_resets_on_step_zero():
    """A new game's step=0 obs clears any previous lost records."""
    # Game 1: lose planet 1
    p_g1_t1 = [_planet(0, owner=0, ships=30, x=10.0, y=10.0),
                _planet(1, owner=0, ships=5, x=70.0, y=10.0)]
    p_g1_t2 = [_planet(0, owner=0, ships=30, x=10.0, y=10.0),
                _planet(1, owner=1, ships=5, x=70.0, y=10.0)]
    propose_recapture_missions(_world(p_g1_t1, step=20),
                                _model(_world(p_g1_t1, step=20)))
    propose_recapture_missions(_world(p_g1_t2, step=21),
                                _model(_world(p_g1_t2, step=21)))
    # Game 2 starts: same planets but step=0
    out = propose_recapture_missions(_world(p_g1_t2, step=0),
                                       _model(_world(p_g1_t2, step=0)))
    assert out == []


# ---------------------------------------------------------------------------
# Skipping conditions
# ---------------------------------------------------------------------------


def test_skips_target_fortified_beyond_threshold():
    """If the new owner has built up garrison > threshold, recapture skips."""
    planets_t1 = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=RECENTLY_LOST_GARRISON_MAX + 1,
                 prod=1, x=70.0, y=10.0),  # fortified
    ]
    propose_recapture_missions(_world(planets_t1, step=20),
                                _model(_world(planets_t1, step=20)))
    out = propose_recapture_missions(_world(planets_t2, step=21),
                                       _model(_world(planets_t2, step=21)))
    assert out == []


def test_skips_if_no_owned_source_can_afford():
    """We have only one owned planet with too few ships to capture."""
    planets_t1 = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(1, owner=1, ships=40, prod=1, x=70.0, y=10.0),
    ]
    propose_recapture_missions(_world(planets_t1, step=20),
                                _model(_world(planets_t1, step=20)))
    out = propose_recapture_missions(_world(planets_t2, step=21),
                                       _model(_world(planets_t2, step=21)))
    assert out == []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_urgency_decays_over_window():
    """Score immediately after loss should be higher than score later
    in the window (urgency multiplier decays)."""
    planets_t1 = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=2, x=70.0, y=10.0),
    ]
    planets_t2 = [
        _planet(0, owner=0, ships=200, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    propose_recapture_missions(_world(planets_t1, step=20),
                                _model(_world(planets_t1, step=20)))
    out_immediate = propose_recapture_missions(
        _world(planets_t2, step=21), _model(_world(planets_t2, step=21))
    )
    # 30 turns later — same world state but urgency lower
    out_later = propose_recapture_missions(
        _world(planets_t2, step=51), _model(_world(planets_t2, step=51))
    )
    assert out_immediate[0].score > out_later[0].score
