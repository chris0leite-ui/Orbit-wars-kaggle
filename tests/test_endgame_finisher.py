"""Tests for `agents.buildup_planner.endgame` — the FINISHER phase.

quick_trigger:
  - returns (opp_id, count) when 2P + opp owns 1..K_FINISH planets
  - returns None when opp owns 0 planets (already eliminated)
  - returns None when opp owns >K_FINISH planets (FINISHER not yet)
  - returns None in 4P games (multiple distinct non-me owners)

evaluate:
  - returns None when the closed-form gate `is_winning_state_if_owned`
    fails (mock-driven — gate not satisfied)
  - returns a feasible plan in a real 2P scenario where opp has 1 planet
    and our production lead dominates
"""
from __future__ import annotations

from unittest.mock import patch

from lib.intent import World
from lib.world_model import WorldModel

from agents.buildup_planner import endgame
from agents.buildup_planner.predicates import StrikePlan


def _world(planets, fleets=None, step: int = 0) -> World:
    """Static (omega=0) world from a planet tuple list."""
    obs = {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    return World.from_obs(obs)


# ---- quick_trigger ---------------------------------------------------------


def test_quick_trigger_fires_on_2p_with_one_opp_planet():
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 0, 50.0, 15.0, 3.0, 50, 2),
        (2, 1, 85.0, 85.0, 3.0, 10, 1),   # one opp planet
    ]
    result = endgame.quick_trigger(planets, me=0)
    assert result == (1, 1)


def test_quick_trigger_skips_when_opp_above_k_finish():
    # K_FINISH=3 → 4 opp planets → no trigger.
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 1, 50.0, 15.0, 3.0, 10, 1),
        (2, 1, 85.0, 15.0, 3.0, 10, 1),
        (3, 1, 50.0, 85.0, 3.0, 10, 1),
        (4, 1, 15.0, 85.0, 3.0, 10, 1),
    ]
    assert endgame.quick_trigger(planets, me=0) is None


def test_quick_trigger_skips_when_no_opp():
    """All planets mine or neutral → opp_id stays -1 → no trigger."""
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, -1, 50.0, 15.0, 3.0, 10, 1),
    ]
    assert endgame.quick_trigger(planets, me=0) is None


def test_quick_trigger_skips_in_4p():
    """Two distinct non-me owners (players 1 and 2) → 4P → no trigger."""
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 1, 50.0, 15.0, 3.0, 10, 1),
        (2, 2, 85.0, 85.0, 3.0, 10, 1),
    ]
    assert endgame.quick_trigger(planets, me=0) is None


def test_quick_trigger_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BUILDUP_PLANNER_FINISHER_ENABLED", "0")
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 1, 85.0, 85.0, 3.0, 10, 1),
    ]
    assert endgame.quick_trigger(planets, me=0) is None


# ---- evaluate --------------------------------------------------------------


def test_evaluate_returns_none_when_gate_fails():
    """If `is_winning_state_if_owned` returns False, evaluate must return None
    without even attempting the wave search."""
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 1, 85.0, 85.0, 3.0, 10, 1),
    ]
    w = _world(planets)
    m = WorldModel.from_world(w)
    with patch.object(endgame, "is_winning_state_if_owned",
                      return_value=False):
        plan = endgame.evaluate(w, m, me=0, opp_id=1)
    assert plan is None


def test_evaluate_returns_plan_when_one_opp_planet_dominated():
    """Construct a scenario where:
      - I own 5 planets along y=15 and y=85 columns (dominant production)
      - Opp owns 1 planet at (85, 85), small garrison
      - Closed-form gate succeeds (prod lead × remaining turns dominates)
      - At least one feasible wave at some T → plan returned
    """
    planets = [
        # Me — 5 planets, prod=2 each, total prod=10.
        (0, 0, 15.0, 15.0, 3.0, 100, 2),
        (1, 0, 50.0, 15.0, 3.0, 100, 2),
        (2, 0, 85.0, 15.0, 3.0, 100, 2),
        (3, 0, 15.0, 85.0, 3.0, 100, 2),
        (4, 0, 50.0, 85.0, 3.0, 100, 2),
        # Opp — 1 planet, prod=1, small garrison.
        (5, 1, 85.0, 85.0, 3.0,  10, 1),
    ]
    w = _world(planets, step=0)
    m = WorldModel.from_world(w)
    plan = endgame.evaluate(w, m, me=0, opp_id=1)
    assert plan is not None
    assert isinstance(plan, StrikePlan)
    assert plan.target_ids == frozenset({5})
    # Single opp planet → single shot in the wave.
    assert len(plan.shots) == 1
    # Shot must originate from one of my sources and target opp planet 5.
    shot = plan.shots[0]
    assert int(shot.src_id) in {0, 1, 2, 3, 4}
    assert int(shot.tgt_id) == 5
    # Wave must beat the opp garrison * margin (10 * 1.10 = 11).
    assert int(shot.ship_count) > 11


def test_evaluate_disabled_by_env(monkeypatch):
    """Env flag flip should make `evaluate` short-circuit to None
    regardless of state."""
    monkeypatch.setenv("BUILDUP_PLANNER_FINISHER_ENABLED", "0")
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 100, 2),
        (1, 1, 85.0, 85.0, 3.0, 10, 1),
    ]
    w = _world(planets)
    m = WorldModel.from_world(w)
    assert endgame.evaluate(w, m, me=0, opp_id=1) is None
