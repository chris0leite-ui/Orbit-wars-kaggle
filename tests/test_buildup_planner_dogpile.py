"""Tests for `agents.buildup_planner.dogpile` — mid-game multi-source wave.

quick_trigger:
  - returns (opp_id, count) when 2P + opp owns > K_FINISH planets (default 3)
  - returns None when opp owns <= K_FINISH (FINISHER's job)
  - returns None when opp owns 0 (no opp left)
  - returns None in 4P games (multi-opp)
  - returns None when env flag is unset (default OFF)

evaluate:
  - returns None when post-capture production-advantage gate fails
  - returns a feasible plan in a 2P scenario where the top-K opp planets
    are capturable and the post-capture gate clears
  - target_ids contains exactly the top-K by production
"""
from __future__ import annotations

from lib.intent import World
from lib.world_model import WorldModel

from agents.buildup_planner import dogpile
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


def test_quick_trigger_fires_on_2p_above_k_finish(monkeypatch):
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    # K_FINISH=3 → 4 opp planets → dogpile should fire.
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 100, 2),
        (1, 0, 50.0, 15.0, 3.0, 100, 2),
        (2, 1, 85.0, 15.0, 3.0,  10, 1),
        (3, 1, 15.0, 85.0, 3.0,  10, 1),
        (4, 1, 50.0, 85.0, 3.0,  10, 1),
        (5, 1, 85.0, 85.0, 3.0,  10, 1),
    ]
    result = dogpile.quick_trigger(planets, me=0)
    assert result == (1, 4)


def test_quick_trigger_defers_to_finisher_at_or_below_k_finish(monkeypatch):
    """When opp has <= K_FINISH planets, dogpile must defer to FINISHER."""
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 100, 2),
        (1, 0, 50.0, 15.0, 3.0, 100, 2),
        (2, 1, 85.0, 15.0, 3.0,  10, 1),
        (3, 1, 15.0, 85.0, 3.0,  10, 1),
        (4, 1, 50.0, 85.0, 3.0,  10, 1),
    ]
    # opp has 3 planets — FINISHER territory.
    assert dogpile.quick_trigger(planets, me=0) is None


def test_quick_trigger_skips_when_no_opp(monkeypatch):
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, -1, 50.0, 15.0, 3.0, 10, 1),
    ]
    assert dogpile.quick_trigger(planets, me=0) is None


def test_quick_trigger_skips_in_4p(monkeypatch):
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 50, 2),
        (1, 1, 50.0, 15.0, 3.0, 10, 1),
        (2, 2, 85.0, 85.0, 3.0, 10, 1),
        (3, 1, 15.0, 85.0, 3.0, 10, 1),
        (4, 1, 85.0, 15.0, 3.0, 10, 1),
    ]
    assert dogpile.quick_trigger(planets, me=0) is None


def test_quick_trigger_disabled_by_env_default():
    """Default env is unset → _enabled() returns False → trigger returns None."""
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 100, 2),
        (1, 1, 50.0, 15.0, 3.0,  10, 1),
        (2, 1, 85.0, 15.0, 3.0,  10, 1),
        (3, 1, 15.0, 85.0, 3.0,  10, 1),
        (4, 1, 50.0, 85.0, 3.0,  10, 1),
    ]
    assert dogpile.quick_trigger(planets, me=0) is None


# ---- evaluate --------------------------------------------------------------


def test_evaluate_returns_none_when_prod_gate_fails(monkeypatch):
    """If our production after capture doesn't exceed opp * margin, bail."""
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    # Me: 1 planet, prod=1. Opp: 5 planets, prod=2 each (total 10).
    # Top-3 captured → my_prod=1+6=7, opp_prod=10-6=4. 7 > 4*1.10=4.4 → gate ACTUALLY clears.
    # We need a case where it FAILS. Use opp with 8 planets prod=2 (total 16),
    # top-3 captured → my=1+6=7, opp=16-6=10. 7 < 10*1.10=11 → fails.
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 100, 1),  # me, small prod
        (1, 1, 50.0, 15.0, 3.0,  20, 2),
        (2, 1, 85.0, 15.0, 3.0,  20, 2),
        (3, 1, 15.0, 85.0, 3.0,  20, 2),
        (4, 1, 50.0, 85.0, 3.0,  20, 2),
        (5, 1, 85.0, 85.0, 3.0,  20, 2),
        (6, 1, 30.0, 30.0, 3.0,  20, 2),
        (7, 1, 70.0, 30.0, 3.0,  20, 2),
        (8, 1, 30.0, 70.0, 3.0,  20, 2),
    ]
    w = _world(planets, step=0)
    m = WorldModel.from_world(w)
    assert dogpile.evaluate(w, m, me=0, opp_id=1) is None


def test_evaluate_returns_plan_when_top_k_dominated(monkeypatch):
    """Construct a scenario where:
      - I own 5 large planets along y=10 (bottom edge, clean paths to top)
      - Opp owns 4 small planets along y=90 (top edge), top-3 prod=2, id 8 prod=1
      - All vertical-ish paths avoid the sun at (50, 50) radius 10
      - Top-3 capturable; post-capture prod gate clears
    """
    monkeypatch.setenv("BUILDUP_PLANNER_DOGPILE_ENABLED", "1")
    # Geometry: vertical-ish paths only (avoid sun at (50,50) r=10).
    # Me at y=30; opp at y=90 → vertical distance 60 → min T=10 at MAX_SHIP_SPEED=6.
    # The default dogpile horizon (15) leaves a small headroom for convergence.
    planets = [
        # Me — 4 planets at y=30 (one per column) + 1 off-axis source.
        (0, 0, 10.0, 30.0, 3.0, 200, 3),
        (1, 0, 30.0, 30.0, 3.0, 200, 3),
        (2, 0, 70.0, 30.0, 3.0, 200, 3),
        (3, 0, 90.0, 30.0, 3.0, 200, 3),
        (4, 0, 50.0, 10.0, 3.0, 200, 3),
        # Opp — 4 planets at y=90. Top-3 by prod = {5, 6, 7}; id=8 has prod=1.
        (5, 1, 10.0, 90.0, 3.0, 10, 2),
        (6, 1, 30.0, 90.0, 3.0, 10, 2),
        (7, 1, 70.0, 90.0, 3.0, 10, 2),
        (8, 1, 90.0, 90.0, 3.0, 10, 1),
    ]
    w = _world(planets, step=0)
    m = WorldModel.from_world(w)
    plan = dogpile.evaluate(w, m, me=0, opp_id=1)
    assert plan is not None
    assert isinstance(plan, StrikePlan)
    # Top-3 by production must be the targets.
    assert plan.target_ids == frozenset({5, 6, 7})
    assert len(plan.shots) == 3
    # Each shot must originate from one of my sources and target one of the
    # top-3 opp planets. Margin check: ship_count > opp_garrison * 1.20.
    for shot in plan.shots:
        assert int(shot.src_id) in {0, 1, 2, 3, 4}
        assert int(shot.tgt_id) in {5, 6, 7}
        assert int(shot.ship_count) >= 1


def test_evaluate_disabled_by_env_default():
    """Default env unset → evaluate returns None regardless of state."""
    planets = [
        (0, 0, 15.0, 15.0, 3.0, 200, 3),
        (1, 0, 50.0, 15.0, 3.0, 200, 3),
        (2, 1, 85.0, 85.0, 3.0,  10, 1),
        (3, 1, 30.0, 30.0, 3.0,  10, 1),
        (4, 1, 70.0, 30.0, 3.0,  10, 1),
        (5, 1, 30.0, 70.0, 3.0,  10, 1),
    ]
    w = _world(planets, step=0)
    m = WorldModel.from_world(w)
    assert dogpile.evaluate(w, m, me=0, opp_id=1) is None
