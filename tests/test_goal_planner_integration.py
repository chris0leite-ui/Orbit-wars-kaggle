"""End-to-end integration tests for the goal_planner agent."""

from __future__ import annotations

from agents.goal_planner.main import agent as goal_agent
from tests.scenarios.base import _obs, _planet


def test_integration_already_winning_emits_defense_or_nothing():
    # We're dominant; predicate True; portfolio empty → no acquisition.
    # No incoming threats → no defense.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=30, production=1),
        _planet(1, owner=0, x=20.0, y=20.0, ships=30, production=1),
        _planet(2, owner=0, x=30.0, y=30.0, ships=30, production=1),
        _planet(3, owner=1, x=80.0, y=80.0, ships=5, production=1),
    ]
    obs = _obs(planets=planets, step=480, player=0)
    cfg = {"episodeSteps": 500}
    emits = goal_agent(obs, cfg)
    assert emits == [], f"already-winning + no threat: expected idle; got {emits}"


def test_integration_free_capture_launches():
    # Behind on production; portfolio = [neutral planet]; cheap capture
    # available from p0. Agent must emit one launch.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=1),
        _planet(1, owner=-1, x=16.0, y=50.0, ships=5, production=2),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=2),
    ]
    obs = _obs(planets=planets, step=10, player=0)
    cfg = {"episodeSteps": 500}
    emits = goal_agent(obs, cfg)
    assert emits, f"behind + free capture available: expected launch; got {emits}"
    # The launch should source from p0 toward the neutral target.
    assert any(int(e[0]) == 0 for e in emits), (
        f"expected emit from p0; got {emits}"
    )


def test_integration_defense_when_under_threat():
    # Mine p0 (5 ships) under threat from long-range opp fleet (60 ships).
    # Mine p1 (100 ships) is the only reinforcer. Agent must emit a
    # defense launch from p1 toward p0.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=5, production=1),
        _planet(1, owner=0, x=20.0, y=50.0, ships=100, production=1),
        _planet(2, owner=1, x=95.0, y=50.0, ships=10, production=1),
    ]
    fleets = [[99, 1, 95.0, 50.0, 3.14159, 2, 60]]
    obs = _obs(planets=planets, fleets=fleets, step=10, player=0)
    cfg = {"episodeSteps": 500}
    emits = goal_agent(obs, cfg)
    assert emits, f"under threat: expected reinforce; got {emits}"
    # Should emit a defense launch from p1 (the reinforcer).
    sources = {int(e[0]) for e in emits}
    assert 1 in sources, f"expected emit from p1; got {emits}"


def test_integration_no_double_allocation_across_defense_and_acquisition():
    # Single source p0 has 50 ships. Defense needs 30 ships to repel
    # threat at p1. Acquisition wants the neutral p2 (3 ships). After
    # defense, p0 has 20 ships left → can still afford the 8-ship
    # acquisition of p2.
    # Total ships across emits from p0 must NOT exceed 50.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=50, production=1),
        _planet(1, owner=0, x=20.0, y=50.0, ships=5, production=1),
        _planet(2, owner=-1, x=18.0, y=50.0, ships=3, production=2),
        _planet(3, owner=1, x=95.0, y=50.0, ships=10, production=1),
    ]
    # Threat at p1: long-range opp fleet (30 ships).
    fleets = [[99, 1, 95.0, 50.0, 3.14159, 3, 30]]
    obs = _obs(planets=planets, fleets=fleets, step=10, player=0)
    cfg = {"episodeSteps": 500}
    emits = goal_agent(obs, cfg)
    # Sum of ships from p0 must be <= 50.
    p0_total = sum(int(e[2]) for e in emits if int(e[0]) == 0)
    assert p0_total <= 50, f"p0 over-allocated: 50 budget, used {p0_total}; emits={emits}"
