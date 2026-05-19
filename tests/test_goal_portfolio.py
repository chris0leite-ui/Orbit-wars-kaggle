"""Synthetic scenarios for the smallest-sufficient-portfolio identifier (P2)."""

from __future__ import annotations

from lib.goal_planner.portfolio import smallest_winning_portfolio
from lib.goal_planner.predicate import (
    is_winning_state, is_winning_state_if_owned,
)
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet


def _world(planets, step=0, player=0, episode_steps=500):
    obs = _obs(planets=planets, step=step, player=player)
    cfg = {"episodeSteps": episode_steps}
    return World.from_obs(obs, cfg)


def test_portfolio_empty_when_already_winning():
    # Same setup as test_predicate_already_winning — predicate True.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=30, production=1),
        _planet(1, owner=0, x=20.0, y=20.0, ships=30, production=1),
        _planet(2, owner=0, x=30.0, y=30.0, ships=30, production=1),
        _planet(3, owner=1, x=80.0, y=80.0, ships=5, production=1),
    ]
    world = _world(planets, step=480, episode_steps=500)
    assert is_winning_state(world, my_id=0, opp_id=1) is True
    assert smallest_winning_portfolio(world, my_id=0, opp_id=1) == []


def test_portfolio_single_planet_suffices():
    # Two neutrals; flipping ownership of EITHER one flips the predicate.
    # Greedy picks the cheaper one first.
    # Setup: we own 1 planet (prod=1). Opp owns 1 planet (prod=1, ships=10).
    # Neutrals: N1 prod=2 ships=3 (high prod, low cost) — should win.
    #           N2 prod=2 ships=50 (same prod, much higher cost).
    # Without any capture: prod_adv = 1-1 = 0 → False.
    # With N1 captured: prod_adv = 3-1 = 2, opp_pool = 10 + 1*r.
    #   At step=480 r=20: 2*20=40 vs 30 → True.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=20, production=1),
        _planet(1, owner=1, x=90.0, y=90.0, ships=10, production=1),
        _planet(2, owner=-1, x=40.0, y=50.0, ships=3, production=2),
        _planet(3, owner=-1, x=60.0, y=50.0, ships=50, production=2),
    ]
    world = _world(planets, step=480, episode_steps=500)
    assert is_winning_state(world, my_id=0, opp_id=1) is False
    portfolio = smallest_winning_portfolio(world, my_id=0, opp_id=1)
    # N1 is cheaper for the same production → first pick.
    assert portfolio == [2], f"expected [2], got {portfolio}"


def test_portfolio_no_solution():
    # We're too late: only 1 turn left, and opp has a 200-ship fleet
    # in flight that counts in their pool. Even capturing both available
    # neutrals, our prod_advantage (~3) × remaining (1) = 3 << 200.
    # Portfolio identifier must return [].
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=20, production=2),
        _planet(1, owner=1, x=80.0, y=80.0, ships=5, production=1),
        _planet(2, owner=-1, x=40.0, y=10.0, ships=5, production=1),
        _planet(3, owner=-1, x=50.0, y=10.0, ships=5, production=1),
    ]
    # Env raw fleet format: [id, owner, x, y, angle, from_planet_id, ships].
    # See lib/cluster_solver/detector.py:_build_isolated_obs for the
    # canonical reference.
    fleets = [[99, 1, 60.0, 60.0, 0.0, 1, 200]]
    obs = _obs(planets=planets, fleets=fleets, step=499, player=0)
    cfg = {"episodeSteps": 500}
    world = World.from_obs(obs, cfg)

    assert is_winning_state(world, my_id=0, opp_id=1) is False
    # Even acquiring all not-mine planets, predicate stays False
    # (adv*1 < 200 from the in-flight fleet alone).
    all_not_mine = {p.id for p in world.planets if p.owner != 0}
    assert is_winning_state_if_owned(
        world, my_id=0, opp_id=1, extra_planet_ids=all_not_mine,
    ) is False, "precondition: even total acquisition must fail"
    assert smallest_winning_portfolio(world, my_id=0, opp_id=1) == []
