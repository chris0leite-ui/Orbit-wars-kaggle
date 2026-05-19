"""Synthetic scenarios for the winning-state predicate (P1)."""

from __future__ import annotations

from lib.goal_planner.predicate import (
    is_winning_state, opp_pool, prod_advantage, remaining_turns,
)
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet


def _world(planets, step=0, player=0, episode_steps=500):
    obs = _obs(planets=planets, step=step, player=player)
    cfg = {"episodeSteps": episode_steps}
    return World.from_obs(obs, cfg)


def test_predicate_already_winning():
    # 4 planets, all prod=1. We own 3, opp owns 1 with 5 ships. 20 turns left.
    # prod_adv = 3 - 1 = 2
    # opp_pool = 5 + 1*20 = 25
    # 2 * 20 = 40 > 25 → True.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=30, production=1),
        _planet(1, owner=0, x=20.0, y=20.0, ships=30, production=1),
        _planet(2, owner=0, x=30.0, y=30.0, ships=30, production=1),
        _planet(3, owner=1, x=80.0, y=80.0, ships=5, production=1),
    ]
    world = _world(planets, step=480, episode_steps=500)
    assert remaining_turns(world) == 20
    assert prod_advantage(world, my_id=0, opp_id=1) == 2
    assert opp_pool(world, opp_id=1) == 5 + 1 * 20
    assert is_winning_state(world, my_id=0, opp_id=1) is True


def test_predicate_clearly_losing():
    # Mirror: we own 1, opp owns 3.
    # prod_adv = 1 - 3 = -2 → trivial False (prod_advantage <= 0).
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=30, production=1),
        _planet(1, owner=1, x=20.0, y=20.0, ships=30, production=1),
        _planet(2, owner=1, x=30.0, y=30.0, ships=30, production=1),
        _planet(3, owner=1, x=80.0, y=80.0, ships=30, production=1),
    ]
    world = _world(planets, step=480, episode_steps=500)
    assert prod_advantage(world, my_id=0, opp_id=1) == -2
    assert is_winning_state(world, my_id=0, opp_id=1) is False


def test_predicate_breakeven_threshold():
    # Fixed prod_advantage=1, sweep remaining_turns. Threshold equation:
    #   adv * remaining > opp_pool
    #   1 * remaining > 10 + 1 * remaining   →   0 > 10. Never true if opp_prod==adv.
    # So make opp_prod=0 (only their ships matter as a constant pool).
    # Setup: 2 planets, both prod=1, both mine ⇒ prod_adv = 2-0 = 2.
    # Opp has 1 planet with prod=0, ships=10.
    # Then: adv=2, opp_pool = 10 + 0*r = 10. Threshold: 2*r > 10 → r > 5.
    # Flip point: r=6 → True, r=5 → False.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=10, production=1),
        _planet(1, owner=0, x=20.0, y=20.0, ships=10, production=1),
        _planet(2, owner=1, x=80.0, y=80.0, ships=10, production=0),
    ]
    for r in range(0, 10):
        world = _world(planets, step=500 - r, episode_steps=500)
        expected = (2 * r > 10)
        actual = is_winning_state(world, my_id=0, opp_id=1)
        assert actual == expected, (
            f"r={r}: expected {expected}, got {actual} "
            f"(adv=2, opp_pool=10, 2*r={2*r})"
        )
