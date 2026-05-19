"""Synthetic scenarios for the defense module (P4)."""

from __future__ import annotations

from lib.goal_planner.defense import defense_actions
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet


def _world(planets, fleets=None, step=0, player=0, episode_steps=500):
    obs = _obs(planets=planets, fleets=fleets or [], step=step, player=player)
    cfg = {"episodeSteps": episode_steps}
    return World.from_obs(obs, cfg)


def test_defense_reinforces_threatened_planet():
    # Mine p0 has 5 ships, prod=1. Mine p1 has 100 ships, close to p0.
    # Opp fleet (30 ships) launched from far away — long ETA gives the
    # reinforcer time to arrive before opp does.
    # Fleets are slower than ship_speed (speed scales 1/sqrt(ships)), so
    # opp fleet 30 ships from (95, 50) → p0 (10, 50): distance 85,
    # speed ~2.7 → ETA ~32. Reinforcer p1→p0 with ~30 ships: distance 10,
    # speed ~2.7 → ETA ~4. 4 < 32 → defense feasible.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=5, production=1),
        _planet(1, owner=0, x=20.0, y=50.0, ships=100, production=1),
        _planet(2, owner=1, x=95.0, y=50.0, ships=10, production=1),
    ]
    # Env raw fleet format: [id, owner, x, y, angle, from_planet_id, ships].
    # Fleet at (95, 50) heading west (angle=pi) toward p0 at (10, 50).
    # 60 ships overwhelms p0's natural garrison-by-arrival (~36 at ETA=31).
    fleets = [[99, 1, 95.0, 50.0, 3.14159, 2, 60]]
    world = _world(planets, fleets=fleets, step=10)

    plan = defense_actions(world, my_id=0, opp_id=1)
    assert plan, f"expected reinforce launch; got {plan}"
    # Defense should target p0 (the threatened mine planet).
    targets = {L.target_id for L in plan}
    assert 0 in targets, f"defense did not target threatened p0: {plan}"
    # Source should be p1 (the only viable reinforcer).
    sources = {L.src_id for L in plan}
    assert 1 in sources, f"defense did not source from p1: {plan}"


def test_defense_no_action_when_no_threat():
    # No opp fleets in flight → no defense needed.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=5, production=1),
        _planet(1, owner=0, x=30.0, y=50.0, ships=100, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),
    ]
    world = _world(planets, fleets=[], step=10)

    plan = defense_actions(world, my_id=0, opp_id=1)
    assert plan == [], f"no threats present; expected empty plan, got {plan}"


def test_defense_priority_order_by_production():
    # Two mine planets close to a central reinforcer; both under threat
    # from FAR-AWAY opp fleets (long ETA → reinforcer can reach either).
    # p0 (prod=3) should be prioritized over p1 (prod=1).
    # Geometry: p0=(30,30), p1=(35,30) — adjacent. p2=(32,30) reinforcer.
    # Opp fleets coming from y=80 and y=-20 — vertical, far away.
    planets = [
        _planet(0, owner=0, x=30.0, y=30.0, ships=5, production=3),
        _planet(1, owner=0, x=35.0, y=30.0, ships=5, production=1),
        _planet(2, owner=0, x=32.0, y=30.0, ships=100, production=1),
        _planet(3, owner=1, x=95.0, y=5.0, ships=10, production=1),
    ]
    # Fleet (20 ships, speed ~3.35) at (30, 90) heading south (-pi/2)
    # toward p0 (30, 30): distance 60 → ETA ~18.
    # Fleet at (35, 90) heading south toward p1: distance 60 → ETA ~18.
    # Each opp fleet 60 ships > natural garrison accrual at ETA ~18
    # (p0 starts at 5, gains 3*18=54 → 59 by arrival; p1 starts at 5,
    # gains 18 → 23 by arrival). Both need reinforcement.
    fleets = [
        [101, 1, 30.0, 90.0, -1.5708, 3, 60],
        [102, 1, 35.0, 90.0, -1.5708, 3, 60],
    ]
    world = _world(planets, fleets=fleets, step=10)

    plan = defense_actions(world, my_id=0, opp_id=1)
    assert plan, f"expected reinforce launches; got {plan}"
    # First scheduled launch should target p0 (higher-priority threat).
    first = plan[0]
    assert first.target_id == 0, (
        f"expected first defense to target high-prod p0; "
        f"plan order was {[L.target_id for L in plan]}"
    )
