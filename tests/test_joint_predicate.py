"""Unit tests for lib/joint_solver/predicate.

3 hand-built fixtures exercise the closed-form winning-state predicate
across the regimes that matter:
  1. Clearly winning  (large prod advantage, small opp pool).
  2. Clearly losing   (no prod advantage).
  3. Boundary         (just-barely winning, then opp planet capture flips it).

Plus is_winning_state_if_owned re-attribution sanity.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.predicate import (
    EPISODE_STEPS,
    is_winning_state,
    is_winning_state_if_owned,
    opp_pool,
    prod_advantage,
    remaining_turns,
)


def _planet(pid, owner, *, ships=10, production=2, x=0.0, y=0.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, fleets=None):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Fixture 1: clearly winning — me 4 planets prod=3 each; opp 1 planet prod=1.
# ---------------------------------------------------------------------------

def test_clearly_winning_state():
    me = [_planet(i, 0, production=3, ships=20) for i in range(4)]
    opp = [_planet(10, 1, production=1, ships=5)]
    world = _world(my_id=0, planets=me + opp, step=10)

    # prod_advantage = 4*3 − 1 = 11. remaining_turns = 490.
    # opp_pool = 5 + 0 + 1*490 = 495. predicate: 11*490=5390 > 495. TRUE.
    assert prod_advantage(world, 0, 1) == 11
    assert remaining_turns(world) == EPISODE_STEPS - 10
    assert opp_pool(world, 1) == 5 + 1 * (EPISODE_STEPS - 10)
    assert is_winning_state(world, 0, 1) is True


# ---------------------------------------------------------------------------
# Fixture 2: clearly losing — me 1 planet prod=1; opp 3 planets prod=2 each.
# ---------------------------------------------------------------------------

def test_clearly_losing_state():
    me = [_planet(0, 0, production=1, ships=10)]
    opp = [_planet(i, 1, production=2, ships=15) for i in range(1, 4)]
    world = _world(my_id=0, planets=me + opp, step=0)

    # prod_advantage = 1 − 6 = −5  ⇒  early-exit False.
    assert prod_advantage(world, 0, 1) == -5
    assert is_winning_state(world, 0, 1) is False


# ---------------------------------------------------------------------------
# Fixture 3: boundary — predicate False today, flips True if I capture P10.
# ---------------------------------------------------------------------------

def test_boundary_flip_via_portfolio_acquisition():
    # Tight setup: 2 mine prod=2; 2 opp prod=2 — currently equal, no edge.
    # If I take 1 opp planet, prod_advantage becomes (3*2 − 1*2) = 4 (re-attributed).
    me = [_planet(0, 0, production=2, ships=8),
          _planet(1, 0, production=2, ships=8)]
    opp = [_planet(2, 1, production=2, ships=8, x=10.0),
           _planet(3, 1, production=2, ships=8, x=20.0)]
    world = _world(my_id=0, planets=me + opp, step=400, fleets=[])

    # prod_advantage = 4 − 4 = 0 → False.
    assert prod_advantage(world, 0, 1) == 0
    assert is_winning_state(world, 0, 1) is False
    # In-flight fleets are counted in opp_pool.
    fleets_obs = [[100, 1, 5.0, 5.0, 0.0, -1, 4]]  # opp fleet, 4 ships
    world_with_flight = _world(my_id=0, planets=me + opp, step=400,
                               fleets=fleets_obs)
    assert opp_pool(world_with_flight, 1) - opp_pool(world, 1) == 4

    # Hypothetical: if I owned P2, prod_advantage = 2*3 − 1*2 = 4.
    # remaining = 100. opp_pool reduces by (8 + 2*100) = 208.
    # New opp_pool = (8 + 2*100) − 208 = 0. 4*100 = 400 > 0. TRUE.
    assert is_winning_state_if_owned(world, 0, 1, {2}) is True


# ---------------------------------------------------------------------------
# Extra: prod_advantage_if_owned re-attribution from neutral vs opp.
# ---------------------------------------------------------------------------

def test_if_owned_re_attribution_neutral_vs_opp():
    me = [_planet(0, 0, production=2, ships=10)]
    neut = [_planet(1, -1, production=3, ships=5)]
    opp = [_planet(2, 1, production=3, ships=10)]
    world = _world(my_id=0, planets=me + neut + opp, step=0)

    # Acquiring neutral P1: prod_advantage gains +3 (no opp loss).
    # Acquiring opp P2:    prod_advantage gains +6 (us +3, opp −3 → diff +6).
    base_adv = prod_advantage(world, 0, 1)
    # We can't read adjusted adv directly, but we can use the boundary check.
    # Use a sufficiently small horizon to make the predicate sensitive.
    # remaining_turns(step=0) = 500. opp_pool = 10 + 3*500 = 1510. adv = -1.
    # Predicate False today.
    assert base_adv == -1
    assert is_winning_state(world, 0, 1) is False
    # Acquiring just neutral: adv = -1 + 3 = 2. opp_pool unchanged 1510.
    # 2 * 500 = 1000 < 1510 → still False.
    assert is_winning_state_if_owned(world, 0, 1, {1}) is False
    # Acquiring just opp P2: adv = -1 + 6 = 5. opp_pool − (10 + 3*500) = 0.
    # 5 * 500 = 2500 > 0 → True.
    assert is_winning_state_if_owned(world, 0, 1, {2}) is True
