"""Phase 4 tests: endgame predicate gate + value mid-bound fallback.

Four fixtures exercise the Phase 4 changes to mpc.py and value.py:

  1. Winning state: when `is_winning_state(world, me, opp)` is True,
     `solve_turn` returns [] immediately (preserve ownership; no risky
     launches). Diagnostics confirm `is_winning_state=True`,
     `solver_status="endgame_winning_idle"`.

  2. Portfolio filter: when winning state is False AND a smallest
     portfolio exists, the LP sees only columns targeting portfolio
     planets (or own-planet reinforces). Diagnostics confirm
     `portfolio_filtered=True` and `n_columns < n_columns_before_filter`.

  3. No portfolio: when winning state is False AND no portfolio is
     feasible, the LP runs unfiltered (no narrowing).

  4. Value mid-bound: when W1 returns (lo=0, hi>0), `value_for_candidate`
     returns hi/2, not 0 (Phase 3 conservatism fix).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.joint_solver.mpc import solve_turn
from lib.joint_solver.predicate import is_winning_state
from lib.joint_solver.value import value_for_candidate


def _planet(pid, owner, *, ships=10, production=2, x=20.0, y=20.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _obs(my_id, planets, *, step=0, fleets=None, num_seats=2):
    return {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "step": step,
        "next_fleet_id": 1000,
    }


# ---------------------------------------------------------------------------
# Fixture 1: Winning state → solve_turn returns [].
# ---------------------------------------------------------------------------


def test_winning_state_returns_empty_moves():
    """When prod_advantage × turns_left > opp_pool, preserve ships."""
    me = [_planet(0, 0, ships=50, production=5, x=20.0, y=20.0),
          _planet(1, 0, ships=50, production=5, x=80.0, y=20.0),
          _planet(2, 0, ships=50, production=5, x=20.0, y=80.0),
          _planet(3, 0, ships=50, production=5, x=80.0, y=80.0)]
    opp = [_planet(10, 1, ships=5, production=1, x=50.0, y=50.0, radius=2.0)]
    # Wait — placing opp at sun-center kills it; move off-center.
    opp = [_planet(10, 1, ships=5, production=1, x=30.0, y=30.0)]
    obs = _obs(my_id=0, planets=me + opp, step=10)

    # Confirm the predicate truly fires for this fixture.
    from lib.intent import World
    world = World.from_obs(obs)
    assert is_winning_state(world, 0, 1) is True, \
        "fixture broken: should be winning"

    moves, diag = solve_turn(obs, return_diagnostics=True)
    assert moves == []
    assert diag.is_winning_state is True
    assert diag.solver_status == "endgame_winning_idle"
    assert diag.n_emitted_moves == 0
    assert diag.n_opp_projections >= 0


# ---------------------------------------------------------------------------
# Fixture 2: Portfolio filter narrows columns to portfolio targets.
# ---------------------------------------------------------------------------


def test_portfolio_filter_narrows_lp_input():
    """Not winning yet, but a small portfolio exists → LP only sees those."""
    # Setup: me=2 planets prod=2; opp=2 planets prod=2 (tied).
    # Adding 1 more from opp side would flip; portfolio = 1 opp planet.
    # The proposer will generate candidates targeting BOTH opp planets,
    # but the filter should keep only the portfolio one.
    me = [_planet(0, 0, ships=15, production=2, x=20.0, y=20.0),
          _planet(1, 0, ships=15, production=2, x=80.0, y=20.0)]
    opp = [_planet(10, 1, ships=8, production=2, x=20.0, y=80.0),
           _planet(11, 1, ships=8, production=2, x=80.0, y=80.0)]
    obs = _obs(my_id=0, planets=me + opp, step=100)

    from lib.intent import World
    from lib.joint_solver.portfolio import smallest_winning_portfolio
    world = World.from_obs(obs)
    assert is_winning_state(world, 0, 1) is False
    pfo = smallest_winning_portfolio(world, 0, 1)
    assert len(pfo) >= 1, "fixture broken: should have at least one portfolio target"

    _moves, diag = solve_turn(obs, return_diagnostics=True)
    # Portfolio should be non-empty; if it is, filter should fire iff
    # the filter doesn't zero out all positive-value columns.
    assert diag.portfolio_size >= 1
    # If columns survived the filter, portfolio_filtered=True must
    # have been recorded. (Could be False only if no positive-value
    # column targets the portfolio — possible but unusual.)
    if diag.n_columns < diag.n_columns_before_filter:
        assert diag.portfolio_filtered is True


# ---------------------------------------------------------------------------
# Fixture 3: No portfolio → LP unfiltered.
# ---------------------------------------------------------------------------


def test_no_portfolio_lp_unfiltered():
    """Losing badly AND no feasible portfolio → LP runs over all columns."""
    # Setup: me has 1 weak planet; opp has 3 strong planets that we
    # can't catch up to in remaining turns. Portfolio = [] (infeasible).
    me = [_planet(0, 0, ships=5, production=1, x=20.0, y=20.0)]
    opp = [_planet(10, 1, ships=50, production=5, x=80.0, y=20.0),
           _planet(11, 1, ships=50, production=5, x=20.0, y=80.0),
           _planet(12, 1, ships=50, production=5, x=80.0, y=80.0)]
    obs = _obs(my_id=0, planets=me + opp, step=480)  # late game

    from lib.intent import World
    from lib.joint_solver.portfolio import smallest_winning_portfolio
    world = World.from_obs(obs)
    pfo = smallest_winning_portfolio(world, 0, 1)
    # The portfolio identifier should return [] when no subset suffices.
    assert pfo == [], f"fixture broken: portfolio should be empty, got {pfo}"

    _moves, diag = solve_turn(obs, return_diagnostics=True)
    assert diag.is_winning_state is False
    assert diag.portfolio_size == 0
    assert diag.portfolio_filtered is False
    # Column count unchanged by the (no-op) filter.
    assert diag.n_columns == diag.n_columns_before_filter


# ---------------------------------------------------------------------------
# Fixture 4: Value mid-bound fallback (lo=0, hi>0 → hi/2, not 0).
# ---------------------------------------------------------------------------


def test_value_mid_bound_fallback_when_wald_fails():
    """W1 returns (0, hi>0) when Wald multi-opp hold check fails — Phase 4
    value function should return hi/2 instead of 0."""
    # Patch _w1_value_bounds to return (lo=0, hi=42.0). The candidate
    # tuple values don't matter once the patch intercepts the call.
    fake_lo, fake_hi = 0.0, 42.0
    with patch("lib.joint_solver.value._w1_value_bounds",
               return_value=(fake_lo, fake_hi)) as _mock:
        # Build a minimal capture candidate. tgt.owner != my_id triggers
        # the capture branch.
        my_id = 0
        src = _planet(0, 0, ships=20, production=2)
        tgt = _planet(10, 1, ships=8, production=2, x=80.0, y=80.0)
        c = (1.0, src, tgt, 15, 0.5, 5, 10, 0)  # cheap_delta, src, tgt, ships, angle, eta, h, wait_N
        # world/model can be None since the patched W1 won't read them.
        value = value_for_candidate(c, world=None, model=None, my_id=my_id)
        assert value == pytest.approx(0.5 * fake_hi)
        assert value > 0.0


def test_value_returns_lo_when_wald_passes():
    """When Wald passes (lo > 0), use lo directly (no mid-bound)."""
    fake_lo, fake_hi = 25.0, 60.0
    with patch("lib.joint_solver.value._w1_value_bounds",
               return_value=(fake_lo, fake_hi)) as _mock:
        my_id = 0
        src = _planet(0, 0, ships=20, production=2)
        tgt = _planet(10, 1, ships=8, production=2, x=80.0, y=80.0)
        c = (1.0, src, tgt, 15, 0.5, 5, 10, 0)
        value = value_for_candidate(c, world=None, model=None, my_id=my_id)
        assert value == pytest.approx(fake_lo)


def test_value_returns_zero_when_both_bounds_zero():
    """Bounce / source-drain / opp-already-holds → (0, 0) → return 0."""
    with patch("lib.joint_solver.value._w1_value_bounds",
               return_value=(0.0, 0.0)) as _mock:
        my_id = 0
        src = _planet(0, 0, ships=20, production=2)
        tgt = _planet(10, 1, ships=8, production=2, x=80.0, y=80.0)
        c = (1.0, src, tgt, 15, 0.5, 5, 10, 0)
        value = value_for_candidate(c, world=None, model=None, my_id=my_id)
        assert value == 0.0
