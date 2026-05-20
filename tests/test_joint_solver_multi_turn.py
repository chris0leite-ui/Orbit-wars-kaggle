"""Phase 3 tests: multi-turn LP demonstrates coordinated launches.

Three scenarios exercise the multi-turn formulation:

  1. Single-source, multiple wait_N options at same target → LP picks the
     best-value one (not all). Tests source budget over time.

  2. Gang-up: two sources, same target, both fire (wait_N differ) →
     allowed by gang-up cap, both selected if budget permits.

  3. Budget conflict: one source, two competing targets requiring
     more ships than budget → LP picks the higher-value subset.

Plus regression: when MILP is unavailable / infeasible, greedy fallback
produces consistent results.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.joint_solver.lp import (
    DEFAULT_MAX_CONTESTERS_PER_TARGET,
    MultiTurnResult,
    solve_multi_turn,
    _greedy_multi_turn_fallback,
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


def _column(*, column_id, src_id, tgt_id, ships, value, wait_N=0, angle=0.0,
            eta=3, owner=0):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id, ships=ships,
        wait_N=wait_N, angle=angle, eta=eta, owner=owner, value=float(value),
    )


# ---------------------------------------------------------------------------
# Scenario 1: single source, wait_N variants — LP picks the best.
# ---------------------------------------------------------------------------


def test_single_source_wait_variants_picks_best():
    me = [_planet(10, 0, ships=30, production=2)]
    opp = [_planet(20, 1, ships=10, production=1)]
    world = _world(my_id=0, planets=me + opp)

    columns = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, value=20.0, wait_N=0),
        _column(column_id=1, src_id=10, tgt_id=20, ships=18, value=30.0, wait_N=2),
        _column(column_id=2, src_id=10, tgt_id=20, ships=20, value=25.0, wait_N=4),
    ]
    res = solve_multi_turn(columns, world, my_id=0)

    # Each column ALONE fits inside source budget (30 ships at t=0, 34 at t=2,
    # 38 at t=4). Multi-fire from one source is also possible BUT the per-target
    # gang-up cap = 3 (default), and all 3 target the same planet, so LP can
    # pick at most 3.
    # However, source budget at t=0 is 30; if all 3 fire (sum=15+18+20=53)
    # we'd need budget at t=4 (=38) and the cumulative constraint at t=0
    # (Σ wait_N≤0 ships = 15) ≤ 30 OK, at t=2 (= 15+18=33) ≤ 34 OK, at t=4
    # (= 53) ≤ 38 FAIL.
    # So LP picks at most 2; objective max = 30+25=55 or 30+20=50.
    # Optimum: pick (wait_N=2, value=30) and (wait_N=4, value=25): both fit
    # (15+18=33≤34 at t=2; 15+18+20=53>38 at t=4 — NOT both).
    # Optimum is wait_N=2 (value 30) and wait_N=0 (value 20):
    #   t=0: 15 ≤ 30. t=2: 15+18=33 ≤ 34. → fits, obj 50.
    # OR wait_N=2 alone: obj 30. OR wait_N=4 alone: obj 25.
    # Pareto best: wait_N=0 + wait_N=2 with obj 50.
    assert res.status != "empty"
    fired_ids = {c.column_id for c in res.fired_columns}
    assert 1 in fired_ids, f"wait_N=2 (best value) not picked; fired={fired_ids}"
    # Only the wait_N==0 launches emit as moves this turn.
    emitted_wait_zero = [c for c in res.fired_columns if c.wait_N == 0]
    assert len(res.moves) == len(emitted_wait_zero)


# ---------------------------------------------------------------------------
# Scenario 2: gang-up — two sources, same target, both fire.
# ---------------------------------------------------------------------------


def test_two_sources_gang_up_same_target():
    me = [_planet(10, 0, ships=15, production=1),
          _planet(11, 0, ships=15, production=1)]
    opp = [_planet(20, 1, ships=20, production=0)]
    world = _world(my_id=0, planets=me + opp)

    columns = [
        # Both sources, same target, same arrival tick (eta=3); each
        # alone insufficient to capture (15 < 20+0*3 + 1=21), but
        # together (15+15=30 > 21) suffice.
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, value=10.0, eta=3),
        _column(column_id=1, src_id=11, tgt_id=20, ships=15, value=10.0, eta=3),
    ]
    res = solve_multi_turn(columns, world, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # Both should fire (positive value, source budgets independent, gang-up
    # cap >= 2).
    assert 0 in fired and 1 in fired, f"gang-up failed; fired={fired}"
    assert len(res.moves) == 2


# ---------------------------------------------------------------------------
# Scenario 3: budget conflict — one source, two competing targets.
# ---------------------------------------------------------------------------


def test_budget_conflict_picks_higher_value():
    me = [_planet(10, 0, ships=10, production=0)]  # NO production — fixed budget
    opp = [_planet(20, 1, ships=8, production=0),
           _planet(21, 1, ships=8, production=0)]
    world = _world(my_id=0, planets=me + opp)

    # Both columns each need 10 ships; source only has 10 → can only fire ONE.
    columns = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=10, value=50.0),
        _column(column_id=1, src_id=10, tgt_id=21, ships=10, value=30.0),
    ]
    res = solve_multi_turn(columns, world, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # Higher-value target wins.
    assert fired == {0}, f"budget conflict: should pick column 0 only; got {fired}"


# ---------------------------------------------------------------------------
# Scenario 4: wait_N>0 columns are KEPT in solution (Phase 3 vs Phase 2).
# ---------------------------------------------------------------------------


def test_wait_N_positive_kept_in_solution():
    """Phase 2's Slice 10 LP filtered wait_N>0 columns at value=0. Phase 3
    multi-turn LP must keep them if their value is positive — this is the
    core capability change."""
    me = [_planet(10, 0, ships=5, production=10)]  # high prod, tiny initial
    opp = [_planet(20, 1, ships=15, production=0)]
    world = _world(my_id=0, planets=me + opp)

    columns = [
        # Fire-now: not enough ships (5 vs 15 garrison) → low value.
        _column(column_id=0, src_id=10, tgt_id=20, ships=5, value=2.0, wait_N=0),
        # Wait 2 turns: 5 + 2*10 = 25 ships; capture viable. Higher value.
        _column(column_id=1, src_id=10, tgt_id=20, ships=20, value=40.0, wait_N=2),
    ]
    res = solve_multi_turn(columns, world, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # LP must include the wait_N=2 column.
    assert 1 in fired, f"wait_N=2 column dropped; fired={fired}"
    # Both can fit (t=0: 5 ≤ 5; t=2: 5+20=25 ≤ 5+20=25), but the LP picks
    # what maximizes value. wait_N=0 only adds value 2, so picking both
    # gives 42 vs wait_N=2 alone = 40. Either is acceptable; assert ≥40.
    assert res.objective >= 40.0 - 1e-6
    # If wait_N=0 not in fired set, no immediate move emitted (consistent
    # with MPC: only commit wait_N==0 columns).
    if 0 not in fired:
        assert res.moves == []


# ---------------------------------------------------------------------------
# Scenario 5: greedy fallback consistency.
# ---------------------------------------------------------------------------


def test_greedy_fallback_consistent_with_milp():
    """When run on the same scenario, MILP and greedy should agree on the
    SET of fired columns for unambiguous problems."""
    me = [_planet(10, 0, ships=20, production=0),
          _planet(11, 0, ships=20, production=0)]
    opp = [_planet(20, 1, ships=10, production=0),
           _planet(21, 1, ships=10, production=0)]
    world = _world(my_id=0, planets=me + opp)

    columns = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, value=50.0),
        _column(column_id=1, src_id=11, tgt_id=21, ships=15, value=40.0),
        _column(column_id=2, src_id=10, tgt_id=21, ships=15, value=10.0),  # worse
        _column(column_id=3, src_id=11, tgt_id=20, ships=15, value=10.0),  # worse
    ]
    res_milp = solve_multi_turn(columns, world, my_id=0)
    res_greedy = _greedy_multi_turn_fallback(
        columns, world, my_id=0,
        max_contesters_per_target=DEFAULT_MAX_CONTESTERS_PER_TARGET,
    )
    # Both should pick columns 0 and 1 (best non-conflicting pair).
    fired_milp = {c.column_id for c in res_milp.fired_columns}
    fired_greedy = {c.column_id for c in res_greedy.fired_columns}
    assert fired_milp == {0, 1}, f"milp picked {fired_milp}"
    assert fired_greedy == {0, 1}, f"greedy picked {fired_greedy}"


# ---------------------------------------------------------------------------
# Scenario 6: empty / no-positive columns.
# ---------------------------------------------------------------------------


def test_empty_columns_returns_no_moves():
    world = _world(my_id=0, planets=[_planet(10, 0, ships=10)])
    res = solve_multi_turn([], world, my_id=0)
    assert res.moves == []
    assert res.status == "empty"


def test_zero_value_columns_dropped():
    me = [_planet(10, 0, ships=10)]
    opp = [_planet(20, 1, ships=5)]
    world = _world(my_id=0, planets=me + opp)
    columns = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=6, value=0.0),
        _column(column_id=1, src_id=10, tgt_id=20, ships=6, value=-1.0),
    ]
    res = solve_multi_turn(columns, world, my_id=0)
    assert res.moves == []
    assert res.status == "no_positive_columns"
