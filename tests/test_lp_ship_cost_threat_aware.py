"""Phase 4 Step 2 (lighthouse plan): source-aware ship cost in the LP
objective.

The LP at `lib/joint_solver/lp_outcome.py` previously priced ships
uniformly at `SHIP_COST * col.ships`. A ship sent from a planet under
inbound enemy threat carries DEFENSIVE VALUE at the source — its
removal costs more than 1 unit per ship. The fix multiplies the per-
column cost by `SHIP_COST_THREAT_MULT` (default 2.0) when the source
has any enemy threat (`WorldModel.time_to_enemy_threat(src) is not None`).

Per Rule 40, this prices something the LP currently fails to see —
the defensive value of ships at threatened sources — rather than
adding a cap or threshold.

Pin tests (Rule 38). Tests 1-2 exercise `_ship_cost` in isolation;
test 3 is the integration cycle — pre-fix (`SHIP_COST_THREAT_MULT=1.0`,
no-op) the LP picks the threatened source; post-fix (`=2.0`) the LP
prefers the rear source.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.joint_solver.lp_outcome import (
    SHIP_COST,
    SHIP_COST_THREAT_ETA_THRESHOLD,
    SHIP_COST_THREAT_MULT,
    _ship_cost,
    solve_outcome_aware,
)
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


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


def _model_with_threat_at(src_id, *, planet_ids, threat_eta):
    """Build a WorldModel whose `time_to_enemy_threat(src_id, ...)`
    returns `threat_eta`. Done via the ledger: stuff an in-flight enemy
    fleet at the source so `incoming_enemy_eta` resolves to threat_eta.
    """
    ledger = {pid: [] for pid in planet_ids}
    # (eta, owner=enemy, ships) → counts as inbound enemy fleet.
    ledger[src_id] = [(int(threat_eta), 1, 50)]
    return WorldModel(ledger=ledger, timelines={}, horizon=200)


# ---------------------------------------------------------------------------
# Test 1 — threatened source costs more than a rear source.
# ---------------------------------------------------------------------------

def test_ship_cost_threatened_source_costs_more():
    """`_ship_cost` returns `SHIP_COST * ships` for a rear source (no
    threat within threshold) and `SHIP_COST * SHIP_COST_THREAT_MULT *
    ships` for a threatened source (inbound fleet OR close opp planet
    within `SHIP_COST_THREAT_ETA_THRESHOLD`).
    """
    # P0 mine at (0, 50). P1 mine at (10, 50). No opp planet so no
    # geographical threats — isolates the test on the in-flight branch.
    me = [_planet(0, 0, ships=100, x=0.0, y=50.0),
          _planet(1, 0, ships=100, x=10.0, y=50.0)]
    world = _world(my_id=0, planets=me)
    # P0 has an inbound enemy fleet at eta=5 (well below threshold);
    # P1 has no inbound and no opp planets → rear.
    model = _model_with_threat_at(0, planet_ids=[0, 1], threat_eta=5)

    col_threatened = Column(column_id=100, src_id=0, tgt_id=0,
                            ships=20, wait_N=0, angle=0.0, eta=10,
                            owner=0, value=100.0)
    col_rear = Column(column_id=101, src_id=1, tgt_id=0,
                      ships=20, wait_N=0, angle=0.0, eta=10,
                      owner=0, value=100.0)

    cost_threatened = _ship_cost(col_threatened, world, model, my_id=0)
    cost_rear = _ship_cost(col_rear, world, model, my_id=0)

    assert cost_threatened == pytest.approx(
        SHIP_COST * SHIP_COST_THREAT_MULT * 20
    )
    assert cost_rear == pytest.approx(SHIP_COST * 20)
    assert cost_threatened > cost_rear, (
        f"threatened source ({cost_threatened}) should cost more than "
        f"rear source ({cost_rear})"
    )


# ---------------------------------------------------------------------------
# Test 1b — indirect threat: close opp planet (potential launch) fires the
# multiplier; far opp planet (eta > threshold) doesn't.
#
# PI 2026-05-21: "also indirect fleet over close opponent planets" — the
# multiplier must fire on geographical proximity to opp, not only on
# in-flight fleets.
# ---------------------------------------------------------------------------

def test_ship_cost_close_opp_planet_fires_multiplier():
    """A source with a CLOSE opp planet (potential launch within threshold)
    pays the multiplier. A source with only a FAR opp planet does not.
    """
    # P0 mine at (0, 50): NO opp planet anywhere close — rear.
    # P1 mine at (50, 50): an opp planet sits at (55, 50), distance 5
    #   from P1 → potential launch eta ≈ 5 (fleet_speed(1)=1) < threshold 30.
    # No inbound fleets — purely geographical.
    me = [_planet(0, 0, ships=100, x=0.0, y=50.0),
          _planet(1, 0, ships=100, x=50.0, y=50.0)]
    opp = [_planet(2, 1, ships=1, production=1, x=55.0, y=50.0)]
    world = _world(my_id=0, planets=me + opp)
    model = WorldModel(
        ledger={0: [], 1: [], 2: []}, timelines={}, horizon=200,
    )

    col_far = Column(column_id=110, src_id=0, tgt_id=1,
                     ships=10, wait_N=0, angle=0.0, eta=5,
                     owner=0, value=100.0)
    col_near = Column(column_id=111, src_id=1, tgt_id=0,
                      ships=10, wait_N=0, angle=0.0, eta=5,
                      owner=0, value=100.0)

    # Sanity: P0 should be far enough from P2 that its threat eta > threshold.
    # P0 is at (0,50), P2 at (55,50), fleet_speed(1)=1.0 → threat eta=55 > 30.
    threat_p0 = model.time_to_enemy_threat(0, 0, world)
    threat_p1 = model.time_to_enemy_threat(1, 0, world)
    assert threat_p0 is not None and threat_p0 > SHIP_COST_THREAT_ETA_THRESHOLD, (
        f"fixture: P0 should be far from opp (threat eta > "
        f"{SHIP_COST_THREAT_ETA_THRESHOLD}); got {threat_p0}"
    )
    assert threat_p1 is not None and threat_p1 <= SHIP_COST_THREAT_ETA_THRESHOLD, (
        f"fixture: P1 should be close to opp (threat eta <= "
        f"{SHIP_COST_THREAT_ETA_THRESHOLD}); got {threat_p1}"
    )

    cost_far = _ship_cost(col_far, world, model, my_id=0)
    cost_near = _ship_cost(col_near, world, model, my_id=0)

    assert cost_far == pytest.approx(SHIP_COST * 10), (
        f"P0 (far from opp) should pay base cost; got {cost_far}"
    )
    assert cost_near == pytest.approx(
        SHIP_COST * SHIP_COST_THREAT_MULT * 10
    ), (
        f"P1 (close to opp) should pay {SHIP_COST_THREAT_MULT}x base; "
        f"got {cost_near}"
    )


# ---------------------------------------------------------------------------
# Test 2 — no threat anywhere: rear pricing applies to every column.
# ---------------------------------------------------------------------------

def test_ship_cost_no_threat_no_multiplier():
    """On a board with no enemy threat, `_ship_cost` returns the uniform
    `SHIP_COST * col.ships`. Locks the no-op contract on peaceful boards.
    """
    me = [_planet(0, 0, ships=100), _planet(1, 0, ships=100, x=30.0)]
    neutral = [_planet(2, -1, ships=5, x=60.0)]
    world = _world(my_id=0, planets=me + neutral)
    model = WorldModel(
        ledger={0: [], 1: [], 2: []}, timelines={}, horizon=200,
    )

    col_a = Column(column_id=200, src_id=0, tgt_id=2,
                   ships=15, wait_N=0, angle=0.0, eta=5,
                   owner=0, value=100.0)
    col_b = Column(column_id=201, src_id=1, tgt_id=2,
                   ships=15, wait_N=0, angle=0.0, eta=5,
                   owner=0, value=100.0)

    cost_a = _ship_cost(col_a, world, model, my_id=0)
    cost_b = _ship_cost(col_b, world, model, my_id=0)

    assert cost_a == pytest.approx(SHIP_COST * 15)
    assert cost_b == pytest.approx(SHIP_COST * 15)


# ---------------------------------------------------------------------------
# Test 3 — integration: LP prefers rear source when threatened-source cost
#          tips the balance.
#
# Rule 38 cycle: temporarily setting `SHIP_COST_THREAT_MULT = 1.0` (no-op)
# the LP is indifferent between sources (tie-broken by id); with
# `=2.0` the LP strictly prefers the rear source.
# ---------------------------------------------------------------------------

def test_solve_outcome_aware_prefers_rear_source_over_threatened():
    """Engineered scenario:

    - P0 (mine, ships=20): THREATENED — inbound enemy fleet at eta=5.
    - P1 (mine, ships=20): rear, no threat.
    - P2 (opp, ships=5): single target. Both sources can reach.
    - One column from P0, one from P1, identical otherwise.

    With ship_cost UNIFORM: both columns have equal cost, LP fires
    either (deterministic tie-break by id ⇒ column 300 from P0 fires).

    With ship_cost source-aware: column 300 (from threatened P0) costs
    `2.0 * 6 = 12` while column 301 (from rear P1) costs `1.0 * 6 = 6`.
    LP fires column 301.
    """
    me = [_planet(0, 0, ships=20),
          _planet(1, 0, ships=20, x=30.0)]
    # P2 small + cheap so 6 ships at eta=3 reliably captures it
    # (garrison at eta=3 = 1 + 1*3 = 4; 6 > 4).
    opp = [_planet(2, 1, ships=1, production=1, x=60.0)]
    world = _world(my_id=0, planets=me + opp)
    # Threat on P0: ledger has an inbound enemy fleet eta=5 with 50 ships.
    model = WorldModel(
        ledger={0: [(5, 1, 50)], 1: [], 2: []},
        timelines={}, horizon=200,
    )

    col_threatened_src = Column(
        column_id=300, src_id=0, tgt_id=2,
        ships=6, wait_N=0, angle=0.0, eta=3,
        owner=0, value=100.0,
    )
    col_rear_src = Column(
        column_id=301, src_id=1, tgt_id=2,
        ships=6, wait_N=0, angle=0.0, eta=3,
        owner=0, value=100.0,
    )

    res = solve_outcome_aware(
        [col_threatened_src, col_rear_src], world, model,
        my_id=0,
        t_end=500,
        alpha_opp_penalty=1.0,
        ship_cost=1.0,
        time_limit_seconds=5.0,
    )

    fired_ids = {int(c.column_id) for c in res.fired_columns}
    # The LP fires at most one of the two (both share a single target).
    # With source-aware cost, it must prefer the rear column (301).
    assert 301 in fired_ids and 300 not in fired_ids, (
        f"LP should prefer rear source (col 301) over threatened "
        f"source (col 300). Got fired={sorted(fired_ids)}, "
        f"status={res.status}, objective={res.objective}, "
        f"chosen={res.per_planet_chosen}"
    )
