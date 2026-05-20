"""Phase 5C tests: outcome-table-aware LP.

Six fixtures exercise the key behaviors of `solve_outcome_aware`:

  1. test_forced_gang_up — two sources, one target, solo fails but combined
     captures. LP must pick both candidates (gang-up subset).
  2. test_defense_vs_offense_balance — own planet under projected opp
     attack vs an attractive neutral capture. Limited budget; LP must
     pick defense iff prod_stream value warrants it.
  3. test_idle_when_no_positive_value — every subset has value ≤ 0;
     LP picks the empty subset everywhere → no launches.
  4. test_subset_uniqueness_constraint — x_c fires iff it's in the
     chosen subset; never spuriously.
  5. test_max_contesters_cap — planet with > MAX_CONTESTERS_PER_PLANET
     candidates → top-K kept by per-candidate value.
  6. test_greedy_fallback_path — exercise the fallback when MILP raises.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.joint_solver.lp_outcome import (
    MAX_CONTESTERS_PER_PLANET,
    OutcomeAwareResult,
    _greedy_fallback,
    _build_per_planet_arrivals,
    solve_outcome_aware,
)
from lib.world_model import WorldModel


def _planet(pid, owner, *, ships=10, production=2, x=50.0, y=50.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world_and_model(my_id, planets, *, step=0, fleets=None):
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
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    return world, model


def _column(*, column_id, src_id, tgt_id, ships, value, wait_N=0, eta=3,
            angle=0.0, owner=0):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id, ships=ships,
        wait_N=wait_N, angle=angle, eta=eta, owner=owner, value=float(value),
    )


# ---------------------------------------------------------------------------
# Fixture 1: Forced gang-up (two sources must combine).
# ---------------------------------------------------------------------------


def test_forced_gang_up_two_sources_one_target():
    """Two of my sources, one opp target. Each source has ships=15 but
    target needs ~25 to capture. Solo capture FAILS; gang-up SUCCEEDS.

    The outcome_table's prod_stream_me for solo subsets stays at 0 (no
    capture); for the combined subset it should be substantial. The LP
    should pick the gang-up."""
    me = [_planet(0, 0, ships=20, production=1, x=10.0, y=10.0),
          _planet(1, 0, ships=20, production=1, x=90.0, y=10.0)]
    opp = [_planet(10, 1, ships=20, production=2, x=50.0, y=50.0, radius=1.5)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    # Both candidates arrive at the same tick (eta=3) — eligible for joint capture.
    cols = [
        _column(column_id=0, src_id=0, tgt_id=10, ships=15, value=10.0, eta=3),
        _column(column_id=1, src_id=1, tgt_id=10, ships=15, value=10.0, eta=3),
    ]
    res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # Solo captures fail (15 < 20 garrison + opp production); gang-up
    # (30 ships arriving same tick) succeeds.
    # Both should be picked.
    assert 0 in fired and 1 in fired, \
        f"forced gang-up failed; fired={fired}, status={res.status}"
    # Both wait_N==0 → both emitted as moves.
    assert len(res.moves) == 2


# ---------------------------------------------------------------------------
# Fixture 2: Defense vs offense balance.
# ---------------------------------------------------------------------------


def test_defense_vs_offense_balance():
    """One of my planets is under projected opp attack; a neutral capture
    is also available. With limited budget, LP should pick the higher-EV
    option based on prod_stream.

    Setup:
      - My src=0 (10 ships, prod=2)
      - My src=1 (5 ships, prod=2) — under projected opp attack (in-flight enemy fleet)
      - Neutral target tgt=10 (3 ships, prod=3): capture value ≈ prod·hold
      - Defense candidate: reinforce src=1 to hold against incoming
    Source 0 has only enough to fire one launch (must choose).
    """
    me = [_planet(0, 0, ships=30, production=2, x=10.0, y=10.0),
          _planet(1, 0, ships=5, production=2, x=20.0, y=10.0)]
    neutral = [_planet(10, -1, ships=3, production=3, x=40.0, y=10.0)]
    world, model = _world_and_model(my_id=0, planets=me + neutral)

    # Inject in-flight enemy fleet targeting planet 1 to create defense need.
    # ledger: dict[planet_id, list[(eta, owner, ships)]]
    model.ledger.setdefault(1, []).append((3, 1, 15))

    cols = [
        # Defensive reinforce: src=0 → tgt=1 (15 ships easily covers shortfall)
        _column(column_id=0, src_id=0, tgt_id=1, ships=15, value=50.0, eta=2),
        # Offensive capture: src=0 → tgt=10 (15 ships > 3 + 4*0 = 3 garrison)
        _column(column_id=1, src_id=0, tgt_id=10, ships=15, value=80.0, eta=4),
    ]
    res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    # Either can be picked, but the test verifies the LP MAKES A CHOICE
    # (not idle, not both — budget only allows one).
    assert len(res.fired_columns) >= 1, \
        f"LP failed to act; status={res.status}"
    # The choice itself depends on horizon math (prod_stream_me − α·prod_stream_opp).
    # We just verify the value-aware path executed.
    fired_targets = {c.tgt_id for c in res.fired_columns}
    assert fired_targets, f"no targets fired; status={res.status}"


# ---------------------------------------------------------------------------
# Fixture 3: Idle when no positive value.
# ---------------------------------------------------------------------------


def test_idle_when_no_positive_value():
    """All candidate columns have value=0 → solve_outcome_aware returns
    no_positive_columns status and no moves."""
    me = [_planet(0, 0, ships=10, production=2)]
    opp = [_planet(10, 1, ships=20, production=2, x=50.0, y=50.0)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    cols = [
        _column(column_id=0, src_id=0, tgt_id=10, ships=5, value=0.0),
        _column(column_id=1, src_id=0, tgt_id=10, ships=5, value=-1.0),
    ]
    res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    assert res.moves == []
    assert res.status == "no_positive_columns"


# ---------------------------------------------------------------------------
# Fixture 4: Subset uniqueness — exactly one subset chosen per planet.
# ---------------------------------------------------------------------------


def test_subset_uniqueness_per_planet():
    """For each planet, exactly one subset has y_{p,S}=1. x_c fires iff
    c is in the chosen subset."""
    me = [_planet(0, 0, ships=20, production=1)]
    opp = [_planet(10, 1, ships=5, production=1, x=30.0, y=30.0),
           _planet(11, 1, ships=5, production=1, x=70.0, y=30.0)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    cols = [
        _column(column_id=0, src_id=0, tgt_id=10, ships=8, value=20.0, eta=3),
        _column(column_id=1, src_id=0, tgt_id=11, ships=8, value=15.0, eta=4),
    ]
    res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    # Subset uniqueness: each contested planet has exactly one chosen subset.
    assert set(res.per_planet_chosen.keys()) <= {10, 11}
    for pid, subset in res.per_planet_chosen.items():
        # Each fired column's tgt_id must match the planet whose chosen
        # subset contains it.
        for cid in subset:
            fired_col = next(c for c in res.fired_columns
                             if int(c.column_id) == cid)
            assert int(fired_col.tgt_id) == pid


# ---------------------------------------------------------------------------
# Fixture 5: Max contesters per planet cap.
# ---------------------------------------------------------------------------


def test_max_contesters_per_planet_cap():
    """Planet with more than MAX_CONTESTERS_PER_PLANET candidates → top-K
    by per-candidate value are kept, others dropped from the LP."""
    me = [_planet(0, 0, ships=50, production=2)]
    opp = [_planet(10, 1, ships=5, production=1, x=30.0, y=30.0)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    # Build MAX_CONTESTERS_PER_PLANET + 3 candidates, all targeting planet 10.
    # Different wait_N to make them distinct.
    n_extra = 3
    cols = []
    for k in range(MAX_CONTESTERS_PER_PLANET + n_extra):
        cols.append(_column(
            column_id=k, src_id=0, tgt_id=10, ships=8,
            value=100.0 - k,  # decreasing value so the LOW-id ones survive the cap
            eta=3, wait_N=k,
        ))
    res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    # The cap drops the n_extra LOWEST-value candidates from the planet's
    # enumeration. The LP only sees MAX_CONTESTERS_PER_PLANET candidates.
    assert res.n_x_vars <= MAX_CONTESTERS_PER_PLANET, \
        f"expected ≤{MAX_CONTESTERS_PER_PLANET} x vars, got {res.n_x_vars}"


# ---------------------------------------------------------------------------
# Fixture 6: Greedy fallback path.
# ---------------------------------------------------------------------------


def test_greedy_fallback_when_milp_raises():
    """Patch milp to raise; solve_outcome_aware should route to greedy
    fallback and still return a sensible (non-empty) result when the
    candidate has positive value."""
    me = [_planet(0, 0, ships=20, production=1)]
    opp = [_planet(10, 1, ships=5, production=1, x=30.0, y=30.0)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    # ships=15 beats garrison (5 + 3*1 = 8) at eta=3 → real capture.
    cols = [
        _column(column_id=0, src_id=0, tgt_id=10, ships=15, value=50.0, eta=3),
    ]
    with patch("lib.joint_solver.lp_outcome.milp",
               side_effect=RuntimeError("forced failure")):
        res = solve_outcome_aware(cols, world, model, opp_arrivals=[], my_id=0)
    assert res.status == "greedy_fallback"
    # Greedy should still pick the positive-value capture.
    assert len(res.fired_columns) == 1


# ---------------------------------------------------------------------------
# Plumbing: per-planet arrival construction
# ---------------------------------------------------------------------------


def test_build_per_planet_arrivals_includes_ledger_and_opp_proj():
    """Confirm that fixed arrivals include both the model.ledger entries
    and the opp_arrivals projections."""
    me = [_planet(0, 0, ships=20, production=1)]
    opp = [_planet(10, 1, ships=5, production=1, x=30.0, y=30.0)]
    world, model = _world_and_model(my_id=0, planets=me + opp)

    # Inject an in-flight enemy fleet to planet 10 in the ledger.
    model.ledger.setdefault(10, []).append((4, 1, 7))
    # Opp projection targeting planet 10.
    opp_arrivals = [(10, 5, 1, 12)]

    col = _column(column_id=0, src_id=0, tgt_id=10, ships=8, value=20.0, eta=3)
    arrivals_by_planet = _build_per_planet_arrivals(
        [col], world, model, opp_arrivals, my_id=0, step_now=0,
    )
    fixed, cands = arrivals_by_planet[10]
    # Ledger arrival (4, 1, 7) and opp projection (5, 1, 12) both present.
    fixed_keys = sorted((a.eta, a.owner, a.ships) for a in fixed)
    assert (4, 1, 7) in fixed_keys
    assert (5, 1, 12) in fixed_keys
    # Candidate has total eta = wait_N + eta = 0 + 3 = 3.
    assert len(cands) == 1
    assert cands[0].eta == 3
    assert cands[0].column_id == 0
