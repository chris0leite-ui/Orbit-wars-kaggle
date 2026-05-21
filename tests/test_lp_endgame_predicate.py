"""Phase 4 (lighthouse plan): endgame predicate term in LP objective.

The LP at `lib/joint_solver/lp_outcome.py::_value_for_outcome` priced
only `me_prod − α · opp_prod`. The fourth term in the lighthouse
objective formulation — `λ · I[is_winning_state(post_world)]` — was
never wired. Phase 4 adds it as a per-(planet, subset) bonus computed
by `_endgame_bonus`.

Pin tests (Rule 38). Tests 1-5 exercise the helper in isolation;
test 6 is the integration cycle: pre-fix (`LAMBDA_ENDGAME=0`) the LP
picks the wrong subset because raw production-stream value prefers
a non-tipping capture; post-fix (`LAMBDA_ENDGAME=1000`) the predicate
bonus dominates and the LP picks the tipping capture.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.joint_solver.outcome_table import OutcomeRow
from lib.joint_solver.lp_outcome import (
    LAMBDA_ENDGAME,
    _endgame_bonus,
    solve_outcome_aware,
)
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _planet(pid, owner, *, ships=10, production=2, x=0.0, y=0.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world_from_planets(my_id, planets, *, step=0, fleets=None):
    """Build a World via `World.from_obs` to match the predicate's
    code path (it reads `world.obs_raw["fleets"]` and `world.planets_by_id`).
    """
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


def _row(*, subset, owner_T, me_prod=0, opp_prod=0):
    """Construct a minimal OutcomeRow for helper-level tests.

    `me_prod` / `opp_prod` populate `prod_stream` for owners 0 / 1.
    """
    stream = {}
    if me_prod:
        stream[0] = int(me_prod)
    if opp_prod:
        stream[1] = int(opp_prod)
    return OutcomeRow(
        subset=tuple(subset),
        owner_T=int(owner_T),
        ships_T=0.0,
        prod_stream=stream,
        prod_stream_discounted={},
    )


# ---------------------------------------------------------------------------
# Test 1 — capture that tips the predicate from False→True awards +λ.
# ---------------------------------------------------------------------------

def test_endgame_bonus_capture_that_tips():
    """World near tipping point: capturing opp's planet flips
    `is_winning_state` from False to True. The bonus fires.
    """
    me = [_planet(0, 0, production=1, ships=611)]
    opp = [_planet(1, 1, production=2, ships=600, x=20.0)]
    world = _world_from_planets(my_id=0, planets=me + opp, step=0)

    # row says "after this subset, I own planet 1 at horizon T".
    row = _row(subset=(100,), owner_T=0, me_prod=990, opp_prod=10)
    bonus = _endgame_bonus(planet_id=1, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=False)
    assert bonus == pytest.approx(LAMBDA_ENDGAME), (
        f"capture of opp planet 1 should tip is_winning_state and award "
        f"+{LAMBDA_ENDGAME}; got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 2 — capture that doesn't tip the predicate awards 0.
# ---------------------------------------------------------------------------

def test_endgame_bonus_capture_that_does_not_tip():
    """Same world as Test 1; capturing a neutral planet with low
    production doesn't tip the predicate.
    """
    me = [_planet(0, 0, production=1, ships=611)]
    opp = [_planet(1, 1, production=2, ships=600, x=20.0)]
    neutral = [_planet(2, -1, production=1, ships=5, x=40.0)]
    world = _world_from_planets(my_id=0, planets=me + opp + neutral, step=0)

    row = _row(subset=(101,), owner_T=0, me_prod=495, opp_prod=0)
    bonus = _endgame_bonus(planet_id=2, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=False)
    assert bonus == 0.0, (
        f"capture of neutral planet 2 (prod=1) shouldn't tip the predicate; "
        f"got bonus={bonus}"
    )


# ---------------------------------------------------------------------------
# Test 3 — retaining own planet in winning state awards +λ.
# ---------------------------------------------------------------------------

def test_endgame_bonus_retain_own_planet_in_winning_state():
    """World currently winning; subset keeps own planet owner_T==me.
    Each retained own planet contributes +λ — the LP-positive valuation
    of staying in winning state.
    """
    me = [_planet(0, 0, production=5, ships=20),
          _planet(1, 0, production=5, ships=20)]
    opp = [_planet(10, 1, production=1, ships=5, x=30.0)]
    world = _world_from_planets(my_id=0, planets=me + opp, step=10)

    row = _row(subset=(), owner_T=0, me_prod=2475, opp_prod=0)
    bonus = _endgame_bonus(planet_id=0, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=True)
    assert bonus == pytest.approx(LAMBDA_ENDGAME), (
        f"retaining own planet 0 in winning state should award "
        f"+{LAMBDA_ENDGAME}; got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 4 — losing own planet that flips us out of winning awards −λ.
# ---------------------------------------------------------------------------

def test_endgame_bonus_lose_own_planet_that_flips_out():
    """Boundary world (just-barely winning); losing a key own planet
    flips us out. The bonus is the symmetric penalty.
    """
    me = [_planet(0, 0, production=2, ships=8),
          _planet(1, 0, production=2, ships=8),
          _planet(2, 0, production=2, ships=8)]
    opp = [_planet(3, 1, production=2, ships=8, x=20.0)]
    world = _world_from_planets(my_id=0, planets=me + opp, step=400)
    # is_winning_state at step 400: prod_adv = 4, rem = 100, opp_pool = 208.
    # 4*100 = 400 > 208 → True. Losing P0 → adv = -2 → False (flips out).

    row = _row(subset=(200,), owner_T=1, me_prod=0, opp_prod=200)
    bonus = _endgame_bonus(planet_id=0, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=True)
    assert bonus == pytest.approx(-LAMBDA_ENDGAME), (
        f"losing own planet 0 should flip us out of winning state and "
        f"apply -{LAMBDA_ENDGAME}; got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 4b — lose own planet whose loss does NOT flip us out → 0.
#
# Coverage pin: helper's "still winning even if we lose this one"
# branch. A swap-order bug between is_winning_state_if_lost and
# currently_winning would fail this test (test 4 alone doesn't isolate
# the two checks).
# ---------------------------------------------------------------------------

def test_endgame_bonus_lose_own_planet_that_does_not_flip_out():
    """Comfortable lead: lose 1 of 4 own planets, predicate stays True."""
    me = [_planet(i, 0, production=5, ships=20) for i in range(4)]
    opp = [_planet(10, 1, production=1, ships=5, x=30.0)]
    world = _world_from_planets(my_id=0, planets=me + opp, step=10)

    # row: subset that loses planet 0 (owner_T=1).
    row = _row(subset=(400,), owner_T=1, me_prod=0, opp_prod=20)
    bonus = _endgame_bonus(planet_id=0, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=True)
    assert bonus == 0.0, (
        f"losing 1 of 4 planets in a comfortable lead shouldn't apply a "
        f"penalty (is_winning_state_if_lost returns True); got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 4c — lose own planet when NOT currently in winning state → 0.
#
# Coverage pin: helper's `if currently_winning:` gate on the penalty
# branch. Losing a planet can't "flip us out" if we weren't in winning
# state to begin with.
# ---------------------------------------------------------------------------

def test_endgame_bonus_lose_own_planet_when_not_winning():
    """Losing position: losing a planet returns 0 (no winning state to
    flip out of)."""
    me = [_planet(0, 0, production=1, ships=10)]
    opp = [_planet(1, 1, production=5, ships=50, x=20.0)]
    world = _world_from_planets(my_id=0, planets=me + opp, step=0)
    # prod_advantage = 1 − 5 = −4 ⇒ predicate False.

    row = _row(subset=(500,), owner_T=1, me_prod=0, opp_prod=500)
    bonus = _endgame_bonus(planet_id=0, row=row, world=world,
                           my_id=0, opp_id=1, currently_winning=False)
    assert bonus == 0.0, (
        f"losing own planet when not in winning state shouldn't apply a "
        f"penalty; got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 5 — 4P game falls through to no bonus (Phase 4 MVP is 2P-only).
# ---------------------------------------------------------------------------

def test_endgame_bonus_4p_returns_zero():
    """4-player games (4 distinct non-neutral owners) get no bonus —
    Phase 4 MVP is 2P-only. The bonus path is gated by opp_id=None
    upstream (`_derive_opp_id_2p` returns None for non-2P seat counts),
    but the helper should also tolerate explicit opp_id=None.
    """
    planets = [_planet(i, owner, production=2, ships=10, x=10.0 * i)
               for i, owner in enumerate([0, 1, 2, 3])]
    world = _world_from_planets(my_id=0, planets=planets, step=0)

    row = _row(subset=(300,), owner_T=0, me_prod=990, opp_prod=10)
    # opp_id=None mirrors the 2P-derivation returning None for 4P.
    bonus = _endgame_bonus(planet_id=1, row=row, world=world,
                           my_id=0, opp_id=None, currently_winning=False)
    assert bonus == 0.0, (
        f"4P games (opp_id=None) should yield zero endgame bonus; "
        f"got {bonus}"
    )


# ---------------------------------------------------------------------------
# Test 6 — integration: LP picks the predicate-tipping subset.
#
# Rule 38 cycle: set `LAMBDA_ENDGAME = 0` in lp_outcome.py and the
# assertions in this test fail — the LP picks the non-tipping subset
# (cheap neutral capture P2) because its raw value exceeds the
# expensive opp capture P1. Restore `LAMBDA_ENDGAME = 1000` and the
# tipping subset (P1 capture) wins because the bonus dominates.
# ---------------------------------------------------------------------------

def test_solve_outcome_aware_picks_predicate_tipping_subset():
    """Engineered scenario:

    - P0 (mine, prod=1, ships=611): only source, budget = 611 at wait_N=0.
    - P1 (opp,  prod=2, ships=600): expensive to capture (611 ships); but
      capturing tips `is_winning_state` from False to True.
    - P2 (neutral, prod=3, ships=5): cheap to capture (6 ships); does NOT
      tip the predicate.

    Source budget forces a single fire (P1 ⊻ P2; can't bundle, 617 > 611).

    Pre-fix (LAMBDA_ENDGAME=0):
      LP picks P2 (net value +479) over P1 (net value +369).

    Post-fix (LAMBDA_ENDGAME=1000):
      P1 capture gets the +1000 endgame bonus → net +1369 > P2's +479.
      LP picks P1.
    """
    me = [_planet(0, 0, production=1, ships=611)]
    opp = [_planet(1, 1, production=2, ships=600, x=20.0)]
    neutral = [_planet(2, -1, production=3, ships=5, x=40.0)]
    planets = me + opp + neutral
    world = _world_from_planets(my_id=0, planets=planets, step=0)
    model = WorldModel(ledger={0: [], 1: [], 2: []}, timelines={},
                       horizon=500)

    # Hand-craft columns. Per Phase 5C/D, the LP runs the outcome_table
    # on whatever Arrivals these candidate columns translate to.
    col_p1 = Column(
        column_id=100, src_id=0, tgt_id=1,
        ships=611, wait_N=0, angle=0.0, eta=5,
        owner=0, value=1000.0, parent_column_id=None,
    )
    col_p2 = Column(
        column_id=101, src_id=0, tgt_id=2,
        ships=6, wait_N=0, angle=0.0, eta=5,
        owner=0, value=1000.0, parent_column_id=None,
    )

    res = solve_outcome_aware(
        [col_p1, col_p2], world, model,
        my_id=0,
        t_end=500,
        alpha_opp_penalty=1.0,
        ship_cost=1.0,
        time_limit_seconds=5.0,
    )

    # The LP should pick the P1-capturing subset (column_id 100 fires).
    fired_ids = {int(c.column_id) for c in res.fired_columns}
    assert 100 in fired_ids and 101 not in fired_ids, (
        f"LP should pick P1 capture (id 100) over P2 capture (id 101). "
        f"Got fired={sorted(fired_ids)}, status={res.status}, "
        f"objective={res.objective}, chosen={res.per_planet_chosen}"
    )


# ---------------------------------------------------------------------------
# Test 7 — non-regression: 4P games behave identically to pre-Phase 4.
# ---------------------------------------------------------------------------

def test_solve_outcome_aware_4p_no_behaviour_change():
    """In 4P, `_derive_opp_id_2p` returns None ⇒ `_endgame_bonus` returns 0
    everywhere ⇒ the LP's chosen subsets are unchanged from the Phase 5D
    behaviour. Locks the 4P safety contract.
    """
    # 4 owners + 1 neutral target.
    planets = [
        _planet(0, 0, production=1, ships=20, x=0.0),
        _planet(1, 1, production=1, ships=20, x=10.0),
        _planet(2, 2, production=1, ships=20, x=20.0),
        _planet(3, 3, production=1, ships=20, x=30.0),
        _planet(99, -1, production=2, ships=5, x=15.0),
    ]
    world = _world_from_planets(my_id=0, planets=planets, step=0)
    model = WorldModel(ledger={0: [], 1: [], 2: [], 3: [], 99: []},
                       timelines={}, horizon=500)

    col = Column(
        column_id=500, src_id=0, tgt_id=99,
        ships=6, wait_N=0, angle=0.0, eta=5,
        owner=0, value=1000.0, parent_column_id=None,
    )

    res = solve_outcome_aware(
        [col], world, model,
        my_id=0,
        t_end=500,
        alpha_opp_penalty=1.0,
        ship_cost=1.0,
        time_limit_seconds=5.0,
    )

    # No crash, no exception. Either the LP fires the capture (it's the
    # only positive-value option in this minimal world) or doesn't — but
    # the per-planet value should not include any bonus contribution.
    # We can't assert "no bonus" directly, so we assert the objective
    # is the raw value, no LAMBDA_ENDGAME-magnitude offset.
    if 500 in {int(c.column_id) for c in res.fired_columns}:
        # If the LP fired the capture, objective = me_prod_stream −
        # opp_prod_stream − ship_cost. Should be roughly 2 × 495 − 0 − 6
        # = 984. NOT 984 + 1000 (no bonus).
        assert res.objective < LAMBDA_ENDGAME, (
            f"4P game should not award endgame bonus; objective "
            f"{res.objective} suggests a bonus magnitude ≥ "
            f"{LAMBDA_ENDGAME} leaked in"
        )
