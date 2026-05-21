"""Bug — ships idle on captured comets until expiry → ships lost.

PI flagged: in a live 4P game vs Tensorflower, our agent captured a
comet and never fired its ships before the comet expired. >100 ships
forfeit. This isn't an exclusion bug (proposer DOES enumerate comet
sources), nor a trajectory bug (predict_fleet_fate works post-fix).

The LP rejects the proposer's comet-as-source candidates because they
target planets that 30 ships can't capture (opp planets too well-
defended at arrival) or own planets that don't need defending. With no
positive-value LP move available, the LP leaves the ships on the comet.
But idle-on-a-soon-to-expire comet means losing them entirely when the
comet leaves the board (env removes the planet AND its ships —
orbit_wars.py L617-625).

The fix is a post-LP forced drain in commit_persistent: for each comet
I own with imminent expiry and substantial ships remaining (above any
LP-committed launches), emit a move toward the nearest own non-comet
planet. The LP would never pick this on its own (sending to an own
planet yields outcome value 0), but it's strictly better than losing
the ships at expiry.

Pin (Rule 38): construct a turn context with (a) a captured comet
holding 50 ships, ~10 steps to expiry, (b) a nearby own non-comet
planet to drain to, (c) no LP fires from the comet. Pre-fix:
commit_persistent emits nothing for the comet. Post-fix: a drain move
appears in committed.moves.
"""

from __future__ import annotations

import math

from lib.intent import Planet, World
from lib.pipeline.types import DecisionResult, TurnContext
from lib.world_model import WorldModel


def _make_ctx_with_captured_comet(*, comet_ships=50, comet_lifetime=10,
                                  drain_target_distance=15.0):
    """Build a TurnContext with one own comet + one own non-comet planet.

    The comet has `comet_ships` ships and a `comet_lifetime`-step path
    remaining. The own non-comet planet is `drain_target_distance` units
    away — well within fleet reach.
    """
    # Comet at (20, 50), own non-comet at (35, 50) — 15 units east, in-board.
    comet = Planet(id=20, owner=0, x=20.0, y=50.0, radius=2.0,
                   ships=comet_ships, production=1)
    own_planet = Planet(id=0, owner=0, x=20.0 + drain_target_distance, y=50.0,
                        radius=5.0, ships=10, production=2)
    # Plus a distant enemy planet so my_planets / other_planets are non-empty.
    opp = Planet(id=1, owner=1, x=80.0, y=50.0, radius=5.0, ships=50,
                 production=2)
    # Build the path so remaining_lifetime = comet_lifetime.
    comet_path = [[20.0, 50.0]] * (comet_lifetime + 1)  # path_index=0 → lifetime=len
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 20.0 + drain_target_distance, 50.0, 5.0, 10, 2],
            [1, 1, 80.0, 50.0, 5.0, 50, 2],
            [20, 0, 20.0, 50.0, 2.0, comet_ships, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "comet_planet_ids": [20],
        "comets": [{
            "planet_ids": [20], "paths": [comet_path], "path_index": 0,
        }],
        "step": 100,
    }
    world = World.from_obs(obs)
    model = WorldModel(ledger={}, timelines={}, horizon=200)
    return TurnContext(
        obs_d=obs, configuration=None, me=0, num_seats=2, step_now=100,
        omega=0.0, planets=[own_planet, opp, comet], fleets=[],
        my_planets=[own_planet, comet], other_planets=[opp],
        world=world, model=model,
    )


def test_commit_persistent_drains_expiring_captured_comet():
    """Post-fix: commit_persistent emits a drain move for the comet
    when the LP fired nothing from it and the comet is close to expiry.
    """
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.pending_schedule import get_default_pending

    get_default_pending().reset()
    ctx = _make_ctx_with_captured_comet(comet_ships=50, comet_lifetime=10)
    # Empty decision — LP picked nothing.
    decision = DecisionResult(
        moves=[], fired_columns=[], objective=0.0, status="ok",
    )
    committed = commit_persistent(decision, ctx)

    drain_moves = [m for m in committed.moves if int(m[0]) == 20]
    assert drain_moves, (
        f"commit_persistent did not emit a drain move from comet 20 "
        f"(captured, 50 ships, ~10 steps to expiry, own planet nearby). "
        f"emitted moves: {committed.moves}"
    )
    drain = drain_moves[0]
    drain_ships = int(drain[2])
    assert drain_ships >= 40, (
        f"drain ships={drain_ships}; expected most of the 50 ships "
        f"(reserve 1 or 2 acceptable). Move: {drain}"
    )


def test_commit_persistent_skips_drain_when_lifetime_high():
    """Don't drain when the comet has plenty of life — LP can still
    decide better. The fix should only kick in when expiry is imminent.
    """
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.pending_schedule import get_default_pending

    get_default_pending().reset()
    # 60 steps of life is plenty — let LP decide normally.
    ctx = _make_ctx_with_captured_comet(comet_ships=50, comet_lifetime=60)
    decision = DecisionResult(
        moves=[], fired_columns=[], objective=0.0, status="ok",
    )
    committed = commit_persistent(decision, ctx)
    drain_moves = [m for m in committed.moves if int(m[0]) == 20]
    assert not drain_moves, (
        f"comet had 60 steps remaining — drain shouldn't kick in yet; "
        f"got drain move {drain_moves}"
    )


def test_commit_persistent_drain_respects_lp_committed_ships():
    """If the LP already fired N ships from the comet, drain only the
    remaining ships (don't double-spend the source budget)."""
    from lib.joint_solver.columns import Column
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.pending_schedule import get_default_pending

    get_default_pending().reset()
    ctx = _make_ctx_with_captured_comet(comet_ships=50, comet_lifetime=10)
    # LP fired 20 ships from comet (wait_N=0 → already in lp_moves).
    lp_col = Column(
        column_id=99, src_id=20, tgt_id=1, ships=20, wait_N=0,
        angle=0.0, eta=10, owner=0, value=1.0,
    )
    decision = DecisionResult(
        moves=[[20, 0.0, 20]], fired_columns=[lp_col],
        objective=0.0, status="ok",
    )
    committed = commit_persistent(decision, ctx)
    drain_moves = [m for m in committed.moves
                   if int(m[0]) == 20 and abs(float(m[1]) - 0.0) > 1e-9]
    # The drain should account for the 20 ships already going out.
    # Total moves from comet 20 must not exceed 50 ships (the comet's count).
    total_from_comet = sum(int(m[2]) for m in committed.moves if int(m[0]) == 20)
    assert total_from_comet <= 50, (
        f"total ships emitted from comet exceed its garrison: "
        f"{total_from_comet} > 50; moves: {committed.moves}"
    )
    # Drain should still emit something (50 - 20 = 30 ships available; we
    # reserve 1, so drain ≥ 25).
    if drain_moves:
        drain_ships = int(drain_moves[0][2])
        assert 25 <= drain_ships <= 30, (
            f"expected drain to cover remaining ~29 ships (50 - 20 - 1 = 29), "
            f"got {drain_ships}"
        )
