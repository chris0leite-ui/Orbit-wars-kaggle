"""Pin tests for LP_PENDING_AWARE_BUDGET — Phase ζ.1 fix.

Bug: `solve_outcome_aware` builds its source-budget constraint from
`world.planets_by_id[src].ships` directly, so already-committed wait_N>0
fires from prior turns (held by `pending_schedule` until their decant)
are ignored. The LP can over-commit ships, leaving commit_persistent +
the env to silently cap one of the two same-source fires.

Coverage:

1. OFF (default) — pending fire is ignored; the LP allows a same-source
   over-commit. This pins the BUG behavior so we can prove the fix flips
   it. (Rule 38: fix-verification reproduces the failure state.)
2. ON — pending fire is deducted from the budget; the LP picks a single
   feasible fire instead of double-committing.
3. ON — future-decant pending fire (`fire_step > step_now`) only binds
   constraints with `wait_N >= (fire_step - step_now)`; immediate fires
   (`wait_N == 0`) stay feasible if the source has the ships now.
4. Unit pins for the `_pending_ships_consumed_by` helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.joint_solver.lp_outcome import (
    _pending_ships_consumed_by,
    _pending_aware_budget_enabled,
    solve_outcome_aware,
)
from lib.pipeline.pending_schedule import (
    PendingSchedule,
    ScheduledFire,
    get_default_pending,
)
from lib.world_model import WorldModel


@pytest.fixture(autouse=True)
def _isolate_pending(monkeypatch):
    """Clear the module-level pending singleton between tests so commits
    from one test never leak into another. Also clear LP_PENDING_AWARE_BUDGET
    so each test opts in explicitly."""
    monkeypatch.delenv("LP_PENDING_AWARE_BUDGET", raising=False)
    get_default_pending().reset()
    yield
    get_default_pending().reset()


# ---------------------------------------------------------------------------
# Helpers (mirror test_lp_outcome.py).
# ---------------------------------------------------------------------------


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
# Unit: _pending_ships_consumed_by helper.
# ---------------------------------------------------------------------------


def test_pending_consumed_helper_empty_returns_zero():
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=0,
                                      pending_fires=[]) == 0


def test_pending_consumed_helper_only_matches_source():
    fires = [
        ScheduledFire(src_id=0, tgt_id=5, ships=7, angle=0.0,
                      fire_step=10, committed_at_step=6, wait_N_original=4),
        ScheduledFire(src_id=1, tgt_id=5, ships=11, angle=0.0,
                      fire_step=10, committed_at_step=6, wait_N_original=4),
    ]
    # Asking for src=0 only counts the src=0 fire's 7 ships.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=0,
                                      pending_fires=fires) == 7
    assert _pending_ships_consumed_by(sid=1, step_now=10, u=0,
                                      pending_fires=fires) == 11


def test_pending_consumed_helper_filters_by_wait_horizon():
    """A future-decant fire (fire_step > step_now + u) doesn't bind
    constraints with smaller u — those LP fires resolve BEFORE the
    pending one decants."""
    # Pending fire decants at step_now+5.
    fires = [
        ScheduledFire(src_id=0, tgt_id=5, ships=20, angle=0.0,
                      fire_step=15, committed_at_step=10, wait_N_original=5),
    ]
    # LP fire with wait_N=0 (immediate): resolves at step_now=10, BEFORE
    # the pending fire's step 15. Doesn't compete → consumption is 0.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=0,
                                      pending_fires=fires) == 0
    # LP fire with wait_N=4: still before pending's step 15. 0.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=4,
                                      pending_fires=fires) == 0
    # LP fire with wait_N=5: same tick as pending. Must compete → 20.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=5,
                                      pending_fires=fires) == 20
    # LP fire with wait_N=7: after pending. Pending already consumed → 20.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=7,
                                      pending_fires=fires) == 20


def test_pending_consumed_helper_clamps_past_due_fires():
    """A pending fire whose fire_step < step_now (should have been pruned)
    is treated as immediate (d=0). Defensive — prune_past should normally
    catch it."""
    fires = [
        ScheduledFire(src_id=0, tgt_id=5, ships=8, angle=0.0,
                      fire_step=8, committed_at_step=4, wait_N_original=4),
    ]
    # step_now=10, fire_step=8 → d=clamp(8-10, 0)=0. Binds u=0.
    assert _pending_ships_consumed_by(sid=0, step_now=10, u=0,
                                      pending_fires=fires) == 8


# ---------------------------------------------------------------------------
# Behaviour: LP gate OFF → over-commit allowed (BUG, pinned for Rule 38).
# ---------------------------------------------------------------------------


def test_off_default_lp_overcommits_with_pending():
    """With LP_PENDING_AWARE_BUDGET unset, the LP doesn't know about
    pending wait_N>0 fires. Source has exactly enough ships for ONE fire,
    but the LP picks a same-source wait_N=0 fire AND the pending fire
    decants this turn → over-commit reproduces the seed-42 step-5 bug.

    Setup:
      - Source planet 0 has 20 ships.
      - Pending fire: 20 ships, fire_step == step_now (decants this turn).
      - LP candidate: 20 ships, wait_N=0 from same source → if LP picks
        it, total commanded = 40 > 20.

    Without the fix, the LP picks the candidate. With the fix (next
    test) it must reject it.
    """
    assert not _pending_aware_budget_enabled()  # default OFF

    me = _planet(0, 0, ships=20, production=1, x=10.0, y=10.0)
    tgt = _planet(10, 1, ships=10, production=2, x=50.0, y=50.0)
    world, model = _world_and_model(my_id=0, planets=[me, tgt], step=5)

    # Inject the pending fire (decants this turn → fire_step=5).
    get_default_pending().commit(
        my_id=0, new_fires=[ScheduledFire(
            src_id=0, tgt_id=10, ships=20, angle=0.0,
            fire_step=5, committed_at_step=1, wait_N_original=4,
        )],
    )

    # Candidate: same source, same ship count, wait_N=0.
    cols = [_column(
        column_id=0, src_id=0, tgt_id=10, ships=20, value=100.0, eta=3,
    )]
    res = solve_outcome_aware(cols, world, model, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # BUG: LP fires column 0 even though pending already consumes the 20.
    assert 0 in fired, (
        f"expected over-commit (bug pin) but LP didn't fire; "
        f"status={res.status}"
    )


# ---------------------------------------------------------------------------
# Behaviour: LP gate ON → over-commit blocked.
# ---------------------------------------------------------------------------


def test_on_blocks_overcommit_with_immediate_pending(monkeypatch):
    """Same setup as the OFF test, but LP_PENDING_AWARE_BUDGET=1. The
    pending fire claims all 20 ships at u=0 → LP candidate can't fire."""
    monkeypatch.setenv("LP_PENDING_AWARE_BUDGET", "1")

    me = _planet(0, 0, ships=20, production=1, x=10.0, y=10.0)
    tgt = _planet(10, 1, ships=10, production=2, x=50.0, y=50.0)
    world, model = _world_and_model(my_id=0, planets=[me, tgt], step=5)

    get_default_pending().commit(
        my_id=0, new_fires=[ScheduledFire(
            src_id=0, tgt_id=10, ships=20, angle=0.0,
            fire_step=5, committed_at_step=1, wait_N_original=4,
        )],
    )

    cols = [_column(
        column_id=0, src_id=0, tgt_id=10, ships=20, value=100.0, eta=3,
    )]
    res = solve_outcome_aware(cols, world, model, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # FIX: LP refuses the over-commit; column 0 is rejected.
    assert 0 not in fired, (
        f"fix should block over-commit; LP still fired column 0. "
        f"fired={fired}, status={res.status}"
    )


def test_on_allows_immediate_when_future_pending_only(monkeypatch):
    """LP_PENDING_AWARE_BUDGET=1 but the pending fire decants in the
    FUTURE (fire_step > step_now). A wait_N=0 LP fire from the same
    source resolves BEFORE the pending one → no competition; the LP
    can still fire.

    Setup: source has 20 ships now. Pending: 20 ships, fire_step=10
    (step_now=5; decants in 5 ticks). LP candidate: 5 ships, wait_N=0
    against a neutral 3-ship target (capturable solo).
    """
    monkeypatch.setenv("LP_PENDING_AWARE_BUDGET", "1")

    me = _planet(0, 0, ships=20, production=1, x=10.0, y=10.0)
    # Neutral target (owner=-1) — doesn't accumulate ships, so 5 captures.
    tgt = _planet(10, -1, ships=3, production=2, x=50.0, y=50.0)
    world, model = _world_and_model(my_id=0, planets=[me, tgt], step=5)

    # Pending fires in 5 ticks (well after the LP fire's wait_N=0).
    get_default_pending().commit(
        my_id=0, new_fires=[ScheduledFire(
            src_id=0, tgt_id=10, ships=20, angle=0.0,
            fire_step=10, committed_at_step=1, wait_N_original=9,
        )],
    )

    # Small immediate fire that fits in current source ships.
    cols = [_column(
        column_id=0, src_id=0, tgt_id=10, ships=5, value=100.0, eta=3,
    )]
    res = solve_outcome_aware(cols, world, model, my_id=0)
    fired = {c.column_id for c in res.fired_columns}
    # The wait_N=0 fire at u=0 doesn't compete with the wait_N=5 pending.
    assert 0 in fired, (
        f"fix should permit immediate fire when pending decants later; "
        f"fired={fired}, status={res.status}"
    )
