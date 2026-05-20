"""Regression test for mpc.solve_turn silent-idle on empty schedule.

Pre-fix bug at `lib/joint_solver/mpc.py:191`: when the opening planner
returns `OpeningPlan(n_vars > 0, schedule=[])` (i.e., MILP found
candidates but couldn't pick any), the dispatch condition
`if op.n_vars > 0 or op.schedule:` is True → solve_turn returns `[]`
without ever invoking the Phase-4 LP. Silent idle turn.

Expected post-fix behaviour: when `schedule == []`, fall through to
the Phase-4 LP path. Either the LP emits something or returns [] for
defensible reasons — but at minimum the LP was given a chance to
run, and diagnostics reflect that.
"""

from __future__ import annotations

from unittest.mock import patch

from lib.joint_solver import mpc
from lib.joint_solver.opening_planner import OpeningPlan


def _minimal_obs(step=5, my_id=0, omega=0.05):
    """Minimal obs dict that makes solve_turn reach the opening dispatch.

    Two planets, one mine + one opp's, both orbiting (omega>0). Source
    has enough ships to make column-gen non-empty even if the MILP
    returns empty.
    """
    return {
        "player": my_id,
        "planets": [
            # (id, owner, x, y, radius, ships, production)
            (0, my_id, 30.0, 50.0, 1.5, 30, 2),
            (1, 1 - my_id, 70.0, 50.0, 1.5, 30, 2),
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }


def test_solve_turn_falls_through_on_empty_schedule_with_nonzero_n_vars():
    """When opening_plan returns n_vars>0 but schedule=[], the Phase-4
    LP MUST run (not silent-idle).

    Pre-fix: returns []; diagnostics solver_status starts with "opening:".
    Post-fix: solver_status does NOT start with "opening:" — it reflects
    the Phase-4 LP path.
    """
    obs = _minimal_obs(step=5)

    empty_plan = OpeningPlan(
        schedule=[], objective=0.0, n_vars=5,
        n_constraints=3, status="milp_no_solution_greedy",
        pruning_waterfall={},
    )

    with patch.object(mpc, "opening_plan", return_value=empty_plan) as mock_op:
        moves, diag = mpc.solve_turn(obs, return_diagnostics=True)
        assert mock_op.called, "opening_plan was not invoked"

    # The KEY assertion: solver_status should reflect the Phase-4 LP run,
    # NOT the opening's empty-schedule outcome. Post-fix, solver_status
    # is the lp_outcome status (e.g., "milp_ok", "greedy_fallback"); pre-
    # fix, it would be "opening:milp_no_solution_greedy".
    assert not diag.solver_status.startswith("opening:"), (
        f"solver_status='{diag.solver_status}' — solve_turn stayed in "
        f"opening mode with an empty schedule. Should have fallen through "
        f"to Phase-4 LP. Diagnostics: {diag}")


def test_solve_turn_stays_in_opening_when_schedule_has_only_future_entries():
    """When opening_plan returns a non-empty schedule with NO fire_step==step_now
    entries (intentional planning wait), solve_turn MUST return [] WITHOUT
    falling through. This preserves the "stateless re-derivation +
    commit-and-execute" contract.

    The schedule says "I plan to fire at step 7" while we're at step 5;
    the next call at step 6 will re-derive and pick fresh. This is the
    planner's intent, not a silent idle.
    """
    obs = _minimal_obs(step=5)
    from lib.joint_solver.opening_planner import ScheduleEntry

    future_plan = OpeningPlan(
        schedule=[ScheduleEntry(
            fire_step=7, src_id=0, tgt_id=1,
            ships=15, angle=0.0, eta=5, value=10.0,
        )],
        objective=10.0, n_vars=3, n_constraints=2,
        status="milp_ok", pruning_waterfall={},
    )

    with patch.object(mpc, "opening_plan", return_value=future_plan):
        moves, diag = mpc.solve_turn(obs, return_diagnostics=True)

    assert moves == [], (
        f"Expected [] (planner's intentional wait); got {moves}.")
    # Post-fix: solver_status should reflect the opening dispatch (not
    # Phase-4 LP), since the planner DID find a schedule and we honor it.
    assert diag.solver_status.startswith("opening:"), (
        f"solver_status='{diag.solver_status}' — non-empty schedule with "
        f"no fire-now entry should stay in opening mode (intentional wait), "
        f"not fall through to Phase-4 LP. Diagnostics: {diag}")


def test_solve_turn_emits_when_schedule_has_fire_now_entry():
    """Sanity check (must always pass, pre- AND post-fix): when the
    schedule contains a fire_step==step_now entry, solve_turn emits it.
    """
    obs = _minimal_obs(step=5)
    from lib.joint_solver.opening_planner import ScheduleEntry

    fire_now_plan = OpeningPlan(
        schedule=[
            ScheduleEntry(fire_step=5, src_id=0, tgt_id=1,
                          ships=15, angle=0.0, eta=5, value=10.0),
        ],
        objective=10.0, n_vars=3, n_constraints=2,
        status="milp_ok", pruning_waterfall={},
    )

    with patch.object(mpc, "opening_plan", return_value=fire_now_plan):
        moves, diag = mpc.solve_turn(obs, return_diagnostics=True)

    assert len(moves) == 1, f"Expected 1 emitted move; got {moves}"
    assert moves[0][0] == 0, f"src_id should be 0; got {moves[0]}"
    assert moves[0][2] == 15, f"ships should be 15; got {moves[0]}"
    assert diag.solver_status.startswith("opening:"), (
        f"Should be opening dispatch; got {diag.solver_status}")
