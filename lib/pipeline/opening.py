"""Opening dispatch — optional pipeline override.

TurnContext → OpeningResult.

For `step_now < OPENING_HORIZON`, the analytical agent runs a one-shot
multi-turn MILP (`opening_plan`) and commits to its schedule. The
three-case dispatch from `mpc.solve_turn:198-238`:

  (a) Schedule non-empty AND has fire_step == step_now entries:
      Emit those (the rest of schedule is planning intent; next turn
      re-derives and refreshes).
  (b) Schedule non-empty but NO fire_step == step_now entry:
      Planner's intentional wait; return [] with diagnostics.
  (c) Schedule empty (MILP infeasible / no candidates):
      Fall through to standard pipeline (return committed=None).

This stage runs BEFORE the standard Stages 2-7. If committed is non-None,
the pipeline short-circuits.
"""

from __future__ import annotations

from lib.joint_solver.opening_planner import OPENING_HORIZON, opening_plan
from lib.joint_solver.opening_search import (
    opening_plan_search,
    opening_search_enabled,
)
from lib.pipeline.types import CommittedMoves, OpeningResult, TurnContext


def opening_default(ctx: TurnContext) -> OpeningResult:
    """Reference opening override (parity with mpc.solve_turn:198-238).

    When `LP_OPENING_SEARCH=1`, dispatches to the Phase η.2 widened
    trajectory-matrix search instead. The search returns the same
    `OpeningPlan` shape, so the rest of this function is identical
    in either path.
    """
    if ctx.step_now >= OPENING_HORIZON:
        return OpeningResult(committed=None)

    if opening_search_enabled():
        op = opening_plan_search(ctx)
    else:
        op = opening_plan(ctx.world, ctx.model, ctx.me, ctx.num_seats)
    if not op.schedule:
        # Case (c): empty schedule → fall through.
        return OpeningResult(
            committed=None,
            diagnostics={
                "kind": "opening_empty_schedule",
                "n_vars": int(op.n_vars),
                "status": str(op.status),
            },
        )

    # Cases (a) and (b): schedule non-empty.
    opening_moves = [
        [int(e.src_id), float(e.angle), int(e.ships)]
        for e in op.schedule if int(e.fire_step) == ctx.step_now
    ]

    # Diagnostics matching mpc.solve_turn's MpcDiagnostics population.
    wait_dist: dict[int, int] = {}
    for e in op.schedule:
        offset = int(e.fire_step) - ctx.step_now
        wait_dist[offset] = wait_dist.get(offset, 0) + 1
    emitted_targets = [
        {"src_id": int(e.src_id), "tgt_id": int(e.tgt_id),
         "ships": int(e.ships), "angle": float(e.angle),
         "wait_N": 0}
        for e in op.schedule if int(e.fire_step) == ctx.step_now
    ]
    diagnostics = {
        "kind": "opening_schedule_committed",
        "n_vars": int(op.n_vars),
        "n_constraints": int(op.n_constraints),
        "n_schedule": len(op.schedule),
        "n_emitted": len(opening_moves),
        "objective": float(op.objective),
        "status": f"opening:{op.status}",
        "wait_distribution": wait_dist,
        "emitted_targets": emitted_targets,
    }

    return OpeningResult(
        committed=CommittedMoves(moves=opening_moves, persisted_state=None),
        diagnostics=diagnostics,
    )
