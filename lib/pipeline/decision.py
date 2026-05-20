"""Stage 5 — Decision rule.

(PrerankedColumns, OppModelResult, TurnContext) → DecisionResult.

The reference implementation wraps `solve_outcome_aware` from
`lib.joint_solver.lp_outcome`. That solver:
  1. Drops columns with `value <= 0` (the amputation pre-filter).
  2. Builds per-planet outcome tables via `enumerate_outcomes` (Stage 6
     happens inside this step — it's the leaf evaluator).
  3. Solves an MILP `max Σ y_{p,S} · [π_p(S)[me] − α·π_p(S)[opp]]
     − β Σ n_c x_c` with subset-uniqueness, candidate-subset linkage,
     and source-budget-over-time constraints.
  4. Returns `(moves, fired_columns, objective, status, ...)`.

`moves` is the wait_N==0 subset of fired_columns; the wait_N>0
selections are kept in `fired_columns` only (informational —
mpc.solve_turn:331 emits only wait_N==0). The Phase-D substitutes
(decision_maximin, decision_saddle, decision_ibr, decision_stackelberg)
will replace this with game-theoretic decision rules over outcome-table
leaves.
"""

from __future__ import annotations

from lib.joint_solver.lp_outcome import OutcomeAwareResult, solve_outcome_aware
from lib.pipeline.types import (
    DecisionResult, OppModelResult, PrerankedColumns, TurnContext,
)


def decision_outcome_aware_milp(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.3,
) -> DecisionResult:
    """Reference Stage-5 implementation.

    Mirror of `mpc.solve_turn:311-315`. Returns the LP's chosen moves
    plus diagnostics.
    """
    res: OutcomeAwareResult = solve_outcome_aware(
        cols.columns, ctx.world, opp.augmented_model,
        my_id=int(ctx.me),
        time_limit_seconds=float(time_limit_seconds),
    )
    return DecisionResult(
        moves=res.moves,
        fired_columns=res.fired_columns,
        objective=float(res.objective) if res.objective == res.objective else 0.0,
        status=str(res.status),
        n_x_vars=int(res.n_x_vars),
        n_y_vars=int(res.n_y_vars),
        n_constraints=int(res.n_constraints),
        per_planet_chosen=dict(res.per_planet_chosen),
        per_planet_value=dict(res.per_planet_value),
    )
