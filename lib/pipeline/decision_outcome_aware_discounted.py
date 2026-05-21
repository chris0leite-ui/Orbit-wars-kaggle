"""Phase F1 — discounted-leaf, truncated-horizon decision rule.

Calls solve_outcome_aware with:
  - discount_gamma = 0.99 (per-tick γ-weighted prod_stream; closes the
    undiscounted-T=500 hypothesis from Phase D analysis)
  - t_end = step_now + 200 (truncate to a realistic episode-length
    horizon instead of T=500)

Everything else matches the reference decision (the LP body itself
is shared via the new discount_gamma kwarg on solve_outcome_aware).
"""

from __future__ import annotations

from lib.joint_solver.lp_outcome import (
    ALPHA_OPP_PENALTY, SHIP_COST, T_END, OutcomeAwareResult,
    solve_outcome_aware,
)
from lib.pipeline.types import (
    DecisionResult, OppModelResult, PrerankedColumns, TurnContext,
)


# Defaults: γ = 0.99 matches `value_for_candidate`'s pv_horizon constant
# (lib.scoring.pv_horizon at lib/joint_solver/value.py:28). Horizon
# step+200 covers a median episode-length tail (games end at step
# 100-200 per state/current.md observations).
DEFAULT_DISCOUNT_GAMMA = 0.99
DEFAULT_HORIZON_TAIL = 200


def decision_outcome_aware_discounted(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.3,
    discount_gamma: float = DEFAULT_DISCOUNT_GAMMA,
    horizon_tail: int = DEFAULT_HORIZON_TAIL,
) -> DecisionResult:
    """Phase F1 stage-5 alternative — discounted leaf + truncated horizon."""
    step_now = int(ctx.step_now)
    # Per-call truncated horizon. Bounded above by T_END so we never
    # extend the LP's view past the env's actual episode end.
    t_end = min(int(T_END), step_now + int(horizon_tail))

    res: OutcomeAwareResult = solve_outcome_aware(
        cols.columns, ctx.world, opp.augmented_model,
        my_id=int(ctx.me),
        t_end=int(t_end),
        time_limit_seconds=float(time_limit_seconds),
        discount_gamma=float(discount_gamma),
    )
    return DecisionResult(
        moves=res.moves,
        fired_columns=res.fired_columns,
        objective=float(res.objective) if res.objective == res.objective else 0.0,
        status=f"outcome_aware_discounted(γ={discount_gamma:.3f},h={t_end}):{res.status}",
        n_x_vars=int(res.n_x_vars),
        n_y_vars=int(res.n_y_vars),
        n_constraints=int(res.n_constraints),
        per_planet_chosen=dict(res.per_planet_chosen),
        per_planet_value=dict(res.per_planet_value),
    )
