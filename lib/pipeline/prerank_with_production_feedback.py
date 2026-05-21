"""Phase F2a — prerank stage that adds production-feedback columns.

Pipeline order is:
  Stage 2: candidates_default          → prerank tuples
  Stage 4: opp_greedy_roi              → augmented model
  Stage 3: prerank_with_production_feedback (THIS)
           - build base columns from prerank tuples (value_for_candidate)
           - append compound columns (parent_column_id linkage)
  Stage 5: decision_outcome_aware_linked (lp_outcome_with_linkage)

The base portion is identical to prerank_passthrough (value computed for
diagnostics; no v<=0 filter). The new piece is the compound-column
generator from candidates_production_feedback.
"""

from __future__ import annotations

from lib.joint_solver.columns import column_from_candidate
from lib.joint_solver.predicate import is_winning_state
from lib.joint_solver.portfolio import smallest_winning_portfolio
from lib.joint_solver.value import DEFAULT_GAMMA, value_for_candidate
from lib.pipeline.candidates_production_feedback import (
    generate_compound_candidates,
)
from lib.pipeline.types import CandidateSet, PrerankedColumns, TurnContext


def prerank_with_production_feedback(
    cset: CandidateSet,
    ctx: TurnContext,
    *,
    gamma: float = DEFAULT_GAMMA,
    augmented_model=None,
) -> PrerankedColumns:
    """Stage 3 (Phase F2a): base columns + compound columns.

    Compound columns are generated AFTER base columns are built (so the
    parent_column_id linkage uses base column_ids). They're appended to
    the returned PrerankedColumns. The downstream Stage-5 LP enforces
    `x_compound <= x_parent` via linkage constraints.
    """
    model = augmented_model if augmented_model is not None else ctx.model

    # 1. Build base columns from prerank (diagnostic value, no v<=0 drop).
    base_columns = []
    for idx, c in enumerate(cset.prerank):
        try:
            _v = float(value_for_candidate(
                c, ctx.world, model, my_id=ctx.me, gamma=float(gamma),
            ))
        except Exception:
            _v = 0.0
        # Force value=1.0 (passthrough — let the LP rank on outcome-table
        # values, not value_for_candidate's noisy Wald bounds). Matches
        # prerank_passthrough behavior.
        base_columns.append(column_from_candidate(
            c, column_id=idx, owner=int(ctx.me), value=1.0,
        ))

    # 2. Append compound columns (parent_column_id → base.column_id).
    compound_columns = generate_compound_candidates(
        base_columns, ctx, next_col_id_start=len(base_columns),
    )

    columns = list(base_columns) + list(compound_columns)
    n_before_filter = len(columns)

    # 3. Endgame portfolio focus (2P only — mirror prerank_passthrough).
    winning_now = False
    portfolio_filtered = False
    portfolio: list = []
    if ctx.num_seats == 2:
        opp_id = 1 - ctx.me
        try:
            winning_now = bool(is_winning_state(ctx.world, ctx.me, opp_id))
        except Exception:
            winning_now = False
        try:
            portfolio = smallest_winning_portfolio(ctx.world, ctx.me, opp_id)
        except Exception:
            portfolio = []
        if portfolio:
            portfolio_set = set(int(pid) for pid in portfolio)
            my_planet_ids = {int(p.id) for p in ctx.my_planets}
            filtered = [
                c for c in columns
                if int(c.tgt_id) in portfolio_set or int(c.tgt_id) in my_planet_ids
            ]
            if any(c.value > 0.0 for c in filtered):
                columns = filtered
                portfolio_filtered = True

    return PrerankedColumns(
        columns=columns,
        n_before_filter=n_before_filter,
        n_after_filter=len(columns),
        portfolio=portfolio,
        portfolio_filtered=portfolio_filtered,
        is_winning_state=winning_now,
    )
