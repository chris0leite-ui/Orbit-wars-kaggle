"""Stage 3 — Pre-rank / pre-filter.

CandidateSet → PrerankedColumns.

The reference implementation:
  1. Computes `value_for_candidate` for each prerank tuple (Wald W1/W2
     lower bounds with ½·hi midpoint fallback).
  2. Builds a `Column` for each candidate via `column_from_candidate`.
  3. Drops columns with `value <= 0` — the "amputation" pre-filter
     that the analytical agent has used since Phase 3.
  4. Applies endgame portfolio focus (2P only) if a smallest-winning
     portfolio exists.

Step 3 is the load-bearing failure-mode driver flagged in the plan:
`value_for_candidate` flips sign on small ledger drift, so the LP gets
an amputated candidate set whose composition shifts turn to turn. The
Phase-C alternative `prerank_passthrough` replaces step 3 with a
pass-through (no drop) or dominance-only filter.
"""

from __future__ import annotations

from lib.joint_solver.columns import column_from_candidate
from lib.joint_solver.predicate import is_winning_state
from lib.joint_solver.portfolio import smallest_winning_portfolio
from lib.joint_solver.value import DEFAULT_GAMMA, value_for_candidate
from lib.pipeline.types import CandidateSet, PrerankedColumns, TurnContext


def _build_columns(prerank, world, model, *, my_id: int,
                   gamma: float = DEFAULT_GAMMA):
    """Mirror of `lib.joint_solver.mpc._build_columns`."""
    columns = []
    for idx, c in enumerate(prerank):
        value = float(value_for_candidate(
            c, world, model, my_id=int(my_id), gamma=float(gamma),
        ))
        columns.append(column_from_candidate(
            c, column_id=idx, owner=int(my_id), value=value,
        ))
    return columns


def prerank_w1w2_filter(
    cset: CandidateSet, ctx: TurnContext, *,
    gamma: float = DEFAULT_GAMMA,
    augmented_model=None,
) -> PrerankedColumns:
    """Reference Stage-3 implementation (parity with mpc.solve_turn).

    `augmented_model` is the OppModelResult.augmented_model from Stage 4 —
    the value function reads opp-projection-aware timelines. If None,
    falls back to ctx.model (degraded; shouldn't happen in the standard
    composition).
    """
    model = augmented_model if augmented_model is not None else ctx.model

    columns = _build_columns(
        cset.prerank, ctx.world, model, my_id=ctx.me, gamma=gamma,
    )
    n_before_filter = len(columns)

    # Endgame portfolio focus (2P only — 4P bypasses).
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
            # Only apply filter if it leaves a positive-value column;
            # otherwise it would zero out the LP. Defensive.
            if any(c.value > 0.0 for c in filtered):
                columns = filtered
                portfolio_filtered = True

    # NOTE: The "drop value <= 0" filter is applied inside `solve_outcome_aware`
    # at `lp_outcome.py:331` (filtering `active` columns). We preserve that
    # behavior in the reference Stage-5 decision rule, NOT here — so the
    # PrerankedColumns we return still contains v<=0 entries; they get
    # filtered downstream. This matches `mpc.solve_turn`'s exact behavior.

    return PrerankedColumns(
        columns=columns,
        n_before_filter=n_before_filter,
        n_after_filter=len(columns),
        portfolio=portfolio,
        portfolio_filtered=portfolio_filtered,
        is_winning_state=winning_now,
    )
