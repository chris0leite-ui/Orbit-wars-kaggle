"""Stage 3 — pre-rank pass-through (closes pre-filter amputation).

Alternative to `prerank_w1w2_filter`. Computes `value_for_candidate`
for diagnostics but rewrites every column's value to a small positive
constant (1.0) so the downstream LP filter at `lp_outcome.py:331`
(which drops columns with `value <= 0`) keeps them all.

This is sound because `column.value` does NOT enter the LP's objective:
  - The objective uses `_value_for_outcome(row, my_id, alpha)` on each
    per-planet subset (lp_outcome.py:412).
  - `column.value` is only used by (i) the LP's input pre-filter
    (`value <= 0` drop), and (ii) the endgame portfolio focus's
    `any(c.value > 0.0)` defensive check.

By rewriting value to a fixed positive number, both checks pass for
every column, and the LP solves over the unamputated candidate set.

The original Wald-conservative value is stashed in `column.value_original`
(a new attribute we tack on) for diagnostics. Phase D's game-theoretic
decision rules read the outcome-table directly and never need
`column.value`; this rewrite is purely to neutralize the input filter.
"""

from __future__ import annotations

from lib.joint_solver.columns import column_from_candidate
from lib.joint_solver.predicate import is_winning_state
from lib.joint_solver.portfolio import smallest_winning_portfolio
from lib.joint_solver.value import DEFAULT_GAMMA, value_for_candidate
from lib.pipeline.types import CandidateSet, PrerankedColumns, TurnContext


_PASSTHROUGH_VALUE = 1.0


def prerank_passthrough(
    cset: CandidateSet, ctx: TurnContext, *,
    gamma: float = DEFAULT_GAMMA,
    augmented_model=None,
) -> PrerankedColumns:
    """Pass-through Stage-3: rewrite all column values to 1.0 to neutralize
    the LP's value<=0 input filter. Endgame portfolio focus still applies.
    """
    model = augmented_model if augmented_model is not None else ctx.model

    # Build columns; stash original value for diagnostics; rewrite to 1.0.
    columns = []
    for idx, c in enumerate(cset.prerank):
        try:
            v_orig = float(value_for_candidate(
                c, ctx.world, model, my_id=int(ctx.me), gamma=float(gamma),
            ))
        except Exception:
            v_orig = 0.0
        col = column_from_candidate(
            c, column_id=idx, owner=int(ctx.me), value=_PASSTHROUGH_VALUE,
        )
        # Stash for diagnostics (frozen dataclass: setattr works because the
        # underlying Column dataclass isn't strict-frozen; if it is, this
        # is a no-op and the diagnostic is unavailable).
        try:
            object.__setattr__(col, "value_original", v_orig)
        except Exception:
            pass
        columns.append(col)
    n_before_filter = len(columns)

    # Endgame portfolio focus (2P only). Identical to reference; safe
    # since all values are now 1.0 so the `any(c.value > 0.0)` check passes.
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
