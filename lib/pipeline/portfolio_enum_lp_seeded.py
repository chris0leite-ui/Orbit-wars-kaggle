"""LP-seeded portfolio enumeration.

Augments `enumerate_top_k_portfolios` (greedy beam by cheap_delta) with
the LP's own chosen portfolio as one of the slots. Cheap_delta is a
per-candidate proposer score; the LP optimizes the JOINT outcome-table
value. Beam search misses portfolios that score high jointly but not
per-candidate (e.g., a 3-fleet gang-up on a single planet).

Implementation: call `solve_outcome_aware` once (the Phase C reference
decision rule) to extract its `fired_columns`. That portfolio becomes
the first non-empty slot in the returned list; greedy beam fills the
rest. Cost: ~15 ms additional per turn (Phase B funnel said ms_decision
p90=13.1, max=14.4 for the MILP — the LP-seeded enum doubles that to
~30 ms total per-turn).

Used by Phase D's maximin when the cheap_delta beam alone misses
LP-optimal portfolios.
"""

from __future__ import annotations

from lib.joint_solver.lp_outcome import solve_outcome_aware
from lib.pipeline.portfolio_enum import enumerate_top_k_portfolios
from lib.pipeline.types import OppModelResult, PrerankedColumns, TurnContext


def enumerate_top_k_portfolios_lp_seeded(
    cols: PrerankedColumns,
    ctx: TurnContext,
    opp: OppModelResult,
    *,
    k: int = 8,
    max_portfolio_size: int = 6,
    lp_time_limit_seconds: float = 0.05,
) -> list[list]:
    """Top-K portfolios including the LP's chosen portfolio as one slot.

    Returns: [empty, lp_chosen, *beam_portfolios] truncated to k.

    If the LP returns no positive-value portfolio (no fired_columns),
    falls through to plain beam enumeration.
    """
    columns = cols.columns
    if not columns:
        return [[]]

    # 1. Run the LP once to get its chosen portfolio.
    lp_portfolio: list = []
    try:
        res = solve_outcome_aware(
            columns, ctx.world, opp.augmented_model,
            my_id=int(ctx.me),
            time_limit_seconds=float(lp_time_limit_seconds),
        )
        lp_portfolio = list(res.fired_columns or [])
    except Exception:
        lp_portfolio = []

    # 2. Run plain beam to fill the rest.
    beam_portfolios = enumerate_top_k_portfolios(
        columns, ctx, k=int(k), max_portfolio_size=int(max_portfolio_size),
    )

    # Compose: [empty, lp_portfolio (if non-empty + distinct), beam (skip dupes)]
    out: list[list] = []
    seen: set[frozenset] = set()
    # Empty first (idle baseline; required for maximin)
    out.append([])
    seen.add(frozenset())

    if lp_portfolio:
        key = frozenset(int(c.column_id) for c in lp_portfolio)
        if key not in seen:
            out.append(lp_portfolio)
            seen.add(key)

    for p in beam_portfolios:
        key = frozenset(int(c.column_id) for c in p)
        if key in seen:
            continue
        out.append(p)
        seen.add(key)
        if len(out) >= int(k):
            break

    return out[:k]
