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


def _primary_source(portfolio: list) -> int | None:
    """Source contributing the most ships in this portfolio.

    Used by `min_distinct_primary_sources` constraint to enforce
    source-diversity across the top-K. Ties broken by min src_id.
    Empty portfolio returns None.
    """
    if not portfolio:
        return None
    src_ships: dict[int, int] = {}
    for c in portfolio:
        src_ships[int(c.src_id)] = src_ships.get(int(c.src_id), 0) + int(c.ships)
    if not src_ships:
        return None
    max_ships = max(src_ships.values())
    candidates = sorted(s for s, v in src_ships.items() if v == max_ships)
    return candidates[0]


def enumerate_top_k_portfolios_lp_seeded(
    cols: PrerankedColumns,
    ctx: TurnContext,
    opp: OppModelResult,
    *,
    k: int = 8,
    max_portfolio_size: int = 6,
    lp_time_limit_seconds: float = 0.05,
    min_distinct_primary_sources: int = 1,
) -> list[list]:
    """Top-K portfolios including the LP's chosen portfolio as one slot.

    Returns: [empty, lp_chosen, *beam_portfolios] truncated to k.

    If the LP returns no positive-value portfolio (no fired_columns),
    falls through to plain beam enumeration.

    ITEM 4a (composed-noodling-riddle.md): when
    `min_distinct_primary_sources > 1`, enforce source-diversity in the
    non-empty slots. After the standard beam build, replace later slots
    with portfolios from the beam that have a NEW primary source (a
    source not yet seen among already-accepted portfolios). Up to
    `min_distinct_primary_sources` distinct sources are guaranteed if
    the column pool offers them; otherwise we keep the highest-value
    beam portfolios. The empty portfolio is exempt (it has no source).

    Default = 1 preserves existing behaviour (no diversity enforcement).
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

    # 2. Run plain beam to fill the rest. Pull more than k so we have
    # alternatives to satisfy the diversity constraint.
    beam_k = max(int(k), int(min_distinct_primary_sources) * 3)
    beam_portfolios = enumerate_top_k_portfolios(
        columns, ctx, k=beam_k, max_portfolio_size=int(max_portfolio_size),
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

    # Track primary sources accepted so far (excluding empty / None).
    accepted_sources: set[int] = {
        s for s in (_primary_source(p) for p in out if p) if s is not None
    }

    # First pass: enforce source-diversity. Greedily add beam portfolios
    # whose primary source is NEW, until we hit min_distinct_primary_sources
    # OR we run out of beam variants.
    for p in beam_portfolios:
        if len(accepted_sources) >= int(min_distinct_primary_sources):
            break
        key = frozenset(int(c.column_id) for c in p)
        if key in seen:
            continue
        psrc = _primary_source(p)
        if psrc is None or psrc in accepted_sources:
            continue
        out.append(p)
        seen.add(key)
        accepted_sources.add(psrc)
        if len(out) >= int(k):
            break

    # Second pass: fill remaining slots with highest-value beam portfolios
    # (no diversity constraint on these — they're the "fallback variants").
    for p in beam_portfolios:
        if len(out) >= int(k):
            break
        key = frozenset(int(c.column_id) for c in p)
        if key in seen:
            continue
        out.append(p)
        seen.add(key)

    return out[:k]
