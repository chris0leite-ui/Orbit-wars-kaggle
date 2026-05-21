"""Top-K portfolio enumeration for game-theoretic decision rules.

A "portfolio" is a list of Columns representing a set of launches I
commit to fire (with their wait_N timings). Per-source ship-budget
feasibility: at each fire_step u, `Σ_{c: src(c)=s, wait_N(c)<=u} n_c
≤ initial_ships(s) + production(s) · u`.

The enumeration is beam-search style:
  - Sort columns by `cheap_delta` (proposer's first-pass score) desc.
  - Build top-1 portfolio: highest-cheap-delta single column (if any).
  - Build top-2: greedy add next-highest column respecting budget.
  - Continue to top-K_max_portfolio_size.
  - Then build perturbations: drop-one variants of the largest portfolio.
  - Return up to K distinct portfolios sorted by aggregate cheap_delta.

ALWAYS includes the empty portfolio (idle baseline) as the first slot —
critical for maximin since "idle" is a valid play that some opp models
may not punish.

This is the simplest viable enumeration for Phase D MVP. Future
alternatives (multi-seed beam search, dominance pruning, MILP relaxation)
register alongside.
"""

from __future__ import annotations

from lib.joint_solver.columns import Column
from lib.pipeline.types import TurnContext


def _source_inventory(columns: list[Column], ctx: TurnContext) -> dict[int, tuple[int, int]]:
    """Return {src_id: (initial_ships, production_per_step)} for sources we use."""
    inv: dict[int, tuple[int, int]] = {}
    for col in columns:
        sid = int(col.src_id)
        if sid in inv:
            continue
        planet = ctx.world.planets_by_id.get(sid)
        if planet is None:
            continue
        inv[sid] = (int(planet.ships), int(planet.production))
    return inv


def _is_feasible(portfolio: list[Column], inv: dict[int, tuple[int, int]]) -> bool:
    """Check per-source ship budget across all fire steps in the portfolio."""
    if not portfolio:
        return True
    # Collect fire-step thresholds per source.
    fire_steps_by_src: dict[int, set[int]] = {}
    for col in portfolio:
        fire_steps_by_src.setdefault(int(col.src_id), set()).add(int(col.wait_N))
    for sid, fire_steps in fire_steps_by_src.items():
        if sid not in inv:
            return False
        initial, prod = inv[sid]
        for u in sorted(fire_steps):
            ships_used = sum(
                int(c.ships) for c in portfolio
                if int(c.src_id) == sid and int(c.wait_N) <= u
            )
            budget = initial + prod * max(0, u)
            if ships_used > budget:
                return False
    return True


def _aggregate_value(portfolio: list[Column]) -> float:
    """Aggregate cheap_delta across a portfolio."""
    return sum(float(c.cheap_delta) for c in portfolio)


def enumerate_top_k_portfolios(
    columns: list[Column],
    ctx: TurnContext,
    *,
    k: int = 8,
    max_portfolio_size: int = 6,
    include_empty: bool = True,
) -> list[list[Column]]:
    """Return up to `k` distinct feasible portfolios.

    Always includes the empty portfolio first if `include_empty=True`
    (the "idle" baseline; necessary for maximin to consider not-firing
    as a strategy).

    Constructs portfolios via greedy beam:
      - Sort columns by cheap_delta desc.
      - For each starting "seed" column in the top `k` by cheap_delta:
        - Start portfolio with that seed.
        - Greedily extend with next-highest-cheap_delta column subject
          to source-budget feasibility, up to max_portfolio_size.
      - Deduplicate by frozenset of column_ids.

    Returns the top `k` portfolios by aggregate cheap_delta.
    """
    inv = _source_inventory(columns, ctx)
    # Filter to our positive-cheap_delta columns with a valid source.
    sorted_cols = sorted(
        [c for c in columns if int(c.src_id) in inv],
        key=lambda c: float(c.cheap_delta), reverse=True,
    )

    portfolios: list[list[Column]] = []
    seen: set[frozenset] = set()

    if include_empty:
        portfolios.append([])
        seen.add(frozenset())

    # Top-K seeds by cheap_delta; each seed initializes a greedy beam.
    for seed_idx in range(min(k, len(sorted_cols))):
        seed = sorted_cols[seed_idx]
        portfolio = [seed]
        if not _is_feasible(portfolio, inv):
            continue
        # Greedy extend.
        for cand in sorted_cols:
            if len(portfolio) >= max_portfolio_size:
                break
            if int(cand.column_id) == int(seed.column_id):
                continue
            if any(int(c.column_id) == int(cand.column_id) for c in portfolio):
                continue
            candidate_portfolio = portfolio + [cand]
            if _is_feasible(candidate_portfolio, inv):
                portfolio = candidate_portfolio
        key = frozenset(int(c.column_id) for c in portfolio)
        if key in seen:
            continue
        seen.add(key)
        portfolios.append(portfolio)

    # Sort portfolios by aggregate cheap_delta (descending); empty stays first.
    # Keep the empty portfolio anchored.
    empty = [p for p in portfolios if not p]
    non_empty = sorted(
        (p for p in portfolios if p),
        key=lambda p: _aggregate_value(p), reverse=True,
    )
    result = (empty + non_empty)[:k]
    return result
