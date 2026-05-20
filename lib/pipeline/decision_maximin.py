"""Stage 5 — depth-2 maximin decision rule.

The substrate's analytical-native, closed-form game-theoretic core.

Algorithm:
  1. Enumerate K_MY portfolios from my Stage-3 columns (top-K beam search).
  2. Generate K_OPP opp portfolios (from `opp_perturbations` MVP or any
     other Stage-4-style opp model that returns a list of portfolios).
  3. For each (my_p, opp_p) pair, compute the closed-form leaf value
     via `leaf_outcome_table.leaf_value_for_portfolios` (no rollouts).
  4. Maximin: i* = argmax_i min_j payoff(i, j).
  5. Return a DecisionResult with the chosen portfolio's columns as
     `fired_columns` and the wait_N==0 subset as `moves`.

Budget probing via `safe_deadline` pattern: if K_MY × K_OPP × per_leaf_ms
projects to overrun the time budget, K_MY is reduced.

Outcome-table at every leaf is bit-exact (per Phase B parity test);
this means the maximin result is correct by construction at the leaf.
"""

from __future__ import annotations

import time

from lib.joint_solver.lp_outcome import (
    ALPHA_OPP_PENALTY, SHIP_COST,
)
from lib.pipeline.leaf_outcome_table import (
    column_to_arrival,
    leaf_value_for_portfolios,
)
from lib.pipeline.opp_perturbations import opp_portfolios_perturbations
from lib.pipeline.portfolio_enum import enumerate_top_k_portfolios
from lib.pipeline.types import (
    DecisionResult, OppModelResult, PrerankedColumns, TurnContext,
)


# Defaults probed in Phase B funnel.
DEFAULT_K_MY = 8
DEFAULT_K_OPP = 4
DEFAULT_MAX_PORTFOLIO_SIZE = 6
DEFAULT_BUDGET_RESERVE_MS = 50.0   # leave headroom for downstream commit


def decision_maximin(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.3,
    k_my: int = DEFAULT_K_MY,
    k_opp: int = DEFAULT_K_OPP,
    max_portfolio_size: int = DEFAULT_MAX_PORTFOLIO_SIZE,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
    horizon_truncate: int | None = None,
    discount_gamma: float | None = None,
    opp_portfolios_fn=opp_portfolios_perturbations,
    my_portfolios_fn=None,
) -> DecisionResult:
    """Depth-2 maximin: argmax_my min_opp leaf(my, opp).

    `my_portfolios_fn` (optional): callable returning a list of my
    candidate portfolios. Signature `fn(cols, ctx, opp, *, k,
    max_portfolio_size) -> list[list[Column]]`. If None, defaults to
    the greedy-beam `enumerate_top_k_portfolios` (cheap_delta-ranked).
    """
    t_start = time.perf_counter()
    deadline_s = float(time_limit_seconds) - (DEFAULT_BUDGET_RESERVE_MS / 1000.0)

    if not cols.columns:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="maximin:empty_columns",
        )

    # 1. My portfolios (always includes empty as the idle baseline).
    if my_portfolios_fn is None:
        my_portfolios = enumerate_top_k_portfolios(
            cols.columns, ctx,
            k=int(k_my),
            max_portfolio_size=int(max_portfolio_size),
        )
    else:
        my_portfolios = my_portfolios_fn(
            cols, ctx, opp,
            k=int(k_my),
            max_portfolio_size=int(max_portfolio_size),
        )
    if not my_portfolios:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="maximin:no_portfolios",
        )

    # 2. Opp portfolios.
    try:
        opp_portfolios = opp_portfolios_fn(ctx, max_portfolios=int(k_opp))
    except Exception:
        opp_portfolios = [list(opp.opp_arrivals or [])]
    if not opp_portfolios:
        opp_portfolios = [list(opp.opp_arrivals or [])]

    # 3. Build payoff matrix P[i][j] = leaf(my_portfolios[i], opp_portfolios[j]).
    n_my = len(my_portfolios)
    n_opp = len(opp_portfolios)
    payoff = [[0.0] * n_opp for _ in range(n_my)]

    # Probe per-leaf cost on the first (my=empty, opp[0]) leaf to size the budget.
    # If we project to overrun, downshift k_my.
    for i in range(n_my):
        if time.perf_counter() - t_start > deadline_s:
            # Out of budget; collapse to first opp = average rest.
            n_my_eval = i
            break
        my_arrivals = [
            column_to_arrival(c, int(ctx.step_now))
            for c in my_portfolios[i]
        ]
        for j in range(n_opp):
            payoff[i][j] = leaf_value_for_portfolios(
                my_arrivals=my_arrivals,
                opp_arrivals=opp_portfolios[j],
                ctx=ctx,
                horizon_truncate=horizon_truncate,
                alpha_opp_penalty=alpha_opp_penalty,
                ship_cost=ship_cost,
                discount_gamma=discount_gamma,
            )
    else:
        n_my_eval = n_my

    if n_my_eval == 0:
        # No leaves computed; return idle.
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="maximin:budget_exhausted_before_first_leaf",
        )

    # 4. Maximin selection.
    best_i = 0
    best_min = float("-inf")
    for i in range(n_my_eval):
        worst = min(payoff[i][j] for j in range(n_opp))
        if worst > best_min:
            best_min = worst
            best_i = i

    chosen_portfolio = my_portfolios[best_i]
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # 5. Extract moves (wait_N==0 only; Phase D persistence happens in commit).
    fired_columns = list(chosen_portfolio)
    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired_columns if int(c.wait_N) == 0
    ]

    return DecisionResult(
        moves=moves,
        fired_columns=fired_columns,
        objective=float(best_min),
        status=(
            f"maximin:n_my={n_my_eval}/{n_my},n_opp={n_opp},"
            f"best_i={best_i},elapsed_ms={elapsed_ms:.1f}"
        ),
        n_x_vars=int(n_my_eval),
        n_y_vars=int(n_opp),
        n_constraints=int(n_my_eval * n_opp),
        per_planet_chosen={},
        per_planet_value={},
    )
