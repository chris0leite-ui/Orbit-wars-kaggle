"""Stage 5 — Stackelberg-leader decision rule.

The leader commits, follower best-responds, leader picks the X
maximizing payoff under the follower's best response.

  for each my_portfolio[i]:
      opp_response[i] = mirror_analytical_opp_BR(ctx, my_portfolio[i])
      payoff[i]       = leaf(my_portfolio[i], opp_response[i], ctx)
  return argmax_i payoff[i]

Different from maximin (which assumes opp is adversarial across a SET
of responses). Here opp is a SINGLE point estimate per my action,
produced by `predict_opp_response_to_my_portfolio` — opp solves its
own analytical LP given my committed arrivals merged into its view.

This gives real action-dependent game-theoretic content: different my
portfolios produce different opp responses, so the payoff matrix is
non-trivially shaped.

Used by Phase D v3. Falls back to greedy ROI as opp if the
mirror-analytical opp computation fails for a portfolio.
"""

from __future__ import annotations

import time

from lib.joint_solver.lp_outcome import ALPHA_OPP_PENALTY, SHIP_COST
from lib.joint_solver.opp_projection import predict_opp_multi_launch
from lib.pipeline.leaf_outcome_table import (
    column_to_arrival, leaf_value_for_portfolios,
)
from lib.pipeline.opp_mirror_analytical import predict_opp_response_to_my_portfolio
from lib.pipeline.portfolio_enum import enumerate_top_k_portfolios
from lib.pipeline.types import (
    DecisionResult, OppModelResult, PrerankedColumns, TurnContext,
)


DEFAULT_K_MY = 8
DEFAULT_MAX_PORTFOLIO_SIZE = 6
DEFAULT_BUDGET_RESERVE_MS = 50.0
DEFAULT_OPP_TIME_LIMIT_S = 0.04


def _greedy_roi_arrivals(ctx: TurnContext) -> list[tuple[int, int, int, int]]:
    """Fallback opp arrivals (action-independent greedy ROI) in (pid, eta_rel, owner, ships).

    `predict_opp_multi_launch` returns eta_absolute; subtract step_now
    to get eta_rel.
    """
    try:
        opp_abs = predict_opp_multi_launch(
            ctx.world, int(ctx.me), int(ctx.num_seats),
        )
    except Exception:
        return []
    step_now = int(ctx.step_now)
    out: list[tuple[int, int, int, int]] = []
    for (pid, eta_abs, owner, ships) in opp_abs:
        eta_rel = int(eta_abs) - step_now
        if eta_rel <= 0:
            continue
        out.append((int(pid), int(eta_rel), int(owner), int(ships)))
    return out


def decision_stackelberg_leader(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.3,
    k_my: int = DEFAULT_K_MY,
    max_portfolio_size: int = DEFAULT_MAX_PORTFOLIO_SIZE,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
    horizon_truncate: int | None = None,
    discount_gamma: float | None = None,
    opp_time_limit_seconds: float = DEFAULT_OPP_TIME_LIMIT_S,
    my_portfolios_fn=None,
) -> DecisionResult:
    """Stackelberg-leader: I commit, opp best-responds, I pick argmax_i."""
    t_start = time.perf_counter()
    deadline_s = float(time_limit_seconds) - (DEFAULT_BUDGET_RESERVE_MS / 1000.0)

    if not cols.columns:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="stackelberg:empty_columns",
        )

    # 1. My portfolios.
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

    # 2. For each my portfolio, compute opp's best-response via
    # mirror-analytical, then leaf value.
    n_my = len(my_portfolios)
    n_opp_mirror_succeeded = 0
    # Bug #8: distinguish "opp legitimately has nothing to do" from
    # "the inner mirror LP failed". Both used to be merged into a single
    # n_opp_fallback counter.
    n_opp_mirror_empty = 0
    n_opp_mirror_failed = 0
    payoffs: list[float] = []
    opp_responses: list[list] = []

    fallback_opp = _greedy_roi_arrivals(ctx)

    for i, my_p in enumerate(my_portfolios):
        # Budget guard: leave room for the remaining portfolios + commit overhead.
        if time.perf_counter() - t_start > deadline_s:
            # Pad remaining with -inf so they're not selected.
            payoffs.extend([float("-inf")] * (n_my - len(payoffs)))
            opp_responses.extend([[] for _ in range(n_my - len(opp_responses))])
            break

        my_arrivals = [column_to_arrival(c, ctx.step_now) for c in my_p]

        # Opp's best response to this my portfolio.
        try:
            opp_arrivals, opp_status = predict_opp_response_to_my_portfolio(
                ctx, my_p,
                time_limit_seconds=float(opp_time_limit_seconds),
                return_status=True,
            )
        except Exception:
            opp_arrivals, opp_status = [], "failed"
        if opp_status == "ok":
            n_opp_mirror_succeeded += 1
        elif opp_status == "empty":
            n_opp_mirror_empty += 1
            # Empty is a legitimate "opp does nothing" — don't apply
            # fallback; use the empty list as the actual response.
        else:  # "failed"
            n_opp_mirror_failed += 1
            opp_arrivals = fallback_opp

        # Leaf value: my portfolio vs opp's best response.
        leaf = leaf_value_for_portfolios(
            my_arrivals, opp_arrivals, ctx,
            horizon_truncate=horizon_truncate,
            alpha_opp_penalty=float(alpha_opp_penalty),
            ship_cost=float(ship_cost),
            discount_gamma=discount_gamma,
        )
        payoffs.append(float(leaf))
        opp_responses.append(opp_arrivals)

    if not payoffs:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="stackelberg:no_portfolios_evaluated",
        )

    # 3. Argmax.
    best_i = max(range(len(payoffs)), key=lambda i: payoffs[i])
    chosen_portfolio = my_portfolios[best_i]

    fired_columns = list(chosen_portfolio)
    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired_columns if int(c.wait_N) == 0
    ]

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    status = (
        f"stackelberg:n_my={n_my},best_i={best_i},"
        f"opp_mirror={n_opp_mirror_succeeded},"
        f"opp_empty={n_opp_mirror_empty},opp_failed={n_opp_mirror_failed},"
        f"elapsed_ms={elapsed_ms:.1f}"
    )

    return DecisionResult(
        moves=moves,
        fired_columns=fired_columns,
        objective=float(payoffs[best_i]),
        status=status,
    )
