"""Stage 5 — Phase ε.1 adversarial maximin search.

Plan: /root/.claude/plans/composed-noodling-riddle.md (Phase ε.1).

Architecture (PI directive 2026-05-22 "use the prior to seed a smart
search"):
  - "Prior" = the LP-relaxation joint solver (currently `solve_outcome_aware`).
    It produces a recommended portfolio per turn. We enumerate top-K
    using `enumerate_top_k_portfolios_lp_seeded`.
  - "Smart search" = adversarial perturbation. For each of our K
    candidate portfolios, the mirror analytical opp computes opp's
    best response. We then evaluate the K×K (us, opp) matrix via the
    closed-form `leaf_value_for_portfolios` and pick our portfolio
    that maximizes the WORST-case opp response — robust selection.

Why not the existing `decision_depth2_search`:
  Depth-2 does opening-only T+1 lookahead via fast_sim.step + a
  recursive LP at T+1. It computes argmax over (my, mirror(my)) pairs.
  This module computes MAXIMIN over the full K×K matrix — different
  selection rule, different robustness guarantee. The two are
  complementary; depth-2 is "what if I'm right about opp's response";
  maximin is "what if opp is targeting my OTHER candidate."

Wallclock budget (target p95 ≤ 600ms under MILP inner):
  base LP solve:           ~50ms
  top-K my enumeration:    ~50ms × (K_my-1) ≈ 100ms at K_my=3
  K_my × mirror solves:    ~40ms × K_my ≈ 120ms at K_my=3
  K_my × K_my closed-form: ~3ms × 9 ≈ 30ms
  buffer + bookkeeping:    ~50ms
  TOTAL p50 ≈ 350ms — comfortably under 600ms budget.

Gate: opt-in via `LP_MAXIMIN_SEARCH=1`. Default OFF for clean A/B.

Closed-form leaf semantics (NOT fast_sim rollout): we use the same
math the LP optimizes against. The hypothesis is that K-step rollouts
add fidelity but cost wallclock; Phase ε.2 can swap to fast_sim
once ε.1 is validated.
"""
from __future__ import annotations

import os
import time

from lib.joint_solver.lp_outcome import ALPHA_OPP_PENALTY, SHIP_COST
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.leaf_outcome_table import (
    column_to_arrival,
    leaf_value_for_portfolios,
)
from lib.pipeline.opp_mirror_analytical import (
    predict_opp_response_to_my_portfolio,
)
from lib.pipeline.portfolio_enum_lp_seeded import (
    enumerate_top_k_portfolios_lp_seeded,
)
from lib.pipeline.types import (
    DecisionResult,
    OppModelResult,
    PrerankedColumns,
    TurnContext,
)


DEFAULT_K_MY = 3
DEFAULT_INNER_LP_TIME_LIMIT_S = 0.05
DEFAULT_OPP_TIME_LIMIT_S = 0.04
DEFAULT_BUDGET_RESERVE_MS = 100.0


def _maximin_enabled() -> bool:
    """Re-read env per call so tests / A/B harnesses can toggle without
    importlib.reload."""
    return os.environ.get("LP_MAXIMIN_SEARCH", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _k_my() -> int:
    raw = os.environ.get("LP_MAXIMIN_K_MY", "")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_K_MY


def decision_lagrangian_maximin(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.5,
    k_my: int | None = None,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
) -> DecisionResult:
    """Adversarial maximin search over top-K our × top-K opp portfolios.

    Returns the our-portfolio that maximizes the WORST closed-form leaf
    value across opp's K best responses. Falls back to the plain LP
    when the gate is off or any step fails.
    """

    # Fast path: feature off → plain LP.
    if not _maximin_enabled():
        return decision_outcome_aware_milp(
            cols, opp, ctx, time_limit_seconds=time_limit_seconds,
        )

    if not cols.columns:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="maximin:empty_columns",
        )

    t_start = time.perf_counter()
    deadline_s = float(time_limit_seconds) - DEFAULT_BUDGET_RESERVE_MS / 1000.0
    K = int(k_my) if k_my is not None else _k_my()

    # 1. Base LP for fallback and as a top-K seed.
    base_result = decision_outcome_aware_milp(
        cols, opp, ctx, time_limit_seconds=DEFAULT_INNER_LP_TIME_LIMIT_S,
    )

    # 2. Top-K my portfolios (LP-seeded).
    try:
        my_portfolios = enumerate_top_k_portfolios_lp_seeded(
            cols, ctx, opp, k=K,
            lp_time_limit_seconds=DEFAULT_INNER_LP_TIME_LIMIT_S,
        )
    except Exception:
        my_portfolios = []
    if not my_portfolios:
        return base_result

    # 3. Top-K opp responses — one per our portfolio.
    opp_responses: list[list] = []
    for my_p in my_portfolios:
        # Budget guard.
        if time.perf_counter() - t_start > deadline_s:
            break
        try:
            arrivals, _status = predict_opp_response_to_my_portfolio(
                ctx, my_p,
                time_limit_seconds=DEFAULT_OPP_TIME_LIMIT_S,
                return_status=True,
            )
        except Exception:
            arrivals = []
        opp_responses.append(arrivals or [])

    if not opp_responses:
        return base_result

    # 4. Evaluate the K × K_resp matrix via closed-form leaf.
    K_eff = len(opp_responses)  # may be < K if budget guard fired
    matrix: list[list[float]] = []
    for i, my_p in enumerate(my_portfolios[:K_eff]):
        my_arr = [column_to_arrival(c, ctx.step_now) for c in my_p]
        row: list[float] = []
        for j in range(K_eff):
            try:
                v = leaf_value_for_portfolios(
                    my_arr, opp_responses[j], ctx,
                    alpha_opp_penalty=float(alpha_opp_penalty),
                    ship_cost=float(ship_cost),
                )
            except Exception:
                v = float("-inf")
            row.append(v)
        matrix.append(row)

    if not matrix or not matrix[0]:
        return base_result

    # 5. Maximin selection.
    best_i = -1
    best_value = float("-inf")
    for i, row in enumerate(matrix):
        worst = min(row)
        if worst > best_value:
            best_value = worst
            best_i = i

    if best_i < 0 or best_value == float("-inf"):
        return base_result

    chosen = my_portfolios[best_i]
    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in chosen if int(c.wait_N) == 0
    ]
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return DecisionResult(
        moves=moves,
        fired_columns=list(chosen),
        objective=float(best_value),
        status=(
            f"maximin:K={K_eff},best={best_i},"
            f"worst_v={best_value:.0f},elapsed_ms={elapsed_ms:.0f}"
        ),
    )
