"""Stage 5 — Phase ε.2.a depth-2 LP search wrapper.

Plan: /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md

Opening-only (step_now < OPENING_HORIZON) opt-in wrapper around the
plain LP. The kinematic table's per-turn rebuild + fast_sim.step's
~0.5 ms per-tick simulation make this affordable inside the 600 ms
budget.

Algorithm:

  if LP_DEPTH2_SEARCH unset OR step_now >= OPENING_HORIZON:
      return decision_outcome_aware_milp(cols, opp, ctx, ...)  # unchanged path

  enumerate top-K my portfolios at T via LP-seeded greedy beam
  for each my_portfolio in top-K:
      opp_resp_T = mirror-analytical opp response  # Phase D v3 reuse
      leaf_T     = closed-form leaf(my, opp_resp, ctx)  # leaf_outcome_table

      # Forward one tick with my action only (opp idle in continuation).
      # Asymmetry is fine: opp's response IS modeled at T via leaf_T.
      snap_T1   = fast_sim.step(snap_T, [my_moves, []])

      # T+1 evaluation: run the pipeline at the simulated state, score leaf.
      ctx_T1    = build_turn_context_from_snap(snap_T1)
      cols_T1   = propose + prerank at T+1
      opp_T1    = opp_greedy_roi(ctx_T1)  # cheap opp model at T+1
      my_dec_T1 = LP at T+1
      leaf_T1   = closed-form leaf(my_T1, opp_T1, ctx_T1)

      total = leaf_T + leaf_T1
  pick argmax(total)

Wallclock-aware: skips continuation for low-priority portfolios when
budget runs out; falls through to the plain LP's choice if all skip.

Gate: opt-in via `LP_DEPTH2_SEARCH=1`. Default OFF for clean A/B.
"""

from __future__ import annotations

import os
import time

from lib import fast_sim
from lib.joint_solver.lp_outcome import ALPHA_OPP_PENALTY, SHIP_COST
from lib.pipeline.candidates import candidates_default
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.leaf_outcome_table import (
    column_to_arrival,
    leaf_value_for_portfolios,
)
from lib.pipeline.opp_mirror_analytical import predict_opp_response_to_my_portfolio
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.portfolio_enum_lp_seeded import (
    enumerate_top_k_portfolios_lp_seeded,
)
from lib.pipeline.prerank_passthrough import prerank_passthrough
from lib.pipeline.types import (
    DecisionResult,
    OppModelResult,
    PrerankedColumns,
    TurnContext,
)


DEFAULT_OPENING_HORIZON = 30
DEFAULT_K_MY = 4
DEFAULT_BUDGET_RESERVE_MS = 100.0
DEFAULT_OPP_TIME_LIMIT_S = 0.04
DEFAULT_INNER_LP_TIME_LIMIT_S = 0.05


def _depth2_search_enabled() -> bool:
    """Re-read env var per call so tests/A/B harnesses can toggle without
    importlib.reload."""
    return os.environ.get("LP_DEPTH2_SEARCH", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _opening_horizon() -> int:
    raw = os.environ.get("LP_DEPTH2_OPENING_HORIZON", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_OPENING_HORIZON


def _k_my() -> int:
    raw = os.environ.get("LP_DEPTH2_K_MY", "")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_K_MY


def decision_depth2_search(
    cols: PrerankedColumns,
    opp: OppModelResult,
    ctx: TurnContext,
    *,
    time_limit_seconds: float = 0.5,
    k_my: int | None = None,
    opening_horizon: int | None = None,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
) -> DecisionResult:
    """Depth-2 LP search wrapper. Opens-only; otherwise plain LP."""

    # Fast path: feature off OR past opening horizon → just call the LP.
    horizon = int(opening_horizon) if opening_horizon is not None else _opening_horizon()
    if not _depth2_search_enabled() or int(ctx.step_now) >= horizon:
        return decision_outcome_aware_milp(
            cols, opp, ctx, time_limit_seconds=time_limit_seconds,
        )

    if not cols.columns:
        return DecisionResult(
            moves=[], fired_columns=[], objective=0.0,
            status="depth2:empty_columns",
        )

    t_start = time.perf_counter()
    deadline_s = float(time_limit_seconds) - DEFAULT_BUDGET_RESERVE_MS / 1000.0

    K = int(k_my) if k_my is not None else _k_my()

    # 1. Always run the base LP — gives us a fallback choice + the LP's
    # portfolio for inclusion in the top-K.
    base_result = decision_outcome_aware_milp(
        cols, opp, ctx, time_limit_seconds=DEFAULT_INNER_LP_TIME_LIMIT_S,
    )

    # 2. Enumerate top-K my portfolios (LP-seeded).
    my_portfolios = enumerate_top_k_portfolios_lp_seeded(
        cols, ctx, opp, k=K,
    )
    if not my_portfolios:
        # Fall back to base LP's choice.
        return base_result

    # 3. Build T snapshot for fast_sim.step.
    try:
        snap_T = fast_sim.from_obs(
            ctx.obs_d, ctx.configuration or {}, episode_seed=0,
            num_seats=int(ctx.num_seats),
        )
    except Exception:
        # If snapshot construction fails (rare; defensive), use the base LP.
        return base_result

    me = int(ctx.me)
    num_seats = int(ctx.num_seats)

    # 4. For each portfolio: leaf_T + leaf_T1.
    totals: list[float] = []
    n_evaluated = 0
    n_t1_pipeline_skipped = 0

    for i, my_p in enumerate(my_portfolios):
        # Budget guard: leave room for the remaining portfolios.
        if time.perf_counter() - t_start > deadline_s:
            # Pad remaining with -inf so they're not selected.
            totals.extend([float("-inf")] * (len(my_portfolios) - len(totals)))
            break

        # --- T leaf ---
        my_arr_T = [column_to_arrival(c, ctx.step_now) for c in my_p]
        try:
            opp_arr_T, _opp_status = predict_opp_response_to_my_portfolio(
                ctx, my_p,
                time_limit_seconds=DEFAULT_OPP_TIME_LIMIT_S,
                return_status=True,
            )
        except Exception:
            opp_arr_T = []
        leaf_T = leaf_value_for_portfolios(
            my_arr_T, opp_arr_T, ctx,
            alpha_opp_penalty=float(alpha_opp_penalty),
            ship_cost=float(ship_cost),
        )

        # --- Simulate to T+1 via fast_sim.step (my action only) ---
        my_moves_T = [
            [int(c.src_id), float(c.angle), int(c.ships)]
            for c in my_p if int(c.wait_N) == 0
        ]
        actions: list[list] = [[] for _ in range(num_seats)]
        actions[me] = my_moves_T

        leaf_T1 = 0.0
        try:
            snap_T1 = fast_sim.step(snap_T, actions)
            # If the game ended at T (rare in opening), no continuation.
            if not snap_T1.fake_env.done:
                # Build T+1 context from snap_T1's obs.
                obs_T1 = _obs_from_snap(snap_T1, me)
                ctx_T1 = perception_default(obs_T1, ctx.configuration)
                if (not ctx_T1.is_empty_obs
                        and not ctx_T1.is_no_targets
                        and ctx_T1.world is not None
                        and ctx_T1.model is not None):
                    # T+1 pipeline: candidates → opp_greedy → prerank → LP.
                    # Order matches lib/pipeline/compose.py: opp BEFORE
                    # prerank so prerank uses the opp-augmented model.
                    try:
                        cset_T1 = candidates_default(ctx_T1)
                        opp_T1 = opp_greedy_roi(ctx_T1)
                        cols_T1 = prerank_passthrough(
                            cset_T1, ctx_T1,
                            augmented_model=opp_T1.augmented_model,
                        )
                        dec_T1 = decision_outcome_aware_milp(
                            cols_T1, opp_T1, ctx_T1,
                            time_limit_seconds=DEFAULT_INNER_LP_TIME_LIMIT_S,
                        )
                        my_arr_T1 = [
                            column_to_arrival(c, ctx_T1.step_now)
                            for c in dec_T1.fired_columns
                        ]
                        leaf_T1 = leaf_value_for_portfolios(
                            my_arr_T1, opp_T1.opp_arrivals, ctx_T1,
                            alpha_opp_penalty=float(alpha_opp_penalty),
                            ship_cost=float(ship_cost),
                        )
                    except Exception:
                        # T+1 pipeline failed; continuation contributes 0.
                        n_t1_pipeline_skipped += 1
                        leaf_T1 = 0.0
        except Exception:
            # fast_sim.step failed (rare); continuation contributes 0.
            leaf_T1 = 0.0

        totals.append(leaf_T + leaf_T1)
        n_evaluated += 1

    if not totals or all(t == float("-inf") for t in totals):
        # All portfolios had budget skip — fall through to base LP.
        return base_result

    # 5. Argmax.
    best_idx = max(range(len(totals)), key=lambda i: totals[i])
    chosen = my_portfolios[best_idx]

    fired = list(chosen)
    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired if int(c.wait_N) == 0
    ]

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    status = (
        f"depth2:k={len(my_portfolios)},eval={n_evaluated},"
        f"best={best_idx},skip_t1={n_t1_pipeline_skipped},"
        f"elapsed_ms={elapsed_ms:.0f}"
    )

    # If our argmax IS the base LP's portfolio (typical when search
    # confirms LP), reuse base_result's moves to preserve any
    # downstream invariants. Otherwise, emit our chosen portfolio.
    return DecisionResult(
        moves=moves,
        fired_columns=fired,
        objective=float(totals[best_idx]),
        status=status,
    )


def _obs_from_snap(snap, me: int) -> dict:
    """Extract the agent-visible observation for seat `me` from a Snapshot.

    fast_sim stores per-seat state at `snap.state[i].observation`. The
    observation is a kaggle Struct; coerce to plain dict for the
    perception stage (which accepts both but prefers dict for clarity).
    """
    obs = snap.state[int(me)].observation
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
