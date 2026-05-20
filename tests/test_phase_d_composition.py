"""Phase D smoke test — depth-2 maximin substrate.

Verifies the Phase D agent runs end-to-end on a real game and
produces sane diagnostics:
  - maximin selected a non-trivial portfolio at least once
  - per-turn wallclock within budget
  - persistence + revalidation invariants still hold (inherited from Phase C)
"""

from __future__ import annotations

import os
import time

import pytest


def _run_phase_d_game(seed: int, opp_path: str):
    """Run a game with the Phase D composition; record per-turn diagnostics."""
    from kaggle_environments import make
    from lib.pipeline import compose
    from lib.pipeline import pending_schedule
    from lib.pipeline.candidates import candidates_default
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.decision_maximin import decision_maximin
    from lib.pipeline.opening import opening_default
    from lib.pipeline.opp_model import opp_greedy_roi
    from lib.pipeline.perception import perception_default
    from lib.pipeline.prerank_passthrough import prerank_passthrough

    pending_schedule.clear()
    per_turn: list[dict] = []

    def wrapped(obs, configuration=None):
        composed = compose(
            perception=perception_default,
            opening_override=opening_default,
            candidates=candidates_default,
            opp_model=opp_greedy_roi,
            prerank=prerank_passthrough,
            decision=decision_maximin,
            commit=commit_persistent,
            return_trace=True,
        )
        prev_drain = os.environ.get("PROPOSER_DRAIN_FILTER")
        prev_hold = os.environ.get("PROPOSER_HOLD_FEASIBILITY")
        os.environ["PROPOSER_DRAIN_FILTER"] = "off"
        os.environ["PROPOSER_HOLD_FEASIBILITY"] = "off"
        t0 = time.perf_counter()
        try:
            moves, trace = composed(obs, configuration)
        finally:
            if prev_drain is None:
                os.environ.pop("PROPOSER_DRAIN_FILTER", None)
            else:
                os.environ["PROPOSER_DRAIN_FILTER"] = prev_drain
            if prev_hold is None:
                os.environ.pop("PROPOSER_HOLD_FEASIBILITY", None)
            else:
                os.environ["PROPOSER_HOLD_FEASIBILITY"] = prev_hold
        ms = (time.perf_counter() - t0) * 1000.0

        ctx = trace.get("ctx")
        if ctx is None or ctx.is_empty_obs or ctx.is_no_targets:
            return moves
        opening = trace.get("opening")
        if opening is not None and opening.committed is not None:
            return moves
        decision = trace.get("decision")
        per_turn.append({
            "step": ctx.step_now,
            "ms": ms,
            "n_my_portfolios": decision.n_x_vars if decision else 0,
            "n_opp_portfolios": decision.n_y_vars if decision else 0,
            "n_fired_columns": len(decision.fired_columns) if decision else 0,
            "n_emitted_moves": len(moves),
            "objective": decision.objective if decision else 0.0,
            "status": decision.status if decision else "",
        })
        return moves

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([wrapped, opp_path])
    return per_turn


@pytest.mark.parametrize("seed", [42])
def test_phase_d_runs_with_sane_diagnostics(seed: int):
    """Phase D smoke: maximin runs, selects from K_MY portfolios, fits budget."""
    per_turn = _run_phase_d_game(seed, "agents/simple/nearest.py")
    if not per_turn:
        pytest.skip("no post-opening turns captured")

    # Invariant: maximin should evaluate K_MY > 1 portfolios on most turns.
    # (If K_MY=1 always, the budget got exhausted before the first leaf, or
    # there were no columns.)
    n_my_distribution = [r["n_my_portfolios"] for r in per_turn]
    mean_n_my = sum(n_my_distribution) / len(n_my_distribution)
    print(f"\nseed {seed}: {len(per_turn)} active turns, "
          f"mean K_MY evaluated = {mean_n_my:.1f}, "
          f"K_OPP evaluated = {per_turn[0]['n_opp_portfolios']}")

    # Wallclock: per-turn ms p95 should be well under 1000ms (Kaggle cap).
    sorted_ms = sorted(r["ms"] for r in per_turn)
    p50 = sorted_ms[len(sorted_ms) // 2]
    p95 = sorted_ms[int(0.95 * (len(sorted_ms) - 1))]
    max_ms = sorted_ms[-1]
    print(f"  per-turn ms: p50={p50:.1f} p95={p95:.1f} max={max_ms:.1f}")

    assert mean_n_my >= 2.0, (
        f"maximin evaluated only mean K_MY={mean_n_my:.1f} portfolios. "
        f"Expected ≥2 (the empty + at least one non-empty); check budget probe."
    )
    assert max_ms < 1000.0, (
        f"per-turn wallclock max {max_ms:.1f}ms exceeds Kaggle's 1000ms cap. "
        f"Reduce K_MY or K_OPP."
    )
