"""Phase B Rule-38 pin test — pre-filter amputation.

The current Stage-3 prerank computes `value_for_candidate` for every
column; the value uses Wald-conservative bounds with a ½·hi midpoint
fallback that flips sign on small ledger drift. Downstream at
`lp_outcome.py:331`, columns with `value <= 0` are dropped before the
LP sees them.

This test PINS the failure: across a real game, count the per-turn
fraction of columns whose `value <= 0`. The amputation rate is the
fraction of LP-invisible columns. HANDOVER's diagnosis is that this
rate is high (the LP gets an amputated candidate set whose composition
shifts turn to turn).

After Phase C ships `prerank_passthrough`, this test gets a counterpart
that asserts the pass-through variant feeds all columns to the LP
(amputation rate = 0).
"""

from __future__ import annotations

import os

import pytest


def _run_game_capture_value_distribution(seed: int, opp_path: str) -> list[dict]:
    """Per turn: capture (n_columns, n_columns_with_v_le_0, value distribution)."""
    from kaggle_environments import make
    from lib.pipeline import compose
    from lib.pipeline.candidates import candidates_default
    from lib.pipeline.commit import commit_stateless
    from lib.pipeline.decision import decision_outcome_aware_milp
    from lib.pipeline.opening import opening_default
    from lib.pipeline.opp_model import opp_greedy_roi
    from lib.pipeline.perception import perception_default
    from lib.pipeline.prerank import prerank_w1w2_filter

    per_turn: list[dict] = []

    def wrapped(obs, configuration=None):
        composed = compose(
            perception=perception_default,
            opening_override=opening_default,
            candidates=candidates_default,
            opp_model=opp_greedy_roi,
            prerank=prerank_w1w2_filter,
            decision=decision_outcome_aware_milp,
            commit=commit_stateless,
            return_trace=True,
        )
        prev_drain = os.environ.get("PROPOSER_DRAIN_FILTER")
        prev_hold = os.environ.get("PROPOSER_HOLD_FEASIBILITY")
        os.environ["PROPOSER_DRAIN_FILTER"] = "off"
        os.environ["PROPOSER_HOLD_FEASIBILITY"] = "off"
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
        ctx = trace.get("ctx")
        cols = trace.get("cols")
        if ctx is None or ctx.is_empty_obs or ctx.is_no_targets:
            return moves
        opening = trace.get("opening")
        if opening is not None and opening.committed is not None:
            return moves
        if cols is None:
            return moves
        col_list = cols.columns
        n_total = len(col_list)
        n_v_le_0 = sum(1 for c in col_list if float(c.value) <= 0.0)
        per_turn.append({
            "step": ctx.step_now,
            "n_columns": n_total,
            "n_columns_v_le_0": n_v_le_0,
            "amputation_rate": (n_v_le_0 / n_total) if n_total else 0.0,
            "portfolio_filtered": bool(cols.portfolio_filtered),
        })
        return moves

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([wrapped, opp_path])
    return per_turn


@pytest.mark.parametrize("seed", [42])
def test_prefilter_amputation_nontrivial(seed: int):
    """PIN: the value<=0 pre-filter drops a non-trivial fraction of the
    candidate set on most turns under the current reference pipeline.

    This is the documented failure mode #4 from the Phase B plan.
    """
    per_turn = _run_game_capture_value_distribution(
        seed, "agents/baseline/main.py",
    )

    if not per_turn:
        pytest.skip("no active turns captured (game terminated immediately?)")

    n_turns = len(per_turn)
    rates = [r["amputation_rate"] for r in per_turn if r["n_columns"] > 0]
    if not rates:
        pytest.skip("no turns with any candidate columns")
    mean_rate = sum(rates) / len(rates)
    median_rate = sorted(rates)[len(rates) // 2]
    high_amputation_turns = sum(1 for r in rates if r >= 0.5)

    print(f"\nseed {seed}: {n_turns} active turns, "
          f"mean amputation rate = {mean_rate:.2f}, "
          f"median = {median_rate:.2f}, "
          f"high (>=50%) amputation turns = {high_amputation_turns}/{len(rates)}")

    # Rule 38 pin: assert the amputation is non-trivial. If this ever
    # drops to 0, the failure is closed (Phase C did its job).
    assert mean_rate > 0.05, (
        f"Expected non-trivial amputation rate under the reference "
        f"prerank_w1w2_filter, found mean rate {mean_rate:.3f}. "
        f"Either the failure is closed (unexpected under "
        f"prerank_w1w2_filter) or the test setup is wrong."
    )
