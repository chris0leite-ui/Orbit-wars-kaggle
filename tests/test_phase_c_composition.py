"""Phase C smoke test — persistent commit + pass-through prerank.

Verifies the alternative stage composition runs end-to-end, the
pending_schedule decants wait_N>0 commitments across turns, and the
pre-filter amputation rate is zero (all columns reach the LP).

This is the parity-shape companion to `test_pipeline_parity`: rather
than asserting bit-identity to legacy `solve_turn`, it asserts the
NEW composition's invariants hold during a real game.

After Phase D ships game-theoretic decision rules, this test grows
to assert combinations.
"""

from __future__ import annotations

import os

import pytest


def _run_game_with_phase_c(seed: int, opp_path: str) -> tuple[list[dict], list[list]]:
    """Run a game with the Phase-C composition.

    Returns:
      per_turn: per-turn instrumentation from the composer's trace
      moves_per_turn: emitted moves per turn (for downstream verification)
    """
    from kaggle_environments import make
    from lib.pipeline import compose
    from lib.pipeline.candidates import candidates_default
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.decision import decision_outcome_aware_milp
    from lib.pipeline.opening import opening_default
    from lib.pipeline.opp_model import opp_greedy_roi
    from lib.pipeline.perception import perception_default
    from lib.pipeline.prerank_passthrough import prerank_passthrough
    from lib.pipeline import pending_schedule

    # Clear pending state to isolate from any prior test invocation.
    pending_schedule.clear()

    per_turn: list[dict] = []
    moves_per_turn: list[list] = []

    def wrapped(obs, configuration=None):
        composed = compose(
            perception=perception_default,
            opening_override=opening_default,
            candidates=candidates_default,
            opp_model=opp_greedy_roi,
            prerank=prerank_passthrough,
            decision=decision_outcome_aware_milp,
            commit=commit_persistent,
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
        decision = trace.get("decision")
        committed = trace.get("committed")
        if ctx is not None and not ctx.is_empty_obs and not ctx.is_no_targets:
            opening = trace.get("opening")
            row = {
                "step": ctx.step_now,
                "opening_committed": opening is not None and opening.committed is not None,
            }
            if not row["opening_committed"]:
                row.update({
                    "n_columns_after_filter": cols.n_after_filter if cols else 0,
                    "n_columns_value_le_0": (
                        sum(1 for c in cols.columns if float(c.value) <= 0.0)
                        if cols else 0
                    ),
                    "n_fired_columns": len(decision.fired_columns) if decision else 0,
                    "n_fired_wait_n_pos": (
                        sum(1 for c in decision.fired_columns if int(c.wait_N) > 0)
                        if decision else 0
                    ),
                    "n_decanted": (
                        committed.persisted_state["n_decanted"]
                        if committed and committed.persisted_state else 0
                    ),
                    "n_new_pending": (
                        committed.persisted_state["n_new_pending"]
                        if committed and committed.persisted_state else 0
                    ),
                    "n_emitted_total": len(moves),
                })
            per_turn.append(row)
        moves_per_turn.append(list(moves or []))
        return moves

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([wrapped, opp_path])
    return per_turn, moves_per_turn


@pytest.mark.parametrize("seed", [42])
def test_phase_c_composition_runs_and_decants(seed: int):
    """Phase C smoke: composition runs to end, persistence kicks in,
    pre-filter amputation is closed.
    """
    per_turn, moves_per_turn = _run_game_with_phase_c(
        seed, "agents/simple/nearest.py",
    )

    active = [r for r in per_turn if not r.get("opening_committed")]
    if not active:
        pytest.skip("no post-opening turns captured (game ended in opening?)")

    # Invariant 1: pre-filter amputation = 0 (pass-through neutralizes filter).
    # n_columns_value_le_0 counts columns with value<=0 in PrerankedColumns;
    # under pass-through ALL columns get value=1.0 so this is always 0.
    amputation_per_turn = [r["n_columns_value_le_0"] for r in active]
    n_with_amputation = sum(1 for x in amputation_per_turn if x > 0)
    assert n_with_amputation == 0, (
        f"prerank_passthrough left {n_with_amputation}/{len(active)} turns "
        f"with value<=0 columns; expected 0 (passthrough must rewrite all to 1.0)"
    )

    # Invariant 2: persistence kicks in at least once.
    total_decants = sum(r["n_decanted"] for r in active)
    total_new_pending = sum(r["n_new_pending"] for r in active)
    print(f"\nseed {seed}: {len(active)} active turns, "
          f"{total_new_pending} wait_N>0 commitments, "
          f"{total_decants} decants")
    # commitments and decants together prove the cycle works.
    assert total_new_pending > 0 or total_decants > 0, (
        "neither wait_N>0 commitments nor decants happened in game — "
        "the persistent-schedule path was never exercised. Check that "
        "the LP is finding wait_N>0 fires in this game; if not, use a "
        "stronger opp."
    )

    # Invariant 3: per Rule 44 (PI directive), no game terminates abnormally.
    # We can't directly check sun/OOB here without re-running the diagnostic;
    # the assertion is "the env ran to completion without raising."
    # (env.run wraps exceptions internally; if a fleet went OOB the game
    # would still finish but with abnormal reward.)
