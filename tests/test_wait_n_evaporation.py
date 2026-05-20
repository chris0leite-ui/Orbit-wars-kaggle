"""Phase B Rule-38 pin test — wait_N > 0 column evaporation.

The current analytical pipeline emits only wait_N == 0 columns each
turn (lp_outcome.py:498-499). Columns the LP selects with wait_N > 0
are recorded in `fired_columns` but never emitted; next turn the LP
re-derives from scratch and may pick a different set.

This test PINS the failure: across a real game, we expect at least
one turn where the LP fires a wait_N > 0 column AND the same
(src_id, tgt_id) is NOT in fired_columns at the next turn the agent
acts. That sequence == the evaporation failure mode.

After Phase C ships `commit_persistent`, this test gets a counterpart
that asserts the persistent-schedule variant carries the column
forward (i.e., the same (src, tgt) appears emitted at the original
fire_step). For now, this test ASSERTS the failure (Rule 38).
"""

from __future__ import annotations

import os

import pytest


def _run_game_capture_fired_columns(seed: int, opp_path: str) -> list[dict]:
    """Run analytical agent vs opp; per turn record fired_columns metadata."""
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
        decision = trace.get("decision")
        if ctx is None or ctx.is_empty_obs or ctx.is_no_targets:
            return moves
        opening = trace.get("opening")
        if opening is not None and opening.committed is not None:
            return moves
        fired = decision.fired_columns if decision else []
        per_turn.append({
            "step": ctx.step_now,
            "fired_columns": [
                {
                    "src_id": int(c.src_id), "tgt_id": int(c.tgt_id),
                    "wait_N": int(c.wait_N), "ships": int(c.ships),
                }
                for c in fired
            ],
            # wait_N==0 fires for this turn (the ones that actually emitted)
            "emitted_wait0": [
                {"src_id": int(c.src_id), "tgt_id": int(c.tgt_id)}
                for c in fired if int(c.wait_N) == 0
            ],
            "n_emitted_moves": len(moves),
        })
        return moves

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([wrapped, opp_path])
    return per_turn


@pytest.mark.parametrize("seed", [42])
def test_wait_n_evaporation_present_in_current_pipeline(seed: int):
    """PIN: under the stateless commit_stateless module, wait_N > 0 columns
    selected at turn t do not re-appear at turn t+1 for the same (src, tgt).

    This is the documented failure mode #1 from the Phase B plan. The
    test asserts that the failure IS present — flipping to PASSING happens
    in Phase C when `commit_persistent` is composed in.
    """
    per_turn = _run_game_capture_fired_columns(seed, "agents/baseline/main.py")

    # Build per-step indexes:
    #   wait_n_pos_fires_by_step: turns where wait_N>0 column was fired,
    #     keyed by step → list of (src, tgt, wait_N) tuples.
    #   emitted_wait0_pairs_by_step: turns where wait_N=0 column was emitted,
    #     keyed by step → set of (src, tgt) pairs.
    wait_n_pos_fires_by_step: dict[int, list] = {}
    emitted_wait0_pairs_by_step: dict[int, set] = {}
    for row in per_turn:
        step = row["step"]
        wait_n_pos = [
            (c["src_id"], c["tgt_id"], c["wait_N"])
            for c in row["fired_columns"] if c["wait_N"] > 0
        ]
        if wait_n_pos:
            wait_n_pos_fires_by_step[step] = wait_n_pos
        emitted_wait0_pairs_by_step[step] = {
            (c["src_id"], c["tgt_id"]) for c in row["emitted_wait0"]
        }

    # Evaporation: a wait_N=k column fired at turn t whose (src, tgt) pair
    # never appears as an EMITTED wait_N=0 move within turns [t+1, t+k+2].
    # The +2 slack accommodates LP re-derivation timing.
    evaporation_examples: list[tuple] = []
    for t, wait_pos_fires in sorted(wait_n_pos_fires_by_step.items()):
        for (src, tgt, k) in wait_pos_fires:
            window_end = t + k + 2
            emitted_in_window = False
            for t_next in range(t + 1, window_end + 1):
                if (src, tgt) in emitted_wait0_pairs_by_step.get(t_next, set()):
                    emitted_in_window = True
                    break
            if not emitted_in_window:
                evaporation_examples.append((t, src, tgt, k))

    n_wait_n_pos = sum(len(s) for s in wait_n_pos_fires_by_step.values())
    print(f"\nseed {seed}: {len(per_turn)} active turns, "
          f"{len(wait_n_pos_fires_by_step)} turns with wait_N>0 fires "
          f"({n_wait_n_pos} total wait_N>0 fires), "
          f"{len(evaporation_examples)} evaporations (never emitted as wait_N=0)")

    # Rule 38 pin: assert the failure mode IS present. The wait_N evaporation
    # — wait_N>0 columns the LP selected that never get their wait_N=0
    # decant — is the failure Phase C's commit_persistent closes.
    assert evaporation_examples, (
        f"Expected wait_N>0 evaporation under stateless commit, found none. "
        f"This is unexpected under commit_stateless; either Phase C has "
        f"already shipped or the game produced no wait_N>0 fires."
    )
    # Sanity: most wait_N>0 fires evaporate under stateless commit.
    evap_rate = len(evaporation_examples) / max(1, n_wait_n_pos)
    assert evap_rate >= 0.50, (
        f"Evaporation rate {evap_rate:.2f} unexpectedly low for stateless "
        f"commit. The failure mode should affect >50% of wait_N>0 fires."
    )
