"""Stage 7 — Commit / schedule persistence.

DecisionResult → CommittedMoves.

The reference implementation is stateless: the moves emitted this turn
are exactly the wait_N==0 fires that the decision rule chose. wait_N>0
selections evaporate (this is the documented failure-mode #1 from the
plan; Phase C introduces `commit_persistent` to close it).
"""

from __future__ import annotations

from lib.pipeline.types import CommittedMoves, DecisionResult, TurnContext


def commit_stateless(decision: DecisionResult, ctx: TurnContext) -> CommittedMoves:
    """Reference Stage-7 implementation.

    The DecisionResult.moves already filtered to wait_N==0 inside
    `solve_outcome_aware` (lp_outcome.py:498-499). We just pass it through.
    """
    return CommittedMoves(moves=list(decision.moves), persisted_state=None)
