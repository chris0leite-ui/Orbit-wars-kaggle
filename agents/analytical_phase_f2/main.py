"""Analytical agent — Phase F2a (production-feedback compound candidates).

Closes the action-space blind spot identified after Phases C, D v2,
D v3, F1 all returned 1/4 vs trajectory on identical seeds:

  The LP couldn't enqueue "fire from a planet I'd capture mid-
  horizon." Captured planet's prod_stream value accrued in the
  objective, but those ships were locked out of the action space.
  Trajectory chooser does this implicitly via K-step rollout.

Phase F2a adds compound candidates: for each base capture (src, tgt
opp-owned, arrival within max_lookahead), generate "fire from tgt"
candidates targeting other opp planets. Each compound column has
parent_column_id linking it to the base capture. The LP enforces
`x_compound <= x_parent` via a linkage constraint.

Phase F2b (deferred): opp's symmetric blind spot — opp's compound
launches from planets opp captures from me. Without F2b, F2a is
asymmetric (we see our compound advantage but not opp's). Build F2b
once F2a's signal is clear.

Stage choices:
  Stage 1: perception_default
  Stage 2: candidates_default
  Stage 3: prerank_with_production_feedback  (PHASE F2a)
  Stage 4: opp_greedy_roi
  Stage 5: decision_outcome_aware_milp        (uses linkage-aware
                                              solve_outcome_aware)
  Stage 7: commit_persistent
  Opening override: opening_default
"""

from __future__ import annotations

import os

# Kinematic precomputation table — see plan
# /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md and
# agents/analytical_phase_c/main.py. Bit-parity gated.
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank_with_production_feedback import (
    prerank_with_production_feedback,
)


_AGENT = compose(
    perception=perception_default,
    opening_override=opening_default,
    candidates=candidates_default,
    opp_model=opp_greedy_roi,
    prerank=prerank_with_production_feedback,
    decision=decision_outcome_aware_milp,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase F2a analytical agent (production-feedback compound candidates)."""
    prev_drain = os.environ.get("PROPOSER_DRAIN_FILTER")
    prev_hold = os.environ.get("PROPOSER_HOLD_FEASIBILITY")
    os.environ["PROPOSER_DRAIN_FILTER"] = "off"
    os.environ["PROPOSER_HOLD_FEASIBILITY"] = "off"
    try:
        return _AGENT(obs, configuration)
    finally:
        if prev_drain is None:
            os.environ.pop("PROPOSER_DRAIN_FILTER", None)
        else:
            os.environ["PROPOSER_DRAIN_FILTER"] = prev_drain
        if prev_hold is None:
            os.environ.pop("PROPOSER_HOLD_FEASIBILITY", None)
        else:
            os.environ["PROPOSER_HOLD_FEASIBILITY"] = prev_hold
