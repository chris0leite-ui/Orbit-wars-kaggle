"""Analytical agent — Phase F1 (discounted leaf + truncated horizon).

Phase D analysis showed the analytical-native ceiling vs trajectory sat
at ~1/4 across three structurally different decision rules
(best-response LP, LP-seeded maximin, Stackelberg-leader with reactive
opp). The diagnosis pointed at the LEAF, not the decision rule:
outcome_table sums production undiscounted to T_END=500 even though
games end at step 100-200, and value_for_candidate uses γ=0.99 over
pv_horizon — the LP's objective and the pre-rank value are on
inconsistent scales.

Phase F1 closes that gap. Same Phase C plumbing, but the LP's leaf
reads γ=0.99-discounted prod_stream summed to step_now+200.

Stage choices:
  Stage 1: perception_default
  Stage 2: candidates_default
  Stage 3: prerank_passthrough          (Phase C)
  Stage 4: opp_greedy_roi               (unchanged baseline)
  Stage 5: decision_outcome_aware_discounted  (PHASE F1)
  Stage 7: commit_persistent            (Phase C)
  Opening override: opening_default
"""

from __future__ import annotations

import os

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision_outcome_aware_discounted import (
    decision_outcome_aware_discounted,
)
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank_passthrough import prerank_passthrough


_AGENT = compose(
    perception=perception_default,
    opening_override=opening_default,
    candidates=candidates_default,
    opp_model=opp_greedy_roi,
    prerank=prerank_passthrough,
    decision=decision_outcome_aware_discounted,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase F1 analytical agent."""
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
