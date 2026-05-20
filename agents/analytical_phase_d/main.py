"""Analytical agent — Phase D composition (game-theoretic decision rule).

Builds on Phase C (commit_persistent + prerank_passthrough) with a swap
at Stage 5: depth-2 maximin over outcome-table leaves instead of
best-response-to-fixed-belief MILP. All closed-form, no rollouts.

Stage choices:
  Stage 1: perception_default
  Stage 2: candidates_default
  Stage 3: prerank_passthrough        (Phase C — closes amputation)
  Stage 4: opp_greedy_roi             (Phase C reference; Phase D MVP
                                       perturbations live inside decision)
  Stage 5: decision_maximin           (Phase D — depth-2 maximin over
                                       closed-form outcome-table leaves)
  Stage 6: leaf_outcome_table         (used internally by Stage 5)
  Stage 7: commit_persistent          (Phase C — closes evaporation)
  Opening override: opening_default

This agent is the Phase D MVP. Validate vs `agents/baseline/main.py`
(trajectory) and `agents/analytical_phase_c/main.py` (Phase C anchor)
in a small A/B before any further iteration.
"""

from __future__ import annotations

import os

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision_maximin import decision_maximin
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
    decision=decision_maximin,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase D analytical agent (maximin substrate)."""
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
