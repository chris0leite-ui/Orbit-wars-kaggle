"""Analytical agent — Phase D v2 (LP-seeded maximin).

Same as Phase D MVP but uses LP-seeded portfolio enumeration: maximin
considers the LP's chosen portfolio as one of its K_MY candidates,
plus the greedy-beam portfolios. This closes the suspected
"portfolio-enumeration gap" — cheap_delta beam search misses
joint-optimal portfolios (e.g., 3-fleet gang-ups) that the LP picks.

Stage choices:
  Stage 1: perception_default
  Stage 2: candidates_default
  Stage 3: prerank_passthrough
  Stage 4: opp_greedy_roi (perturbations live inside maximin)
  Stage 5: decision_maximin with my_portfolios_fn = LP-seeded enum
  Stage 6: leaf_outcome_table (inside Stage 5)
  Stage 7: commit_persistent
  Opening override: opening_default
"""

from __future__ import annotations

import os
from functools import partial

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision_maximin import decision_maximin
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.portfolio_enum_lp_seeded import (
    enumerate_top_k_portfolios_lp_seeded,
)
from lib.pipeline.prerank_passthrough import prerank_passthrough


_maximin_lp_seeded = partial(
    decision_maximin,
    my_portfolios_fn=enumerate_top_k_portfolios_lp_seeded,
)


_AGENT = compose(
    perception=perception_default,
    opening_override=opening_default,
    candidates=candidates_default,
    opp_model=opp_greedy_roi,
    prerank=prerank_passthrough,
    decision=_maximin_lp_seeded,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase D v2 analytical agent (LP-seeded maximin)."""
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
