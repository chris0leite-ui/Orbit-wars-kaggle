"""Analytical agent — Phase D v3 (Stackelberg-leader + mirror-analytical opp).

The first composition with a TRULY action-reactive opp model: for each
of my candidate portfolios, opp's best-response is computed by running
the analytical LP from opp's POV with my arrivals merged in. Different
my portfolios → different opp responses → real game-theoretic content
in the decision rule.

Stage choices:
  Stage 1: perception_default
  Stage 2: candidates_default
  Stage 3: prerank_passthrough            (Phase C close)
  Stage 4: opp_greedy_roi                  (pipeline-wide; reactive opp
                                            invoked inside Stage 5)
  Stage 5: decision_stackelberg_leader with LP-seeded my_portfolios
  Stage 6: leaf_outcome_table (inside Stage 5)
  Stage 7: commit_persistent               (Phase C close)
  Opening override: opening_default
"""

from __future__ import annotations

import os
from functools import partial

# Kinematic precomputation table — see plan
# /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md and
# agents/analytical_phase_c/main.py. Bit-parity gated.
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision_stackelberg_leader import decision_stackelberg_leader
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.portfolio_enum_lp_seeded import (
    enumerate_top_k_portfolios_lp_seeded,
)
from lib.pipeline.prerank_passthrough import prerank_passthrough


_stackelberg_lp_seeded = partial(
    decision_stackelberg_leader,
    my_portfolios_fn=enumerate_top_k_portfolios_lp_seeded,
)


_AGENT = compose(
    perception=perception_default,
    opening_override=opening_default,
    candidates=candidates_default,
    opp_model=opp_greedy_roi,
    prerank=prerank_passthrough,
    decision=_stackelberg_lp_seeded,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase D v3 analytical agent (Stackelberg-leader + reactive opp)."""
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
