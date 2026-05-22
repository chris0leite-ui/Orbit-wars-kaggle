"""Analytical agent — Phase C composition.

Same seven-stage pipeline as `agents/analytical/main.py`, but with two
stage swaps that close the documented failure modes:

  Stage 3: prerank_passthrough  (closes pre-filter amputation, #4)
  Stage 7: commit_persistent    (closes wait_N evaporation, #1)

Other stages (perception, candidates, opp_model, decision, opening
override) are unchanged — bit-parity with submission 52857903 at those
stages.

This agent is the Phase C deliverable. Validate vs `agents/baseline/main.py`
in a small A/B (n=4) before promoting to a Kaggle submission.
"""

from __future__ import annotations

import os

# Foundation hardening (PI 2026-05-21): enable orbital arrival safety
# in `WorldModel.time_to_enemy_threat` and `snipe._followon_hold_estimate`.
# The gate defaults OFF in the source for backwards compat with submitted
# sub 52872093; the bundled analytical agent always wants it ON because
# expected_hold and capture-EV scoring were silently mis-scoring orbiting
# captures. `setdefault` so an external caller can still override.
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# Level 1 topology features (PI 2026-05-21: "we need joint optimization
# that considers topology"). Three closed-form per-planet bonuses added
# to the LP's leaf value: reachability of nearby neutrals, mutual defense
# from clustered own planets, recapture-risk penalty for frontier planets.
# Computed once per turn from `lib.geo.sense.sense_state`. Set OFF to
# fall back to the pre-Level-1 objective (Phase 4 Step 1 + foundation
# only); per-feature toggles also available (LP_REACH_BONUS, etc.).
os.environ.setdefault("LP_TOPOLOGY_FEATURES", "1")

# Kinematic precomputation table (Phase γ of
# /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md). Replaces
# the per-call position-rebuild inside predict_fleet_fate with a
# per-turn-cached lookup. Bit-parity verified by 564 brute-force
# (FleetFate-level) and 2 full-game byte-identical assertions
# (seeds 42, 7); wall-clock saves 47-114 ms/step in measured runs.
# `setdefault` so external callers (tests, A/B harnesses) can still
# override.
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit_persistent import commit_persistent
from lib.pipeline.decision_depth2_search import decision_depth2_search
from lib.pipeline.decision_lagrangian_maximin import decision_lagrangian_maximin
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank_passthrough import prerank_passthrough


# Decision-stage router (Phase ε.1):
#   LP_MAXIMIN_SEARCH=1 → decision_lagrangian_maximin (adversarial maximin
#                          over top-K us × top-K opp, closed-form leaf).
#   LP_MAXIMIN_SEARCH=0/unset (default) → decision_depth2_search
#                          (opening-only T+1 lookahead with argmax).
# Routes via the LAZY `_maximin_enabled()` function from the maximin
# module — NOT direct os.environ.get — so hardcoded-constant variant
# bundles (scripts/build_topology_variants.py replaces the function
# body with `return True`/`return False`) correctly route through the
# router. Pre-fix the router read env-var directly, so hardcoded
# variants had no effect → n=16 A/B was measuring LP-vs-LP, not
# maximin (audit/2026-05-23/phase-epsilon-1-router-bug.md).
def _decision_router(cols, opp, ctx, **kw):
    from lib.pipeline.decision_lagrangian_maximin import _maximin_enabled
    if _maximin_enabled():
        return decision_lagrangian_maximin(cols, opp, ctx, **kw)
    return decision_depth2_search(cols, opp, ctx, **kw)


_AGENT = compose(
    perception=perception_default,
    opening_override=opening_default,
    candidates=candidates_default,
    opp_model=opp_greedy_roi,
    prerank=prerank_passthrough,
    decision=_decision_router,
    commit=commit_persistent,
)


def agent(obs, configuration=None):
    """Phase C analytical agent."""
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
