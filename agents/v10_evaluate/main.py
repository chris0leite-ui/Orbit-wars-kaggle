"""v10_evaluate — v7_0 + evaluate_value head, no σ-equiv, no inflight.

The one untested obvious super-version: drop-one chassis with v4_planner's
`evaluate_value` scoring head (production-share + denial-bonus +
ships-share + sole-survivor) replacing the default ship-delta head.

Why this is the last "easy gain" we haven't tested:
- v8_minimal had this same scoring head but ALSO carried σ-equiv
  (which v7.6 bisect proved costs ~54 pp). v8_minimal scored +17 pp
  NEUTRAL vs v7_0 — implying the head's CONTRIBUTION net of σ-equiv's
  drag is ≥ +17 pp.
- σ-equiv is now reverted in lib/ (post-v7.6). A fresh bundle of
  "drop-one + evaluate_value" inherits the strip.
- If the +17 pp contribution survives without σ-equiv's drag,
  Wilson lo should clear the 55% gate.
- If it doesn't, ship v7_0.

This is the simplest, most targeted test of "does the production-share
head help drop-one?"
"""

from __future__ import annotations

from lib.lookahead_planner import evaluate_value
from lib.v7_search import choose_simple_with_4p


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=False,
        value_fn=evaluate_value,
    )
