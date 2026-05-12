"""v7.3 — v7.2 + production-share scoring head.

Swaps the K-step rollout's leaf score from `our_ships - their_ships`
to `evaluate_value` (production-share + denial-bonus + ships-share +
sole-survivor-bonus). Ported from `lib/lookahead_planner.py` on
`origin/claude/research-lookahead-strategy-kfRsy`.

Hypothesis: the original ship-delta scoring head amplifies aggression
(at K=10 the launching seat's in-flight pile is at maximum — every
candidate that LAUNCHES more scores better than one that holds
garrison). The five v3-family loss patterns (elimination by step 158,
recovery deficit, in-flight-volume race) are all garrison-retention
wins that ship-delta can't reward.

`evaluate_value` instead measures production-share at the terminal —
"who owns the planets" rather than "who has more ships flying around."
"""

from __future__ import annotations

from lib.lookahead_planner import evaluate_value
from lib.v7_search import choose_maximin


def agent(obs, configuration=None):
    return choose_maximin(
        obs, configuration,
        K=10,
        wallclock_ms=700.0,
        use_symmetric=True,
        include_recapture=True,
        value_fn=evaluate_value,
    )
