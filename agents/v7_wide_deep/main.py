"""v7_wide_deep — Phase 3c agent, budget-aware kitchen-sink.

Re-spends the compute headroom from Phase 3a/3b on FOUR decision-quality
levers layered on top of v7_0_drop_one (live μ=1094.9). Per the profile
(audit 2026-05-13), the agent-inference bottleneck is the opp-policy
itself (~10 ms / call building WorldModel every step), not the physics.
We work around that two ways:

  - SHARED WorldModel across the two seats' opp_policy calls per step
    (~3.8 ms saved per step). Implemented in `lib.v7_search.score_candidate`
    and `lib.opp_model.{top_tier_mirror,mirror_self}_policy`.
  - LITE follow-up policy (`lib.opp_model.lite_greedy_policy`) for the
    K-1 steps after step 0 — ROI-greedy, no WorldModel, ~1-2 ms / call.

Levers vs v7_0_drop_one:

  1. WIDER candidate set. `enumerator_mode="combined"` unions
     drop_one + target_swap + ship_sweep + archetype + hungarian
     (~17 candidates vs v7_0's ~5). Addresses the "narrow enumeration"
     gap — top-10 players launch 3.5× more often per game.
  2. DEEPER lookahead. K=15 vs v7_0's K=10. Sees one more comet
     capture / recapture cycle.
  3. MAXIMIN over opp pool. Score each candidate against Tier-0
     (v3_snipe defensive) AND Tier-1 (v3.5.1 aggressive) at step 0;
     pick the candidate whose worst-case score is highest. Robust to
     opp-policy uncertainty across the heterogeneous live ladder.
  4. RICHER value function. Composite of ship-delta (baseline) and
     `evaluate_value` (production-share + denial + survivor bonus).

Measured turn timing (80-turn self-play vs v3.5.1, this CPU):
  mean=156 ms, p50=179 ms, p95=461 ms, max=597 ms.
  v7_0 baseline same workload: mean=73, p50=4, p95=342, max=472.

Parity floor: always falls back to v3.5.1 incumbent if no candidate
strictly beats it.
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import delta_us_minus_them_obs
from lib.lookahead_planner import evaluate_value
from lib.opp_model import lite_greedy_policy


_SHIP_DELTA_WEIGHT = 0.6
_RICH_VALUE_WEIGHT = 0.4


def _value_fn(observation, my_id):
    """Composite value head: ship-delta + production/denial/survivor."""
    d = delta_us_minus_them_obs(observation, my_id)
    e = evaluate_value(
        observation, my_id,
        denial_weight=0.4,
        ships_weight=0.05,
        survivor_bonus=5.0,
    )
    return _SHIP_DELTA_WEIGHT * d + _RICH_VALUE_WEIGHT * e


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="combined",
        K=15,
        opp_tiers=[0, 1],
        value_fn=_value_fn,
        followup_policy=lite_greedy_policy,
        wallclock_ms=700.0,
    )
