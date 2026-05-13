"""v7_wide_deep — Phase 3c kitchen-sink agent.

Re-spends the compute headroom from Phase 3a/3b on four levers
simultaneously, layered on top of v7_0_drop_one's drop-one chooser:

  1. WIDER candidate set — `enumerator_mode="combined"` unions
     drop_one + target_swap + ship_sweep + archetype + hungarian,
     producing ~17 candidates vs v7_0's ~5.
  2. DEEPER lookahead — K=25 forward-sim steps (vs v7_0's K=10).
     Sees comet windows, recapture chains, multi-turn gang-ups.
  3. MAXIMIN over an opp pool — score each candidate against
     Tier-0 (v3_snipe defensive) AND Tier-1 (v3.5.1 aggressive);
     pick the candidate whose WORST-case score is highest. Robust
     to opp-policy uncertainty across the heterogeneous live ladder.
  4. RICHER value function — composite of ship-delta (baseline)
     and `evaluate_value` (production-share + denial + survivor
     bonus). Captures non-myopic value at the K-step horizon.

Per the audit, the v7_minimax / v7_1..v7_6 ablations all failed in
the prior compute budget (K=3). Phase 3a/3b headroom lets us run
maximin AND K=25 AND wider candidates simultaneously — the
combination the previous attempts couldn't afford.

Budget envelope (200 µs/step typical):
  17 candidates × 25 steps × 2 opp tiers × 200 µs = ~170 ms
  + enumeration ~5 ms + value-fn ~negligible
  = ~180 ms / turn (20 % of 1 s actTimeout)

Parity floor: always falls back to v3.5.1 incumbent if no candidate
strictly beats it (preserves the v7_0 safety net).
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import delta_us_minus_them_obs
from lib.lookahead_planner import evaluate_value


# Value-function weights. Start point; can be tuned via A/B.
_SHIP_DELTA_WEIGHT = 0.6
_RICH_VALUE_WEIGHT = 0.4


def _value_fn(observation, my_id):
    """Composite value head used by v7_wide_deep's rollout scorer.

    Blends the Phase-2-validated ship-delta with `evaluate_value`'s
    production-share + denial + survivor-bonus signal. The blend
    weighting (0.6 / 0.4) is the start point; Phase 3c A/B can
    re-tune.
    """
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
        K=25,
        opp_tiers=[0, 1],
        value_fn=_value_fn,
        wallclock_ms=800.0,
    )
