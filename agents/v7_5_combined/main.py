"""v7.5 — final combined agent (σ-equiv + maximin + recapture + 4P).

Stacks every load-bearing improvement from v7.1–v7.4 into a single
agent. Final entry point for PI submit authorisation.

What's stacked:
- σ-equivariance (sym_hypot in lib/geometry + _tb tie-break +
  SCORE_ROUND=6 in lib/planner). Audit attributes ~+45 μ over v3.4
  baseline to σ-equiv alone.
- Symmetric scoring (`score_joint_symmetric`) — cancels env P1-bias
  in 2P rollouts.
- 2×2 maximin overlay in 2P — worst-case-best over opp class
  {v3.5.1, drop-smallest(v3.5.1)}.
- Recapture mission class with snipe-scale denominator + top-K cap
  (fixes the regressions documented in
  audit/2026-05-12-recapture-wireup-ab.md).
- 4P-aware drop-one rollout (`score_candidate_4p`) — no more
  pass-through-to-v3.5.1 on the missing half of the ladder. Uses
  best-remaining-opp scoring head.

The scoring head defaults to `delta_us_minus_them`; v7.3's
`evaluate_value` (production-share) is ALSO available and gets
A/B'd separately. v7.5 picks whichever ships PASS.
"""

from __future__ import annotations

from lib.v7_search import choose_with_4p


def agent(obs, configuration=None):
    return choose_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        use_symmetric=True,
        include_recapture=True,
        value_fn=None,   # default = delta_us_minus_them; v7.5 picks ship-delta unless v7.3 PASS swings it
    )
