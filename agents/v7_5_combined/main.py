"""v7.5 — final combined (σ-equiv + recapture + 4P-aware, NO maximin).

v7.1's 2×N maximin matrix lost the A/B against v7_0 by 6/24 = 25%.
Diagnosis: 2×N cells × symmetric scoring (2× cost) = 4× more rollouts
per turn → 700 ms watchdog truncates → maximin defaults to incumbent
(conservative) → loses to v7_0's actual rollout-veto over a single
fixed opponent.

v7.5 drops the maximin overlay, keeps everything else:
- σ-equiv layer (lib/geometry sym_hypot + lib/planner _tb + SCORE_ROUND=6
  + lib/missions/snipe sym_hypot for distance). Library-level: free.
- Recapture mission class (calibrated: snipe-scale denom + top-K=5).
- 4P-aware drop-one rollout with 3 top_tier_mirror opps (replaces
  v7_0's "fall back to v3.5.1 in 4P").
- Single-rollout-per-candidate scoring (no maximin matrix; no
  symmetric scoring → no budget blow-up).
"""

from __future__ import annotations

from lib.v7_search import choose_simple_with_4p


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=True,
        value_fn=None,  # ship-delta head; A/B vs prod-head is v7.3
    )
