"""v9_combined — v7_0 + K=15 + in-flight value head.

The full super-version: drop-one chassis (v7_0's structural-robustness
secret) with both v9 improvements stacked:
- K=15: longer rollout horizon catches eta-11-to-15 captures the
  ship-delta head would otherwise see as "in-flight cost only".
- inflight_value head: composite scoring that adds production credit
  for fleets predicted to capture within +30 turns past terminal.

If both v9_k15 and v9_inflight individually clear the Wilson 55%
gate vs v7_0, v9_combined is the submission target. Final-form agent.

NOT included (confirmed regressive or expensive):
- σ-equivariance (v7.6 bisect FAIL)
- 2×N maximin overlay (v7.1 FAIL — budget blow-up)
- Portfolio candidates (PI saw the pathology on v4_planner live)
- Recapture missions (v7.5 still regressed even after calibration)
- 4P-aware rollout (built but never gated; next session)
"""

from __future__ import annotations

from lib.v7_search import choose_simple_with_4p
from lib.value_heads import inflight_value


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=15,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=False,
        value_fn=inflight_value,
    )
