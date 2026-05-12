"""v9_inflight — v7_0 with the in-flight value head.

Single targeted change vs v7_0_drop_one: scoring head goes from
`delta_us_minus_them` (terminal ship-delta) to `inflight_value`
(ship-delta + 0.5 × Σ production for in-flight fleets predicted-ours
at terminal+30 turns).

Hypothesis: the receding-horizon pathology (audit/2026-05-12-v4-
planner-receding-horizon-pathology.md) means that with K=10, fleets
with eta>10 are at the rollout's terminal still "in flight" — costing
ships but not yet credited as captures. The naive ship-delta head
prefers "noop" over "fire". The composite head reads the terminal
WorldModel out to +30 turns of static substrate and adds production
credit for predicted captures.

Everything else identical to v7_0.
"""

from __future__ import annotations

from lib.v7_search import choose_simple_with_4p
from lib.value_heads import inflight_value


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=False,
        value_fn=inflight_value,
    )
