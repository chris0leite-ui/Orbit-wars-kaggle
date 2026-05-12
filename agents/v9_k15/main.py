"""v9_k15 — v7_0 with K=15 (longer rollout horizon).

Single targeted change vs v7_0_drop_one: bump K from 10 to 15.
Hypothesis: K=10 leaves a blind spot for eta-11-to-15 captures.
fast_sim's 183× speedup makes K=15 affordable (budget: 5 candidates
× 15 steps × ~8 ms = 600 ms; well under the 700 ms watchdog).

Everything else identical to v7_0:
- Drop-one candidate enumeration (incumbent + drop-each-launch).
- v3.5.1 (top_tier_mirror) as Tier-1 opponent.
- ship-delta scoring head.
- 4P → v3.5.1 fallback.
- σ-equiv layer NOT included (reverted in lib/ post-v7.6).
"""

from __future__ import annotations

from lib.v7_search import choose_simple_with_4p


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=15,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=False,
        value_fn=None,  # default: ship-delta
    )
