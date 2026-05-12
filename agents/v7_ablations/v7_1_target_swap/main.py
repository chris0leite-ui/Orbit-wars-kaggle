"""v7.1 — per-source runner-up target swap.

For each owned source that the incumbent launches from, generate one
sibling candidate where that source attacks its second-ranked snipe
target. The simplest additive lever — lets the rollout pick "B" when
the static ROI scorer ranked "A" first by a small margin.
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="target_swap",
        K=10,
        wallclock_ms=700.0,
    )
