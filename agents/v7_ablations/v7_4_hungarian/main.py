"""v7.4 — Hungarian global (source × target) assignment.

Generates one additional sibling candidate: the bipartite-optimal
assignment of sources to targets via `scipy.optimize.linear_sum_
assignment`. settle_plan's per-source greedy + same-turn ledger is
*locally* greedy; this candidate offers a *globally* coordinated
alternative.

If scipy is unavailable in the bundle environment, falls back to the
incumbent (the choose() framework's parity floor still applies).
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="hungarian",
        K=10,
        wallclock_ms=700.0,
    )
