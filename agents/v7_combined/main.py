"""v7_combined — union of every enumerator mode.

Final-form variant: enumerate ALL candidates across drop_one,
target_swap, ship_sweep, archetype, and hungarian, deduplicate by
action key, score each via fast_sim rollout, pick max.

Runs the largest candidate set, so the watchdog is more likely to
trip mid-sweep. The incumbent-first ordering inside `enumerate_
candidates(combined)` guarantees parity floor.

This is the agent we bundle for PI sign-off if any individual
variant (or this one) passes both 2P and 4P gates.
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="combined",
        K=10,
        wallclock_ms=700.0,
    )
