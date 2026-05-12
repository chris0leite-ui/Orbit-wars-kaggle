"""v7.2 — per-source ship-fraction sweep.

For each owned source, vary the ship count of its top-target launch
across {0.5, 0.95} of its garrison (v3.5.1's 0.7 is the incumbent).
Tests whether aggressive_fraction=0.7 is right for every turn or only
on average. Top-10 fingerprint shows both 0.5 (saturation) and 0.95
(concentrated artillery) are winning archetypes.
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="ship_sweep",
        K=10,
        wallclock_ms=700.0,
    )
