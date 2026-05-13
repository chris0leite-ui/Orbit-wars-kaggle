"""Ablation: v7_0_drop_one + MAXIMIN over opp pool (Tier 0 + Tier 1).
All other levers identical to v7_0."""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        opp_tiers=[0, 1],
    )
