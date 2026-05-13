"""Ablation: v7_0_drop_one + WIDER candidates (enumerator=combined).
All other levers identical to v7_0."""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="combined",
        K=10,
    )
