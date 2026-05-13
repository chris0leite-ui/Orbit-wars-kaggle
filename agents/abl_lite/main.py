"""Ablation: v7_0_drop_one + LITE follow-up policy (~1-2 ms vs ~10 ms mirror).
All other levers identical to v7_0."""

from __future__ import annotations

from lib.v7_search import choose
from lib.opp_model import lite_greedy_policy


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        followup_policy=lite_greedy_policy,
    )
