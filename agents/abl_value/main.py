"""Ablation: v7_0_drop_one + COMPOSITE value_fn (ship-delta + denial + survivor).
All other levers identical to v7_0."""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import delta_us_minus_them_obs
from lib.lookahead_planner import evaluate_value


def _value_fn(observation, my_id):
    d = delta_us_minus_them_obs(observation, my_id)
    e = evaluate_value(
        observation, my_id,
        denial_weight=0.4,
        ships_weight=0.05,
        survivor_bonus=5.0,
    )
    return 0.6 * d + 0.4 * e


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        value_fn=_value_fn,
    )
