"""Ablation A1 — Phase C with only `no_launch` in the archetype panel.

Tests whether the joint-scoring layer (3-archetype min-regret) is the
regression vs Phase B.1. Min over 1 archetype = argmax of
value-against-opp-noop, so this variant strips the joint-scoring
layer and leaves the multi-turn enumeration + Tier-2 rollout as the
only added mechanism over Phase A.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.foundation.strategies import analytic_joint  # noqa: F401

from lib.foundation import StrategyCtx, get_strategy
from lib.foundation.agent_loop import (
    get_memory,
    maybe_reset_on_new_game,
    set_memory,
)
from lib.foundation.obs_to_state import my_id_from_obs, obs_to_jax_state


_STRATEGY_NAME = "v8_phase_c_no_panel"


import os as _os
if _os.environ.get("V8_ANALYTIC_PHASE_C_WARMUP", "1") != "0":
    try:
        analytic_joint.warmup_jits()
    except Exception:
        pass


def agent(obs: Any, configuration: Any = None) -> list[list]:
    maybe_reset_on_new_game(obs)
    memory = get_memory()

    state = obs_to_jax_state(obs, configuration=configuration)
    my_id = my_id_from_obs(obs)

    ctx = StrategyCtx(turn_budget_ms=1000.0, raw_obs=obs)
    strategy = get_strategy(_STRATEGY_NAME)
    action_tensor, new_memory = strategy.emit(state, my_id, ctx, memory)

    set_memory(new_memory)

    pids = np.asarray(action_tensor.pids[0, 0, my_id])
    angles = np.asarray(action_tensor.angles[0, 0, my_id])
    ships = np.asarray(action_tensor.ships[0, 0, my_id])
    out: list[list] = []
    for i in range(len(pids)):
        pid = int(pids[i])
        sh = int(ships[i])
        if pid < 0 or sh <= 0:
            continue
        out.append([pid, float(angles[i]), sh])
    return out
