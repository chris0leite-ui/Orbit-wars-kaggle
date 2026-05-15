"""Kaggle entry point for `v8_analytic_phase_c` — Phase C.

Differences from `agents/v8_analytic/main.py`:
1. Imports `lib.foundation.strategies.analytic_joint` to register
   the Phase C strategy.
2. Selects `_STRATEGY_NAME = "v8_analytic_phase_c"`.
3. Passes the raw Kaggle obs through `StrategyCtx.raw_obs` — Phase C
   needs it to build the opp archetype panel (env-format obs required).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Importing the strategy module triggers `register_strategy(...)`.
from lib.foundation.strategies import analytic_joint  # noqa: F401

from lib.foundation import StrategyCtx, get_strategy
from lib.foundation.agent_loop import (
    get_memory,
    maybe_reset_on_new_game,
    set_memory,
)
from lib.foundation.obs_to_state import my_id_from_obs, obs_to_jax_state


_STRATEGY_NAME = "v8_analytic_phase_c"


def agent(obs: Any, configuration: Any = None) -> list[list]:
    """Kaggle-format agent. Returns
    `[[src_planet_id, aim_angle, ships], ...]`.
    """
    maybe_reset_on_new_game(obs)
    memory = get_memory()

    state = obs_to_jax_state(obs, configuration=configuration)
    my_id = my_id_from_obs(obs)

    ctx = StrategyCtx(turn_budget_ms=1000.0, raw_obs=obs)
    strategy = get_strategy(_STRATEGY_NAME)
    action_tensor, new_memory = strategy.emit(state, my_id, ctx, memory)

    set_memory(new_memory)

    return _tensor_to_env_action(action_tensor, my_id)


def _tensor_to_env_action(tensor, my_id: int) -> list[list]:
    """Extract action (`candidate=0`, `turn=0`) for seat `my_id` as
    `[[src_id, aim_angle, ships], ...]`. Skips empty slots."""
    pids = np.asarray(tensor.pids[0, 0, my_id])
    angles = np.asarray(tensor.angles[0, 0, my_id])
    ships = np.asarray(tensor.ships[0, 0, my_id])
    out: list[list] = []
    for i in range(len(pids)):
        pid = int(pids[i])
        sh = int(ships[i])
        if pid < 0 or sh <= 0:
            continue
        out.append([pid, float(angles[i]), sh])
    return out
