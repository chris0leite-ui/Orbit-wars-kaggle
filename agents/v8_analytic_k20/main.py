"""Phase B.1 with K=20 mirror rollout (was K=5).

Hypothesis: B.1's value head can't see fleet arrivals — Orbit Wars ETAs
are 20-80 turns; K=5 rollout terminates before launched fleets land.
Bumping K to 20 lets the rollout simulate most arrivals, so the head
correctly counts captured planets via `future_production`. Tests
whether the "capture/waste/defensibility" edge the iter-line value
heads showed is implicit in horizon depth, not in a new head shape.

Per-leaf cost scales ~linearly with K (one extra `jax_step` per scan
iteration); JIT compile time also grows. K=20 keeps depth modest enough
that the 1000 ms turn budget should hold; K=40 is left as a follow-up.

Smoke-first protocol (per `audit/friction.md`
`ab-strong-opp-before-smoke-against-floor`): run
`fast.py smoke agents/v8_analytic_k20` against random+nearest BEFORE
any A/B against B.1 / v7_0 / iter_v2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lib.foundation.strategies.analytic import AnalyticStrategy
from lib.foundation.strategy import register_strategy

from lib.foundation import StrategyCtx, get_strategy
from lib.foundation.agent_loop import (
    get_memory,
    maybe_reset_on_new_game,
    set_memory,
)
from lib.foundation.obs_to_state import my_id_from_obs, obs_to_jax_state


_STRATEGY_NAME = "v8_analytic_k20"

register_strategy(_STRATEGY_NAME, AnalyticStrategy(K=20))


def agent(obs: Any, configuration: Any = None) -> list[list]:
    maybe_reset_on_new_game(obs)
    memory = get_memory()

    state = obs_to_jax_state(obs, configuration=configuration)
    my_id = my_id_from_obs(obs)

    ctx = StrategyCtx(turn_budget_ms=1000.0)
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
