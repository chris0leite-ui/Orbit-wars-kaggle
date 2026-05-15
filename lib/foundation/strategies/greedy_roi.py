"""Greedy-ROI Strategy adapter — Phase 0 smoke.

Wraps `agents/simple/roi.py`'s pre-existing ROI heuristic in the
foundation `Strategy` Protocol. Purpose: end-to-end smoke test of the
pipeline (Kaggle obs → JAX state → Strategy.emit → ActionTensor →
env action) without inventing new heuristic logic. Plays a single-
launch-per-turn cost-aware ROI policy; loses to v7_0 by design.

Importing this module registers the strategy under the name
`v8_greedy_roi` via `lib.foundation.register_strategy`.
"""

from __future__ import annotations

import math
from typing import Any

from agents.simple import roi as _simple_roi
from lib.foundation.actions import ActionSpec, ActionTensor, specs_to_tensor
from lib.foundation.memory import Memory
from lib.foundation.strategy import (
    Strategy,
    StrategyCtx,
    register_strategy,
)
from lib.game.jax.conversions import jax_to_scalar_obs
from lib.game.jax.jax_types import GameState


class GreedyRoiStrategy:
    """Thin Strategy wrapper around `agents/simple/roi.py::agent`.

    The wrapped agent already runs `realize(intents, obs, mechanisms=
    DEFAULT_MECHANISMS)` — meaning lead-aim, arrival-size, sun-avoid,
    OOB-guard, and the rest of the env-mirror mechanism pipeline are
    all applied. We just convert its env-format output to an
    `ActionTensor`.
    """

    name = "v8_greedy_roi"

    def emit(
        self,
        state: GameState,
        my_id: int,
        ctx: StrategyCtx,
        memory: Memory,
    ) -> tuple[ActionTensor, Memory]:
        # Reconstruct a scalar obs from the JAX state. Slightly
        # wasteful (we already had an obs upstream of obs_to_jax_state)
        # but keeps the Strategy protocol state-only / pure.
        scalar_obs = jax_to_scalar_obs(state)
        scalar_obs["player"] = int(my_id)

        # Some downstream callees in roi expect a `comet_planet_ids`
        # field; jax_to_scalar_obs doesn't include it. Reconstruct
        # from `is_comet` mask + `planets_id`.
        comet_pids = _comet_planet_ids_from_state(state)
        scalar_obs.setdefault("comet_planet_ids", list(comet_pids))
        scalar_obs.setdefault("comets", [])  # roi only needs lifetimes; empty list is safe
        scalar_obs.setdefault("initial_planets", scalar_obs["planets"])

        env_actions = _simple_roi.agent(scalar_obs)

        # env_actions is a list of [src_id, aim_angle, ships] triples.
        specs = [
            ActionSpec(
                from_planet_id=int(a[0]),
                dir_angle=float(a[1]),
                ships=int(a[2]),
                launch_turn=0,
                agent_id=my_id,
            )
            for a in (env_actions or [])
            if len(a) == 3 and int(a[2]) > 0
        ]

        # ActionTensor expects a non-empty candidate list even if no
        # specs (the candidate is "do nothing this turn"). horizon=1
        # because the smoke fires only on the current turn.
        tensor = specs_to_tensor([specs], horizon=1)
        return tensor, memory


def _comet_planet_ids_from_state(state: GameState) -> list[int]:
    """Extract the list of comet planet ids from `state.is_comet`."""
    import numpy as np
    is_comet = np.asarray(state.is_comet)
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    return [int(ids[i]) for i in range(len(alive)) if alive[i] and is_comet[i]]


# Register on import so `agents/v8_greedy_roi/main.py` can look up by name.
register_strategy("v8_greedy_roi", GreedyRoiStrategy())
