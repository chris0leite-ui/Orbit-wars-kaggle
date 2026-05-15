"""Phase B.1 timing checks for `AnalyticStrategy` + `MissionMemory`.

Two assertions:

1. **Parity-with-overhead**: `AnalyticStrategy.emit` with a fresh
   `CompositeMemory` (no pre-commits) and `enable_chainer=True` must
   complete inside the same warm budget as Phase A (3 s on CPU; the
   chainer adds at most ~15 ms of `aim_orbiting` calls per turn,
   well below the noise floor of the JAX beam).

2. **Saturation skip is fast**: when a pre-committed mission covers
   every owned planet for the current seat, `emit` skips the beam
   entirely and completes inside 200 ms. This is the compute-win
   mechanism the success bar in the plan asks for.

Per-turn timing is inherently noisy on shared CI; the absolute
threshold of 200 ms is a 5× safety margin over the expected cost of
the prune + re-aim + tensor-pack path (~5-30 ms in local runs).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation import StrategyCtx
from lib.foundation.actions import ActionSpec
from lib.foundation.agent_loop import reset_memory
from lib.foundation.memory_impls import (
    CommittedMission,
    CompositeMemory,
    MissionMemory,
)
from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies import analytic  # noqa: F401
from lib.foundation.strategies.analytic import AnalyticStrategy


def _seed_state(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    return obs_to_jax_state(obs, configuration=env.configuration)


def _seat_planets(state, seat: int) -> list[int]:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    return [
        int(ids[i]) for i in range(len(alive))
        if bool(alive[i]) and int(owner[i]) == seat
    ]


def _enemy_target(state, seat: int) -> int:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    for i in range(len(alive)):
        if bool(alive[i]) and int(owner[i]) != seat:
            return int(ids[i])
    raise RuntimeError("no enemy/neutral target")


def test_parity_no_chainer_overhead():
    """With no pre-commits and chainer enabled, `emit` warm-turn
    completes inside the Phase A warm budget."""
    reset_memory()
    try:
        state = _seed_state()
        ctx = StrategyCtx(turn_budget_ms=1000.0)
        strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0)

        # Warm JIT.
        strat.emit(state, my_id=0, ctx=ctx, memory=CompositeMemory())

        # Measure warm turn.
        t0 = time.perf_counter()
        strat.emit(state, my_id=0, ctx=ctx, memory=CompositeMemory())
        warm_ms = (time.perf_counter() - t0) * 1000.0
        assert warm_ms < 3000.0, (
            f"warm `emit` took {warm_ms:.0f} ms with chainer enabled; "
            f"expected < 3000 ms (Phase A's warm budget)."
        )
    finally:
        reset_memory()


def test_saturated_turn_skips_beam_under_200ms():
    """A pre-committed wave covering the seat's only owned planet →
    `emit` skips `beam_search` and returns inside 200 ms.
    """
    reset_memory()
    try:
        state = _seed_state()
        seat0 = _seat_planets(state, 0)
        assert len(seat0) == 1, "seed-42 should give seat 0 exactly one planet at init"
        src = seat0[0]
        target = _enemy_target(state, 0)
        step = int(state.step)

        wave = ActionSpec(from_planet_id=src, dir_angle=0.0,
                          ships=3, launch_turn=0, agent_id=0)
        mission = CommittedMission(
            mission_id="sat",
            seat=0,
            turn_committed=step,
            target_planet_id=target,
            source_planet_id=src,
            waves=(wave,),
            waves_fired=0,
        )

        strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0)
        ctx = StrategyCtx(turn_budget_ms=1000.0)

        # Warm any JIT used by the re-aim path (aim_orbiting is pure
        # NumPy, but other state-conversion paths may JIT once).
        warm_mem = CompositeMemory().with_missions(
            MissionMemory(current_turn=step).commit(mission)
        )
        strat.emit(state, my_id=0, ctx=ctx, memory=warm_mem)

        # Fresh memory for the timed call (waves_fired must be 0).
        timed_mem = CompositeMemory().with_missions(
            MissionMemory(current_turn=step).commit(
                CommittedMission(
                    mission_id="sat2",
                    seat=0,
                    turn_committed=step,
                    target_planet_id=target,
                    source_planet_id=src,
                    waves=(wave,),
                    waves_fired=0,
                )
            )
        )
        t0 = time.perf_counter()
        strat.emit(state, my_id=0, ctx=ctx, memory=timed_mem)
        sat_ms = (time.perf_counter() - t0) * 1000.0
        assert sat_ms < 200.0, (
            f"saturated-skip turn took {sat_ms:.0f} ms; expected < 200 ms. "
            f"Likely the beam was called even though pre-commits saturated "
            f"all owned sources."
        )
    finally:
        reset_memory()
