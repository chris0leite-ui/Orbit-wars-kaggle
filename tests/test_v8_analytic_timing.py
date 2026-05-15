"""Timing budget for v8_analytic.

Targets (Phase A, CPU JAX on local CI):
- Cold-JIT turn 1: < 35 s. Dominated by XLA compile of the vmap'd
  `jax_step` + `value_with_future_production` graph (~18 s on CPU).
  The Kaggle production environment may compile faster on GPU; this
  test is a sanity floor, not a production budget.
- Warm turns 2+: < 3 s. With 16 beam calls × ~50 ms each.

This test catches order-of-magnitude regressions (a JIT recompile
every turn → 50+ s per turn). It does NOT certify the production
1000 ms turn budget — that belongs in a `fast.py eval` run with
Kaggle-equivalent hardware.
"""

from __future__ import annotations

import time

from kaggle_environments import make

from agents.v8_analytic.main import agent
from lib.foundation.agent_loop import reset_memory


def test_cold_then_warm_turn_timing():
    reset_memory()
    try:
        env = make("orbit_wars", configuration={"seed": 42}, debug=False)
        env.reset(num_agents=2)

        obs0 = env.state[0].observation

        t0 = time.perf_counter()
        action0 = agent(obs0, env.configuration)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        assert isinstance(action0, list)
        assert cold_ms < 35000.0, (
            f"cold-JIT turn 1 took {cold_ms:.0f} ms; expected < 35000 ms. "
            f"Likely a JIT compile blowup or a missing memoization."
        )

        # Step the env so turn 1 produces a different obs.
        env.step([action0, []])
        obs1 = env.state[0].observation

        t1 = time.perf_counter()
        action1 = agent(obs1, env.configuration)
        warm_ms = (time.perf_counter() - t1) * 1000.0
        assert isinstance(action1, list)
        assert warm_ms < 3000.0, (
            f"warm turn 2 took {warm_ms:.0f} ms; expected < 3000 ms. "
            f"Likely a JIT recompile (cache miss). cold was {cold_ms:.0f} ms."
        )

        # Intentionally NO `warm < cold * X` assertion — when this
        # test runs after another v8_analytic test in the same pytest
        # session, the JAX JIT cache is already warm and "cold" is
        # actually warm too, so the ratio assertion fires spuriously.
        # `reset_memory()` clears foundation memory but not JAX's
        # process-global compile cache. The absolute thresholds above
        # are what catch real regressions.
    finally:
        reset_memory()
