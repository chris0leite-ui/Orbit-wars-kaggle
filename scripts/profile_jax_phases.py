"""Microbench: JAX phases (JIT'd, vmap'd) vs scalar equivalents.

Measures the JAX sub-phase 1b/1c progress in three workloads:

1. Per-step JAX-only phase cost (after JIT compile) on a single game
   state. Demonstrates JIT inlining/fusion benefit.

2. Batched (N=64 games) phase cost via jax.vmap. Demonstrates the
   vectorisation pay-off — the kernel that makes Kaggle GPU runs hit
   the 5-min A/B target.

3. swept_pair_hit_batch (F × P broadcast) at typical lookahead sizes:
   F=50 fleets, P=30 planets. Cumulative cost of the hot loop.

We do NOT compare against pure Python equivalents here (some phases
don't have a clean isolated scalar form). The scalar-vs-JAX
end-to-end comparison is the parity test; this is just a SANITY bench
for "does JIT pay off".
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from kaggle_environments import make

from lib.game.jax import (
    GameState,
    scalar_to_jax,
    production_tick_jit,
    planet_path_compute_jit,
    comet_expire_jit,
    comet_path_advance_jit,
    comet_spawn_jit,
    swept_pair_hit_batch_jit,
)


def _build_n_states(n: int):
    """Build N fresh game states with distinct seeds."""
    states = []
    for s in range(n):
        env = make("orbit_wars", configuration={"seed": s})
        env.reset(num_agents=2)
        states.append(scalar_to_jax(env.state, env.info["seed"]))
    return states


def _stack(states: list[GameState]) -> GameState:
    """Stack a list of GameState into a (N, ...) batched GameState."""
    fields = {}
    for name in states[0]._fields:
        arrs = [getattr(s, name) for s in states]
        fields[name] = jnp.stack(arrs)
    return GameState(**fields)


def _time(fn, label: str, warmup: int = 3, runs: int = 20):
    """Time a callable. JIT warmup + N runs; report median per-call ms."""
    for _ in range(warmup):
        out = fn()
        if isinstance(out, GameState):
            out.planets_x.block_until_ready()
        else:
            out.block_until_ready()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, GameState):
            out.planets_x.block_until_ready()
        else:
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    times.sort()
    median_ms = times[len(times) // 2] * 1000
    print(f"  {label:55s} median={median_ms:6.2f} ms/call")
    return median_ms


def bench_n1():
    print("\n=== N=1 per-phase JIT cost (single game) ===")
    states = _build_n_states(1)
    gs = states[0]

    _time(lambda: production_tick_jit(gs), "production_tick (N=1)")
    _time(lambda: planet_path_compute_jit(gs), "planet_path_compute (N=1)")
    _time(lambda: comet_expire_jit(gs), "comet_expire (N=1)")
    _time(lambda: comet_path_advance_jit(gs), "comet_path_advance (N=1)")
    # comet_spawn at step=49 to trigger fire path
    gs_49 = gs._replace(step=jnp.asarray(49, dtype=jnp.int32))
    _time(lambda: comet_spawn_jit(gs_49), "comet_spawn (N=1, step=49)")


def bench_n64():
    print("\n=== N=64 batched (vmap) per-phase cost ===")
    states = _build_n_states(64)
    batch = _stack(states)

    prod_v = jax.jit(jax.vmap(production_tick_jit.__wrapped__))
    ppc_v = jax.jit(jax.vmap(planet_path_compute_jit.__wrapped__))
    exp_v = jax.jit(jax.vmap(comet_expire_jit.__wrapped__))
    adv_v = jax.jit(jax.vmap(comet_path_advance_jit.__wrapped__))

    _time(lambda: prod_v(batch), "production_tick (N=64 vmap)")
    _time(lambda: ppc_v(batch), "planet_path_compute (N=64 vmap)")
    _time(lambda: exp_v(batch), "comet_expire (N=64 vmap)")
    _time(lambda: adv_v(batch), "comet_path_advance (N=64 vmap)")


def bench_swept_pair():
    print("\n=== swept_pair_hit_batch (F × P broadcast) ===")
    for F, P in [(10, 20), (50, 30), (150, 60)]:
        fold = jnp.asarray(np.random.uniform(0, 100, size=(F, 2)).astype(np.float32))
        fnew = fold + jnp.asarray(np.random.uniform(-5, 5, size=(F, 2)).astype(np.float32))
        pold = jnp.asarray(np.random.uniform(0, 100, size=(P, 2)).astype(np.float32))
        pnew = pold + jnp.asarray(np.random.uniform(-0.5, 0.5, size=(P, 2)).astype(np.float32))
        pr = jnp.asarray(np.random.uniform(1, 4, size=(P,)).astype(np.float32))
        _time(
            lambda: swept_pair_hit_batch_jit(fold, fnew, pold, pnew, pr),
            f"swept_pair_hit_batch (F={F}, P={P})",
        )


def main():
    bench_n1()
    bench_n64()
    bench_swept_pair()


if __name__ == "__main__":
    main()
