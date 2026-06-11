"""Benchmark vmapped jax_step throughput for RL training sizing.

Usage: python -m rl.bench_engine [batch_size]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from kaggle_environments import make

from lib.game.jax.conversions import scalar_to_jax
from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


def build_pool(n_games: int, base_seed: int = 1000, num_agents: int = 2):
    """Pre-generate n initial GameStates via the scalar engine, stacked."""
    states = []
    t0 = time.time()
    for i in range(n_games):
        env = make("orbit_wars", configuration={"seed": base_seed + i})
        gs = scalar_to_jax(env.state, env.info["seed"])
        states.append(gs)
    init_time = time.time() - t0
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    return stacked, init_time


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print(f"devices: {jax.devices()}")

    n_pool = min(batch, 16)  # pool init is slow; tile to batch size
    pool, init_time = build_pool(n_pool)
    print(f"pool init: {n_pool} games in {init_time:.1f}s "
          f"({init_time / n_pool * 1000:.0f} ms/game)")

    reps = batch // n_pool
    state = jax.tree.map(lambda x: jnp.concatenate([x] * reps), pool)

    # Random-ish fixed actions: every agent launches 2 fleets/turn.
    rng = np.random.default_rng(0)
    pids = -np.ones((batch, MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = rng.uniform(0, 2 * np.pi,
                         (batch, MAX_AGENTS, MAX_LAUNCH_PER_AGENT)).astype(np.float32)
    ships = np.zeros((batch, MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    pids[:, :2, :2] = rng.integers(0, 40, (batch, 2, 2))
    ships[:, :2, :2] = rng.integers(1, 5, (batch, 2, 2))
    a_pid = jnp.asarray(pids)
    a_angle = jnp.asarray(angles)
    a_ships = jnp.asarray(ships)

    step_v = jax.jit(jax.vmap(jax_step))

    t0 = time.time()
    state2 = step_v(state, a_pid, a_angle, a_ships)
    jax.block_until_ready(state2.planets_ships)
    compile_time = time.time() - t0
    print(f"compile (batch={batch}): {compile_time:.1f}s")

    n_steps = 50
    t0 = time.time()
    s = state
    for _ in range(n_steps):
        s = step_v(s, a_pid, a_angle, a_ships)
    jax.block_until_ready(s.planets_ships)
    dt = time.time() - t0
    env_steps_per_s = batch * n_steps / dt
    print(f"steady: {dt / n_steps * 1000:.1f} ms/step (batch={batch}) "
          f"= {env_steps_per_s:,.0f} env-steps/s")
    print(f"games/hour (500 steps): {env_steps_per_s / 500 * 3600:,.0f}")


if __name__ == "__main__":
    main()
