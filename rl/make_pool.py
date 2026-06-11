"""Pre-generate stacked initial GameStates for training (and eval).

Usage:
  python -m rl.make_pool --n 1024 --out data/rl_pool_train.npz \
      --base-seed 100000 --p4-frac 0.25

The pool ships to Kaggle inside the code dataset so the training kernel
never needs kaggle_environments.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def build_pool(n: int, base_seed: int, p4_frac: float):
    import jax
    from kaggle_environments import make
    from lib.game.jax.conversions import scalar_to_jax

    states = []
    t0 = time.time()
    every_k = max(1, round(1.0 / p4_frac)) if p4_frac > 0 else 0
    for i in range(n):
        seed = base_seed + i
        num_agents = 4 if (every_k and i % every_k == every_k - 1) else 2
        env = make("orbit_wars", configuration={"seed": seed})
        # 4P games: env.reset with 4 agents
        if num_agents == 4:
            env.reset(num_agents=4)
        gs = scalar_to_jax(env.state, env.info["seed"])
        states.append(jax.tree.map(np.asarray, gs))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/game)")
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *states)
    return stacked


def save_pool(stacked, path: str):
    np.savez_compressed(path, **stacked._asdict())


def load_pool(path: str):
    """Load a pool npz back into a GameState of numpy arrays."""
    from lib.game.jax.jax_types import GameState
    z = np.load(path)
    return GameState(**{k: z[k] for k in GameState._fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--p4-frac", type=float, default=0.25)
    args = ap.parse_args()
    stacked = build_pool(args.n, args.base_seed, args.p4_frac)
    save_pool(stacked, args.out)
    print(f"saved {args.n} init states to {args.out}")


if __name__ == "__main__":
    main()
