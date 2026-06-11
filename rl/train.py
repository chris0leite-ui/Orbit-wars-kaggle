"""PPO self-play training driver.

Usage (local smoke):
  python -m rl.train --pool data/rl_pool_smoke.npz --batch 16 \
      --rollout-steps 16 --iters 3 --out-dir /tmp/rl_smoke

Kaggle GPU (from the kernel script):
  python -m rl.train --pool /kaggle/input/.../rl_pool_train.npz \
      --batch 256 --rollout-steps 32 --hours 8 --out-dir /kaggle/working
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from lib.game.jax.jax_types import GameState, MAX_AGENTS, MAX_PLANETS
from rl import net
from rl.make_pool import load_pool
from rl.ppo import make_optimizer, ppo_update
from rl.rollout import bootstrap_values, rollout_chunk
from rl.scripted import greedy_policy
from rl.features import seat_features, state_tables
from rl.aim import solve_intercept_rows
from rl.rollout import actions_to_launch_tensors
from lib.game.jax.jax_interpreter import jax_step


def to_device_pool(pool_np) -> GameState:
    return jax.tree.map(jnp.asarray, pool_np)


def save_ckpt(path, params, meta):
    params_np = jax.tree.map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump({"params": params_np, "meta": meta}, f)


def load_ckpt(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return jax.tree.map(jnp.asarray, d["params"]), d.get("meta", {})


# ---------------- eval vs scripted (progress probe) ----------------

def eval_vs_greedy(key, params, pool, n_envs: int, n_steps: int = 520):
    """Seat 0 = policy, others = greedy. Returns win rate over envs
    that finished. 2P envs only make sense here; pool may be mixed —
    we just report seat-0 reward over finished games."""
    n_pool = int(pool.step.shape[0])
    idx = jax.random.randint(key, (n_envs,), 0, n_pool)
    state = jax.tree.map(lambda x: x[idx], pool)

    def step_fn(carry, k):
        st, finished, won = carry
        keys = jax.random.split(k, n_envs)

        def act_one(kk, s):
            tables = state_tables(s)
            n0, e0, g0, sm0, tm0 = seat_features(s, tables, 0)
            out = net.sample_actions(kk, params, n0, e0, g0,
                                     s.planets_alive, sm0, tm0)
            tgts = [out["tgt"]]
            fracs = [out["frac"]]
            masks = [sm0]
            for seat in range(1, MAX_AGENTS):
                tg, fr, sm = greedy_policy(s, tables, seat)
                tgts.append(tg); fracs.append(fr); masks.append(sm)
            return (jnp.stack(tgts), jnp.stack(fracs), jnp.stack(masks))

        tgt, frac, mask = jax.vmap(act_one)(keys, st)
        pids, angles, ships = jax.vmap(actions_to_launch_tensors)(
            st, tgt, frac, mask)
        nxt = jax.vmap(jax_step)(st, pids, angles, ships)
        just_done = nxt.done & ~finished
        won = won + jnp.where(
            just_done & (nxt.rewards[:, 0] > 0), 1.0, 0.0)
        finished = finished | nxt.done
        return (nxt, finished, won), None

    keys = jax.random.split(key, n_steps)
    (st, finished, won), _ = jax.lax.scan(
        step_fn, (state, jnp.zeros(n_envs, bool), jnp.zeros(n_envs)), keys)
    n_fin = jnp.maximum(jnp.sum(finished.astype(jnp.float32)), 1.0)
    return jnp.sum(won) / n_fin, jnp.sum(finished.astype(jnp.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--iters", type=int, default=0, help="0 = until --hours")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt-every-min", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-envs", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    print(f"devices: {jax.devices()}")
    pool = to_device_pool(load_pool(args.pool))
    n_pool = int(pool.step.shape[0])
    print(f"pool: {n_pool} init states")

    key = jax.random.PRNGKey(args.seed)
    if args.resume:
        params, meta = load_ckpt(args.resume)
        start_iter = int(meta.get("iter", 0))
        print(f"resumed from {args.resume} at iter {start_iter}")
    else:
        key, k0 = jax.random.split(key)
        params = net.init_params(k0)
        start_iter = 0
    print(f"params: {net.count_params(params):,}")

    optimizer = make_optimizer(args.lr)
    opt_state = optimizer.init(params)

    # Initial env states: sample from pool.
    key, ks = jax.random.split(key)
    idx = jax.random.randint(ks, (args.batch,), 0, n_pool)
    state = jax.tree.map(lambda x: x[idx], pool)

    rollout_jit = jax.jit(rollout_chunk, static_argnames=("n_steps",))
    bootstrap_jit = jax.jit(bootstrap_values)
    eval_jit = jax.jit(eval_vs_greedy, static_argnames=("n_envs", "n_steps"))

    t_start = time.time()
    t_last_ckpt = t_start
    it = start_iter
    env_steps = 0
    while True:
        it += 1
        if args.iters and it > start_iter + args.iters:
            break
        if not args.iters and (time.time() - t_start) > args.hours * 3600:
            break

        key, k_roll, k_upd = jax.random.split(key, 3)
        t0 = time.time()
        state, traj = rollout_jit(k_roll, params, state, pool,
                                  n_steps=args.rollout_steps)
        boot = bootstrap_jit(params, state)
        t_roll = time.time() - t0

        t0 = time.time()
        params, opt_state, metrics = ppo_update(
            k_upd, params, opt_state, args.lr, traj, boot,
            n_minibatch=args.minibatches, n_epochs=args.epochs)
        metrics = jax.tree.map(float, metrics)
        t_upd = time.time() - t0

        env_steps += args.batch * args.rollout_steps
        ep_done = float(jnp.sum(traj["done"].astype(jnp.int32)))
        rec = {
            "iter": it, "env_steps": env_steps,
            "t_roll_s": round(t_roll, 2), "t_upd_s": round(t_upd, 2),
            "episodes_done": ep_done,
            "wall_min": round((time.time() - t_start) / 60, 1),
            **{k: round(v, 5) for k, v in metrics.items()},
        }

        if args.eval_every and it % args.eval_every == 0:
            key, ke = jax.random.split(key)
            wr, nfin = eval_jit(ke, params, pool,
                                n_envs=args.eval_envs)
            rec["wr_vs_greedy"] = round(float(wr), 3)
            rec["eval_finished"] = int(nfin)

        line = json.dumps(rec)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

        now = time.time()
        if (now - t_last_ckpt) / 60 >= args.ckpt_every_min:
            t_last_ckpt = now
            p = os.path.join(args.out_dir, f"ckpt_{it:06d}.pkl")
            save_ckpt(p, params, {"iter": it, "env_steps": env_steps})
            save_ckpt(os.path.join(args.out_dir, "ckpt_latest.pkl"),
                      params, {"iter": it, "env_steps": env_steps})
            print(f"checkpoint -> {p}", flush=True)

    save_ckpt(os.path.join(args.out_dir, "ckpt_final.pkl"), params,
              {"iter": it, "env_steps": env_steps})
    print("done.")


if __name__ == "__main__":
    main()
