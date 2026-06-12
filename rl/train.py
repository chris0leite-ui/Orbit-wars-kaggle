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

def eval_vs_greedy(key, params, pool, n_envs: int, n_steps: int = 520,
                   opp_name: str = "greedy"):
    """Seat 0 = policy, others = a scripted opponent. Returns win rate
    over envs that finished. Pool may be mixed 2P/4P — we report
    seat-0 reward over finished games."""
    from rl.scripted import SCRIPTED_POLICIES
    opp_fn = SCRIPTED_POLICIES[opp_name]
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
                tg, fr, sm = opp_fn(s, tables, seat)
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
    ap.add_argument("--eval-opp", type=str, default="greedy")
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--league", action="store_true",
                    help="half the envs train vs frozen snapshots/greedy")
    ap.add_argument("--snapshot-every", type=int, default=150)
    ap.add_argument("--snapshot-cap", type=int, default=12)
    ap.add_argument("--greedy-frac", type=float, default=0.15)
    ap.add_argument("--opp-refresh", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    cache_dir = os.path.join(args.out_dir, "jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

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

    # Initial env states: sample from pool. With --league, the batch is
    # split into a mirror half and a league half (persistent groups).
    B_league = args.batch // 2 if args.league else 0
    B_mirror = args.batch - B_league
    key, ks, ks2 = jax.random.split(key, 3)
    idx = jax.random.randint(ks, (B_mirror,), 0, n_pool)
    state = jax.tree.map(lambda x: x[idx], pool)
    state_lg = None
    if B_league:
        idx2 = jax.random.randint(ks2, (B_league,), 0, n_pool)
        state_lg = jax.tree.map(lambda x: x[idx2], pool)

    rollout_jit = jax.jit(rollout_chunk,
                          static_argnames=("n_steps", "opp_kind"))
    bootstrap_jit = jax.jit(bootstrap_values)
    eval_jit = jax.jit(eval_vs_greedy,
                       static_argnames=("n_envs", "n_steps", "opp_name"))

    snapshots = []  # league opponents: list of params pytrees (device)
    opp_pick = None
    opp_kind = "greedy"
    learner_seats = None

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

        key, k_roll, k_upd, k_lg = jax.random.split(key, 4)
        t0 = time.time()
        state, traj = rollout_jit(k_roll, params, state, pool,
                                  n_steps=args.rollout_steps)
        if B_league:
            if opp_pick is None or it % args.opp_refresh == 0:
                key, k1, k2 = jax.random.split(key, 3)
                use_scripted = (not snapshots or
                                float(jax.random.uniform(k1)) < args.greedy_frac)
                if use_scripted:
                    from rl.scripted import SCRIPTED_POLICIES
                    names = sorted(SCRIPTED_POLICIES)
                    j = int(jax.random.randint(k2, (), 0, len(names)))
                    opp_kind, opp_pick = names[j], params  # placeholder
                else:
                    opp_kind = "net"
                    j = int(jax.random.randint(k2, (), 0, len(snapshots)))
                    opp_pick = snapshots[j]
                key, k3 = jax.random.split(key)
                learner_seats = jax.random.randint(k3, (B_league,), 0, 2)
            state_lg, traj_lg = rollout_jit(
                k_lg, params, state_lg, pool,
                n_steps=args.rollout_steps, opp_params=opp_pick,
                learner_seats=learner_seats, opp_kind=opp_kind)
            traj = jax.tree.map(
                lambda a, b: jnp.concatenate([a, b], axis=1), traj, traj_lg)
        boot = bootstrap_jit(params, state)
        if B_league:
            boot_lg = bootstrap_jit(params, state_lg)
            boot = jnp.concatenate([boot, boot_lg], axis=0)
        t_roll = time.time() - t0

        t0 = time.time()
        params, opt_state, metrics = ppo_update(
            k_upd, params, opt_state, args.lr, traj, boot,
            n_minibatch=args.minibatches, n_epochs=args.epochs,
            entropy_coef=args.entropy_coef)
        metrics = jax.tree.map(float, metrics)
        t_upd = time.time() - t0

        if args.league and it % args.snapshot_every == 0:
            snapshots.append(jax.tree.map(jnp.copy, params))
            if len(snapshots) > args.snapshot_cap:
                snapshots.pop(1 if len(snapshots) > 2 else 0)

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
                                n_envs=args.eval_envs,
                                opp_name=args.eval_opp)
            rec[f"wr_vs_{args.eval_opp}"] = round(float(wr), 3)
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
