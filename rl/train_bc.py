"""Pretrain a fast neural producer-CLONE by pure behavior cloning.

The clone is a frozen league opponent: a policy net (same architecture
as the learner) trained purely to imitate producer's target+fraction
choices. Putting it in the RL opponent pool lets the learner train to
BEAT producer-style coordinated pressure at GPU speed — the AlphaStar
"supervised agents seed the league" move.

Usage:
  python -m rl.train_bc --bc-npz data/bc_samples.npz \
      --steps 4000 --out-dir /tmp/bc_clone
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.game.jax.jax_types import MAX_AGENTS, MAX_PLANETS
from rl import net
from rl.bc_data import load_bc_npz
from rl.features import state_tables, seat_features
from rl.ppo import bc_loss_feats, featurize_bc
from rl.train import save_ckpt


def make_opt(lr):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))


@partial(jax.jit, static_argnames=("optimizer",))
def bc_step(params, opt_state, optimizer, gs, tgt, frac, mask, seat):
    # Featurize OUTSIDE the grad (aim solver / arrival scan are
    # param-independent): keeps them off the backward pass so each step
    # is just a forward+backward through the 120k-param net.
    feats = jax.vmap(featurize_bc)(gs, seat)
    loss, grads = jax.value_and_grad(bc_loss_feats)(
        params, feats, tgt, frac, mask)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def eval_vs_greedy(params, pool, n_envs=48, n_steps=510, seed=7):
    """Clone (seat 0/2) vs greedy (seat 1/3). Returns clone win rate +
    a pressure proxy (mean peak fleets the clone puts in flight)."""
    from rl.scripted import greedy_policy
    from rl.rollout import actions_to_launch_tensors
    from lib.game.jax.jax_interpreter import jax_step

    n_pool = int(pool.step.shape[0])
    idx = jax.random.randint(jax.random.PRNGKey(seed), (n_envs,), 0, n_pool)
    state = jax.tree.map(lambda x: x[idx], pool)

    def step_fn(carry, k):
        st, fin, won, peak = carry
        keys = jax.random.split(k, n_envs)

        def act_one(kk, s):
            tb = state_tables(s)
            outs = []
            for seat in range(MAX_AGENTS):
                if seat % 2 == 0:
                    n0, e0, g0, sm0, tm0 = seat_features(s, tb, seat)
                    o = net.sample_actions(kk, params, n0, e0, g0,
                                           s.planets_alive, sm0, tm0)
                    outs.append((o["tgt"], o["frac"], sm0))
                else:
                    tg, fr, sm = greedy_policy(s, tb, seat)
                    outs.append((tg, fr, sm))
            tg = jnp.stack([o[0] for o in outs])
            fr = jnp.stack([o[1] for o in outs])
            mk = jnp.stack([o[2] for o in outs])
            return tg, fr, mk

        tg, fr, mk = jax.vmap(act_one)(keys, st)
        pids, angs, shp = jax.vmap(actions_to_launch_tensors)(st, tg, fr, mk)
        nxt = jax.vmap(jax_step)(st, pids, angs, shp)
        n_fleet0 = jnp.sum((nxt.fleets_owner == 0) & nxt.fleets_alive, axis=1)
        peak = jnp.maximum(peak, n_fleet0.astype(jnp.float32))
        jd = nxt.done & ~fin
        won = jnp.where(jd & (nxt.rewards[:, 0] > 0), 1.0, won)
        return (nxt, fin | nxt.done, won, peak), None

    init = (state, jnp.zeros(n_envs, bool), jnp.zeros(n_envs),
            jnp.zeros(n_envs))
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_steps)
    (st, fin, won, peak), _ = jax.lax.scan(jax.jit(step_fn), init, keys)
    return (float(jnp.sum(won)) / max(float(jnp.sum(fin)), 1.0),
            float(jnp.mean(peak)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-npz", type=str, required=True)
    ap.add_argument("--pool", type=str, default="data/rl_pool_train.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"devices: {jax.devices()}", flush=True)

    gs, tgt, frac, mask, seat = load_bc_npz(args.bc_npz)
    gs = jax.tree.map(jnp.asarray, gs)
    tgt = jnp.asarray(tgt); frac = jnp.asarray(frac)
    mask = jnp.asarray(mask); seat = jnp.asarray(seat)
    n = int(tgt.shape[0])
    print(f"BC samples: {n}", flush=True)

    from rl.make_pool import load_pool
    pool = jax.tree.map(jnp.asarray, load_pool(args.pool))

    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    params = net.init_params(k0)
    optimizer = make_opt(args.lr)
    opt_state = optimizer.init(params)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        key, kb = jax.random.split(key)
        bidx = jax.random.randint(kb, (args.batch,), 0, n)
        params, opt_state, loss = bc_step(
            params, opt_state, optimizer,
            jax.tree.map(lambda x: x[bidx], gs),
            tgt[bidx], frac[bidx], mask[bidx], seat[bidx])
        if step % 100 == 0 or step == 1:
            print(json.dumps({
                "step": step, "bc_ce": round(float(loss), 4),
                "wall_min": round((time.time() - t0) / 60, 1),
            }), flush=True)
        if step % args.eval_every == 0:
            wr, peak = eval_vs_greedy(params, pool)
            print(json.dumps({
                "step": step, "clone_wr_vs_greedy": round(wr, 3),
                "clone_peak_fleets": round(peak, 1),
            }), flush=True)
            save_ckpt(os.path.join(args.out_dir, "bc_net.pkl"),
                      params, {"step": step, "bc_ce": float(loss)})

    save_ckpt(os.path.join(args.out_dir, "bc_net.pkl"), params,
              {"step": args.steps})
    print("done.", flush=True)


if __name__ == "__main__":
    main()
