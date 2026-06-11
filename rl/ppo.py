"""PPO update on stored GameState trajectories (features recomputed
inside the loss — memory-light buffers).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from lib.game.jax.jax_types import MAX_AGENTS
from rl import net
from rl.features import seat_features, state_tables
from rl.rollout import GAMMA

LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


def compute_gae(traj, bootstrap_v):
    """traj fields (T,B,...); bootstrap_v (B,A). Returns adv, ret (T,B,A)."""
    values = traj["value"]          # (T,B,A)
    rewards = traj["reward"]        # (T,B,A)
    dones = traj["done"]            # (T,B)

    def scan_fn(carry, xs):
        next_adv, next_v = carry
        r, v, d = xs
        nonterm = (1.0 - d.astype(jnp.float32))[:, None]  # (B,1)
        delta = r + GAMMA * next_v * nonterm - v
        adv = delta + GAMMA * LAMBDA * nonterm * next_adv
        return (adv, v), adv

    (_, _), advs = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(bootstrap_v), bootstrap_v),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advs + values
    return advs, returns


def _loss_one_envstep(params, state, tgt, frac, old_logp, adv, ret, active):
    """Loss pieces for one stored env-step (all seats).

    state: single GameState; tgt/frac (A,P); old_logp/adv/ret/active (A,).
    """
    tables = state_tables(state)

    def per_seat(seat):
        nodes, edges, globals_, src_mask, tgt_mask = seat_features(
            state, tables, seat)
        logp, value, entropy = net.action_logp_value(
            params, nodes, edges, globals_, state.planets_alive,
            src_mask, tgt_mask, tgt[seat], frac[seat])
        n_src = jnp.maximum(jnp.sum(src_mask.astype(jnp.float32)), 1.0)
        return logp, value, entropy / n_src

    logp, value, entropy = jax.vmap(per_seat)(jnp.arange(MAX_AGENTS))

    ratio = jnp.exp(logp - old_logp)
    s1 = ratio * adv
    s2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
    pg = -jnp.minimum(s1, s2)
    v_loss = 0.5 * (value - ret) ** 2
    w = active.astype(jnp.float32)
    kl = (old_logp - logp)
    clipped = (jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32)
    return {
        "pg": jnp.sum(pg * w), "v": jnp.sum(v_loss * w),
        "ent": jnp.sum(entropy * w), "kl": jnp.sum(kl * w),
        "clip": jnp.sum(clipped * w), "n": jnp.sum(w),
    }


def ppo_loss(params, batch):
    out = jax.vmap(_loss_one_envstep, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        params, batch["state"], batch["tgt"], batch["frac"],
        batch["logp"], batch["adv"], batch["ret"], batch["active"])
    n = jnp.maximum(jnp.sum(out["n"]), 1.0)
    pg = jnp.sum(out["pg"]) / n
    v = jnp.sum(out["v"]) / n
    ent = jnp.sum(out["ent"]) / n
    loss = pg + VALUE_COEF * v - ENTROPY_COEF * ent
    metrics = {
        "loss": loss, "pg_loss": pg, "v_loss": v, "entropy": ent,
        "approx_kl": jnp.sum(out["kl"]) / n,
        "clip_frac": jnp.sum(out["clip"]) / n,
    }
    return loss, metrics


def make_optimizer(lr: float = 3e-4):
    return optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(lr),
    )


@partial(jax.jit, static_argnames=("n_minibatch", "n_epochs"))
def ppo_update(key, params, opt_state, optimizer_lr, traj, bootstrap_v,
               n_minibatch: int = 8, n_epochs: int = 2):
    """Full PPO update over one rollout chunk. Returns new params/opt
    state + averaged metrics."""
    optimizer = make_optimizer(optimizer_lr)

    adv, ret = compute_gae(traj, bootstrap_v)
    # Normalize advantages over active seats.
    active = traj["active"]
    w = active.astype(jnp.float32)
    mean = jnp.sum(adv * w) / jnp.maximum(jnp.sum(w), 1.0)
    var = jnp.sum(((adv - mean) ** 2) * w) / jnp.maximum(jnp.sum(w), 1.0)
    adv = (adv - mean) / jnp.sqrt(var + 1e-8)

    T = traj["done"].shape[0]
    B = traj["done"].shape[1]
    N = T * B

    flat = {
        "state": jax.tree.map(
            lambda x: x.reshape((N,) + x.shape[2:]), traj["state"]),
        "tgt": traj["tgt"].reshape((N,) + traj["tgt"].shape[2:]),
        "frac": traj["frac"].reshape((N,) + traj["frac"].shape[2:]),
        "logp": traj["logp"].reshape((N, MAX_AGENTS)),
        "adv": adv.reshape((N, MAX_AGENTS)),
        "ret": ret.reshape((N, MAX_AGENTS)),
        "active": active.reshape((N, MAX_AGENTS)),
    }
    mb_size = N // n_minibatch

    def epoch_fn(carry, ek):
        params, opt_state = carry
        perm = jax.random.permutation(ek, N)

        def mb_fn(carry2, i):
            params, opt_state = carry2
            idx = jax.lax.dynamic_slice_in_dim(perm, i * mb_size, mb_size)
            batch = jax.tree.map(lambda x: x[idx], flat)
            (loss, metrics), grads = jax.value_and_grad(
                ppo_loss, has_aux=True)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(
            mb_fn, (params, opt_state), jnp.arange(n_minibatch))
        return (params, opt_state), metrics

    epoch_keys = jax.random.split(key, n_epochs)
    (params, opt_state), metrics = jax.lax.scan(
        epoch_fn, (params, opt_state), epoch_keys)
    metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
    return params, opt_state, metrics
