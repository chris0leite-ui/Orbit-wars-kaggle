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


def featurize_envstep(state):
    """All-seat features for one stored env-step. Param-independent —
    MUST run outside value_and_grad so the backward pass never
    differentiates through the arrival scan / aim solve (that was a
    14.8 GB OOM at batch 256 on a T4)."""
    tables = state_tables(state)

    def per_seat(seat):
        return seat_features(state, tables, seat)

    nodes, edges, globals_, src_mask, tgt_mask = jax.vmap(per_seat)(
        jnp.arange(MAX_AGENTS))
    return {
        "nodes": nodes, "edges": edges, "globals": globals_,
        "src_mask": src_mask, "tgt_mask": tgt_mask,
        "alive": state.planets_alive,
    }


def _loss_one_envstep(params, f, tgt, frac, old_logp, adv, ret, active):
    """Loss pieces for one env-step from precomputed features `f`."""

    def per_seat(seat):
        logp, value, entropy = net.action_logp_value(
            params, f["nodes"][seat], f["edges"][seat], f["globals"][seat],
            f["alive"], f["src_mask"][seat], f["tgt_mask"][seat],
            tgt[seat], frac[seat])
        n_src = jnp.maximum(
            jnp.sum(f["src_mask"][seat].astype(jnp.float32)), 1.0)
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


def ppo_loss(params, feats, batch, entropy_coef=ENTROPY_COEF):
    out = jax.vmap(_loss_one_envstep, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        params, feats, batch["tgt"], batch["frac"],
        batch["logp"], batch["adv"], batch["ret"], batch["active"])
    n = jnp.maximum(jnp.sum(out["n"]), 1.0)
    pg = jnp.sum(out["pg"]) / n
    v = jnp.sum(out["v"]) / n
    ent = jnp.sum(out["ent"]) / n
    loss = pg + VALUE_COEF * v - entropy_coef * ent
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


# ---------------- behavior-cloning auxiliary update ----------------

def featurize_bc(gs, seat):
    """Per-sample seat features (param-independent — compute OUTSIDE the
    grad so the aim solver / arrival scan never enter the backward pass,
    same OOM/throughput fix as featurize_envstep)."""
    tables = state_tables(gs)
    nodes, edges, globals_, src_mask, tgt_mask = seat_features(
        gs, tables, seat)
    return {
        "nodes": nodes, "edges": edges, "globals": globals_,
        "src_mask": src_mask, "tgt_mask": tgt_mask,
        "alive": gs.planets_alive,
    }


def bc_loss_feats(params, feats, tgt, frac, mask):
    """Cross-entropy on the policy heads from PRECOMPUTED features.

    feats: vmapped dict from featurize_bc; tgt/frac/mask (B, P).
    Only sources with a label (mask) and a mask-legal target contribute.
    """
    def one(f, tgt_i, frac_i, mask_i):
        _, logits, emb = net.forward(
            params, f["nodes"], f["edges"], f["globals"], f["alive"],
            f["tgt_mask"])
        logp_t = jax.nn.log_softmax(logits, axis=-1)
        pick_t = jnp.take_along_axis(
            logp_t, tgt_i[:, None].astype(jnp.int32), axis=1)[:, 0]
        fl = net.frac_logits_for(params, emb, f["edges"],
                                 tgt_i.astype(jnp.int32))
        logp_f = jax.nn.log_softmax(fl, axis=-1)
        pick_f = jnp.take_along_axis(
            logp_f, frac_i[:, None].astype(jnp.int32), axis=1)[:, 0]
        # Drop labels the policy cannot express: producer occasionally
        # fires through the sun margin / at pairs our mask forbids;
        # their -1e9 logits explode the CE (observed 1.4e7 spike).
        legal = jnp.take_along_axis(
            f["tgt_mask"], tgt_i[:, None].astype(jnp.int32), axis=1)[:, 0]
        w = (mask_i & legal).astype(jnp.float32) \
            * f["src_mask"].astype(jnp.float32)
        ce_t = -jnp.sum(pick_t * w)
        ce_f = -jnp.sum(pick_f * w)
        return ce_t + 0.5 * ce_f, jnp.sum(w)

    losses, ns = jax.vmap(one)(feats, tgt, frac, mask)
    n = jnp.maximum(jnp.sum(ns), 1.0)
    return jnp.sum(losses) / n


@partial(jax.jit, static_argnames=())
def bc_update(params, opt_state, optimizer_lr, bc_coef,
              gs_batch, tgt, frac, mask, seat):
    optimizer = make_optimizer(optimizer_lr)
    feats = jax.vmap(featurize_bc)(gs_batch, seat)
    loss, grads = jax.value_and_grad(bc_loss_feats)(
        params, feats, tgt, frac, mask)
    grads = jax.tree.map(lambda g: g * bc_coef, grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@partial(jax.jit, static_argnames=("n_minibatch", "n_epochs"))
def ppo_update(key, params, opt_state, optimizer_lr, traj, bootstrap_v,
               n_minibatch: int = 8, n_epochs: int = 2,
               entropy_coef: float = ENTROPY_COEF):
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
            # Featurize OUTSIDE the grad: param-independent, and keeps
            # the arrival-scan/aim-solve out of the backward pass.
            feats = jax.vmap(featurize_envstep)(batch["state"])
            (loss, metrics), grads = jax.value_and_grad(
                ppo_loss, has_aux=True)(params, feats, batch, entropy_coef)
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
