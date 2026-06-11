"""Policy/value network: small transformer over planet tokens.

Pure-JAX, params as a flat dict of arrays — no flax. This keeps the
eval-time numpy port (rl/numpy_infer.py) a line-for-line mirror.

Tokens: MAX_PLANETS planet tokens + 1 global token. Padded planet slots
are masked out of attention.

Heads:
  value        — scalar from global token
  target       — pointer logits (P, P+1): src->tgt attention + edge-MLP
                 bias; last column = hold
  fraction     — 4-way over {25,50,75,100}% given (src, tgt) pair
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

from lib.game.jax.jax_types import MAX_PLANETS
from rl.features import EDGE_DIM, GLOBAL_DIM, N_FRACS, NODE_DIM

D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
D_FF = 128
D_HEAD = D_MODEL // N_HEADS
N_TOKENS = MAX_PLANETS + 1  # +1 global


def _dense_init(key, fan_in, fan_out, scale=1.0):
    std = scale * math.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, (fan_in, fan_out), jnp.float32) * std


def init_params(key):
    keys = jax.random.split(key, 64)
    ki = iter(keys)
    p = {}
    p["embed_w"] = _dense_init(next(ki), NODE_DIM, D_MODEL)
    p["embed_b"] = jnp.zeros(D_MODEL)
    p["gembed_w"] = _dense_init(next(ki), GLOBAL_DIM, D_MODEL)
    p["gembed_b"] = jnp.zeros(D_MODEL)
    for i in range(N_LAYERS):
        p[f"l{i}_ln1_g"] = jnp.ones(D_MODEL)
        p[f"l{i}_ln1_b"] = jnp.zeros(D_MODEL)
        p[f"l{i}_qkv_w"] = _dense_init(next(ki), D_MODEL, 3 * D_MODEL)
        p[f"l{i}_qkv_b"] = jnp.zeros(3 * D_MODEL)
        p[f"l{i}_proj_w"] = _dense_init(next(ki), D_MODEL, D_MODEL)
        p[f"l{i}_proj_b"] = jnp.zeros(D_MODEL)
        p[f"l{i}_ln2_g"] = jnp.ones(D_MODEL)
        p[f"l{i}_ln2_b"] = jnp.zeros(D_MODEL)
        p[f"l{i}_ff1_w"] = _dense_init(next(ki), D_MODEL, D_FF)
        p[f"l{i}_ff1_b"] = jnp.zeros(D_FF)
        p[f"l{i}_ff2_w"] = _dense_init(next(ki), D_FF, D_MODEL)
        p[f"l{i}_ff2_b"] = jnp.zeros(D_MODEL)
    p["lnf_g"] = jnp.ones(D_MODEL)
    p["lnf_b"] = jnp.zeros(D_MODEL)
    # value head
    p["v1_w"] = _dense_init(next(ki), D_MODEL, D_MODEL)
    p["v1_b"] = jnp.zeros(D_MODEL)
    p["v2_w"] = _dense_init(next(ki), D_MODEL, 1, scale=0.1)
    p["v2_b"] = jnp.zeros(1)
    # target head
    p["tq_w"] = _dense_init(next(ki), D_MODEL, D_MODEL)
    p["tq_b"] = jnp.zeros(D_MODEL)
    p["tk_w"] = _dense_init(next(ki), D_MODEL, D_MODEL)
    p["tk_b"] = jnp.zeros(D_MODEL)
    p["edge1_w"] = _dense_init(next(ki), EDGE_DIM, 32)
    p["edge1_b"] = jnp.zeros(32)
    p["edge2_w"] = _dense_init(next(ki), 32, 1, scale=0.1)
    p["edge2_b"] = jnp.zeros(1)
    p["hold_w"] = _dense_init(next(ki), D_MODEL, 1, scale=0.1)
    p["hold_b"] = jnp.zeros(1)
    # fraction head: input src_emb ++ tgt_emb ++ edge
    p["f1_w"] = _dense_init(next(ki), 2 * D_MODEL + EDGE_DIM, 32)
    p["f1_b"] = jnp.zeros(32)
    p["f2_w"] = _dense_init(next(ki), 32, N_FRACS, scale=0.1)
    p["f2_b"] = jnp.zeros(N_FRACS)
    return p


def _ln(x, g, b):
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.var(x, -1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def encode(params, nodes, globals_, alive_mask):
    """nodes (P, NODE_DIM), globals (GLOBAL_DIM,), alive (P,) bool
    -> token embeddings (P+1, D_MODEL); last token = global."""
    x_p = nodes @ params["embed_w"] + params["embed_b"]            # (P,D)
    x_g = (globals_ @ params["gembed_w"] + params["gembed_b"])[None, :]
    x = jnp.concatenate([x_p, x_g], axis=0)                        # (T,D)
    tok_mask = jnp.concatenate([alive_mask, jnp.ones(1, bool)])    # (T,)

    neg = jnp.float32(-1e9)
    for i in range(N_LAYERS):
        h = _ln(x, params[f"l{i}_ln1_g"], params[f"l{i}_ln1_b"])
        qkv = h @ params[f"l{i}_qkv_w"] + params[f"l{i}_qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        k = k.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        v = v.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / math.sqrt(D_HEAD)         # (H,T,T)
        att = jnp.where(tok_mask[None, None, :], att, neg)
        att = jax.nn.softmax(att, axis=-1)
        out = (att @ v).transpose(1, 0, 2).reshape(N_TOKENS, D_MODEL)
        x = x + out @ params[f"l{i}_proj_w"] + params[f"l{i}_proj_b"]
        h = _ln(x, params[f"l{i}_ln2_g"], params[f"l{i}_ln2_b"])
        h = jax.nn.gelu(h @ params[f"l{i}_ff1_w"] + params[f"l{i}_ff1_b"])
        x = x + h @ params[f"l{i}_ff2_w"] + params[f"l{i}_ff2_b"]
    x = _ln(x, params["lnf_g"], params["lnf_b"])
    # Zero out padded tokens so downstream heads see clean zeros.
    x = jnp.where(tok_mask[:, None], x, 0.0)
    return x


def forward(params, nodes, edges, globals_, alive_mask, tgt_mask):
    """Single-state forward.

    Returns:
      value          ()           state value for this seat
      target_logits  (P, P+1)     masked (−1e9 illegal); col P = hold
      emb            (P+1, D)     token embeddings (for fraction head)
    """
    emb = encode(params, nodes, globals_, alive_mask)
    g = emb[-1]
    v = jax.nn.gelu(g @ params["v1_w"] + params["v1_b"])
    value = (v @ params["v2_w"] + params["v2_b"])[0]

    pe = emb[:MAX_PLANETS]                                          # (P,D)
    q = pe @ params["tq_w"] + params["tq_b"]
    k = pe @ params["tk_w"] + params["tk_b"]
    ptr = q @ k.T / math.sqrt(D_MODEL)                              # (P,P)
    e = jax.nn.gelu(edges @ params["edge1_w"] + params["edge1_b"])
    ebias = (e @ params["edge2_w"] + params["edge2_b"])[..., 0]     # (P,P)
    hold = pe @ params["hold_w"] + params["hold_b"]                 # (P,1)
    logits = jnp.concatenate([ptr + ebias, hold], axis=1)           # (P,P+1)
    logits = jnp.where(tgt_mask, logits, -1e9)
    return value, logits, emb


def frac_logits_for(params, emb, edges, tgt_choice):
    """Fraction logits for each source's CHOSEN target.

    emb (P+1, D), edges (P,P,EDGE_DIM), tgt_choice (P,) int (may be P=hold).
    Returns (P, N_FRACS).
    """
    pe = emb[:MAX_PLANETS]
    safe_t = jnp.clip(tgt_choice, 0, MAX_PLANETS - 1)
    tgt_emb = pe[safe_t]                                            # (P,D)
    src_idx = jnp.arange(MAX_PLANETS)
    e = edges[src_idx, safe_t]                                      # (P,EDGE)
    h = jnp.concatenate([pe, tgt_emb, e], axis=-1)
    h = jax.nn.gelu(h @ params["f1_w"] + params["f1_b"])
    return h @ params["f2_w"] + params["f2_b"]


def sample_actions(key, params, nodes, edges, globals_, alive_mask,
                   src_mask, tgt_mask):
    """Sample per-planet (target, fraction) for one seat; return action,
    log-prob, value, entropy estimate.

    Sources not in src_mask are forced to hold with logp 0 contribution.
    """
    value, logits, emb = forward(params, nodes, edges, globals_,
                                 alive_mask, tgt_mask)
    kt, kf = jax.random.split(key)
    # Per-source categorical over P+1.
    tgt_choice = jax.random.categorical(kt, logits, axis=-1)        # (P,)
    logp_t_all = jax.nn.log_softmax(logits, axis=-1)
    logp_t = jnp.take_along_axis(
        logp_t_all, tgt_choice[:, None], axis=1)[:, 0]              # (P,)

    fl = frac_logits_for(params, emb, edges, tgt_choice)            # (P,4)
    frac_choice = jax.random.categorical(kf, fl, axis=-1)           # (P,)
    logp_f_all = jax.nn.log_softmax(fl, axis=-1)
    logp_f = jnp.take_along_axis(
        logp_f_all, frac_choice[:, None], axis=1)[:, 0]

    is_launch = (tgt_choice < MAX_PLANETS) & src_mask
    logp = jnp.sum(jnp.where(src_mask, logp_t, 0.0)) + \
        jnp.sum(jnp.where(is_launch, logp_f, 0.0))

    # Entropy (for the loss): per-source target entropy + frac entropy
    # on launching sources.
    p_t = jnp.exp(logp_t_all)
    ent_t = -jnp.sum(jnp.where(tgt_mask, p_t * logp_t_all, 0.0), axis=-1)
    p_f = jnp.exp(logp_f_all)
    ent_f = -jnp.sum(p_f * logp_f_all, axis=-1)
    entropy = jnp.sum(jnp.where(src_mask, ent_t, 0.0)) + \
        jnp.sum(jnp.where(is_launch, ent_f, 0.0))

    return {
        "tgt": tgt_choice, "frac": frac_choice,
        "logp": logp, "value": value, "entropy": entropy,
    }


def action_logp_value(params, nodes, edges, globals_, alive_mask,
                      src_mask, tgt_mask, tgt_choice, frac_choice):
    """Recompute logp/value/entropy of a GIVEN action under params
    (PPO update path)."""
    value, logits, emb = forward(params, nodes, edges, globals_,
                                 alive_mask, tgt_mask)
    logp_t_all = jax.nn.log_softmax(logits, axis=-1)
    logp_t = jnp.take_along_axis(
        logp_t_all, tgt_choice[:, None], axis=1)[:, 0]
    fl = frac_logits_for(params, emb, edges, tgt_choice)
    logp_f_all = jax.nn.log_softmax(fl, axis=-1)
    logp_f = jnp.take_along_axis(
        logp_f_all, frac_choice[:, None], axis=1)[:, 0]
    is_launch = (tgt_choice < MAX_PLANETS) & src_mask
    logp = jnp.sum(jnp.where(src_mask, logp_t, 0.0)) + \
        jnp.sum(jnp.where(is_launch, logp_f, 0.0))

    p_t = jnp.exp(logp_t_all)
    ent_t = -jnp.sum(jnp.where(tgt_mask, p_t * logp_t_all, 0.0), axis=-1)
    p_f = jnp.exp(logp_f_all)
    ent_f = -jnp.sum(p_f * logp_f_all, axis=-1)
    entropy = jnp.sum(jnp.where(src_mask, ent_t, 0.0)) + \
        jnp.sum(jnp.where(is_launch, ent_f, 0.0))
    return logp, value, entropy


def count_params(params):
    return sum(int(v.size) for v in params.values())
