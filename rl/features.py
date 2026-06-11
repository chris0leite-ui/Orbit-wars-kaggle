"""Observation builder: GameState -> network features (JAX, jittable).

Per-seat features are built from a single GameState:
  nodes   (P, NODE_DIM)   per-planet tokens (padded slots zeroed)
  edges   (P, P, EDGE_DIM) pairwise source->target geometry
  globals (GLOBAL_DIM,)
  src_mask (P,)           planets this seat may launch from
  tgt_mask (P, P+1)       legal (source, target) pairs + hold col (always 1)

Owner encoding is seat-relative: mine / enemy / neutral. All ship counts
log-scaled. The seat-independent heavy pieces (incoming-fleet arrival
projection, intercept solve) are computed once per state in
`state_tables` and shared across seats.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from lib.game.interpreter import CENTER, ROTATION_RADIUS_LIMIT
from lib.game.jax.jax_types import GameState, MAX_AGENTS, MAX_FLEETS, MAX_PLANETS
from rl.aim import (
    comet_steps_remaining,
    fleet_speed,
    planet_pos_at,
    solve_intercept,
)

# ETA buckets for incoming-fleet projection (upper bounds, inclusive).
ETA_BUCKETS = jnp.array([3.0, 7.0, 12.0, 18.0, 26.0, 48.0])
N_BUCKETS = 6
T_HORIZON = 48  # scan horizon for fleet arrival projection

NODE_DIM = 14 + 2 * N_BUCKETS  # = 26
EDGE_DIM = 6
GLOBAL_DIM = 12
N_FRACS = 4
FRACS = jnp.array([0.25, 0.5, 0.75, 1.0])


def fleet_arrivals(state: GameState):
    """Project every in-flight fleet onto the planet it will first hit.

    Returns:
      arrive_ships (P, MAX_AGENTS, N_BUCKETS) — ships arriving per
        (planet, owner, eta bucket)
      cum_arrivals (P, MAX_AGENTS, N_BUCKETS) — cumulative over buckets
    Scan over t=1..T_HORIZON keeping the first hit per fleet.
    """
    f_pos = jnp.stack([state.fleets_x, state.fleets_y], axis=-1)  # (F,2)
    speed = fleet_speed(state.fleets_ships)
    vel = speed[:, None] * jnp.stack(
        [jnp.cos(state.fleets_angle), jnp.sin(state.fleets_angle)], axis=-1
    )  # (F,2)

    def body(carry, t):
        hit_t, hit_p = carry
        tf = t.astype(jnp.float32)
        fp = f_pos + vel * tf  # (F,2)
        pp = planet_pos_at(state, tf)  # (P,2)
        d = jnp.linalg.norm(fp[:, None, :] - pp[None, :, :], axis=-1)  # (F,P)
        thresh = state.planets_radius[None, :] + 0.5 * speed[:, None]
        hits = (d < thresh) & state.planets_alive[None, :]
        any_hit = jnp.any(hits, axis=1)
        first_p = jnp.argmax(hits, axis=1)
        new = any_hit & (hit_t < 0)
        hit_t = jnp.where(new, t, hit_t)
        hit_p = jnp.where(new, first_p, hit_p)
        return (hit_t, hit_p), None

    init = (-jnp.ones(MAX_FLEETS, jnp.int32), jnp.zeros(MAX_FLEETS, jnp.int32))
    (hit_t, hit_p), _ = jax.lax.scan(body, init, jnp.arange(1, T_HORIZON + 1))

    live = state.fleets_alive & (hit_t > 0) & (state.fleets_owner >= 0)
    bucket = jnp.searchsorted(ETA_BUCKETS, hit_t.astype(jnp.float32))  # (F,)
    bucket = jnp.clip(bucket, 0, N_BUCKETS - 1)

    arrive = jnp.zeros((MAX_PLANETS, MAX_AGENTS, N_BUCKETS), jnp.float32)
    contrib = jnp.where(live, state.fleets_ships.astype(jnp.float32), 0.0)
    safe_p = jnp.where(live, hit_p, 0)
    safe_o = jnp.where(live, state.fleets_owner, 0)
    safe_b = jnp.where(live, bucket, 0)
    arrive = arrive.at[safe_p, safe_o, safe_b].add(contrib)
    cum = jnp.cumsum(arrive, axis=-1)
    return arrive, cum


def state_tables(state: GameState):
    """Seat-independent expensive tables, computed once per state."""
    arrive, cum = fleet_arrivals(state)
    # Representative intercept solve: ships = half the source garrison.
    half = jnp.maximum(state.planets_ships // 2, 1)  # (P,)
    ships_grid = jnp.broadcast_to(half[:, None], (MAX_PLANETS, MAX_PLANETS))
    aim = solve_intercept(state, ships_grid)
    return {"arrive": arrive, "cum": cum, "aim": aim}


def _owner_classes(state: GameState, seat):
    owner = state.planets_owner
    mine = (owner == seat) & state.planets_alive
    neutral = (owner == -1) & state.planets_alive
    enemy = (~mine) & (~neutral) & (owner >= 0) & state.planets_alive
    return mine, enemy, neutral


def seat_features(state: GameState, tables, seat):
    """Build (nodes, edges, globals, src_mask, tgt_mask) for one seat."""
    P = MAX_PLANETS
    alive = state.planets_alive
    mine, enemy, neutral = _owner_classes(state, seat)

    ships = state.planets_ships.astype(jnp.float32)
    log_ships = jnp.log1p(jnp.maximum(ships, 0.0)) / 5.0
    prod = state.planets_prod.astype(jnp.float32) / 5.0
    radius = state.planets_radius / 3.0

    dx = state.planets_x - CENTER
    dy = state.planets_y - CENTER
    r_orb = jnp.sqrt(dx * dx + dy * dy)
    theta = jnp.arctan2(dy, dx)
    idx0 = jnp.sqrt((state.initial_x - CENTER) ** 2 + (state.initial_y - CENTER) ** 2)
    is_rot = ((idx0 + state.planets_radius < ROTATION_RADIUS_LIMIT)
              & ~state.is_comet & alive)
    remain = jnp.minimum(comet_steps_remaining(state), 999.0) / 40.0
    remain = jnp.where(state.is_comet, remain, 0.0)

    # Incoming arrivals relative to seat: (P, A, B) -> mine / enemies.
    arrive = tables["arrive"]  # (P, A, B)
    inc_mine = arrive[:, seat, :]  # (P, B)
    seat_ids = jnp.arange(MAX_AGENTS)
    enemy_seats = (seat_ids != seat) & (seat_ids < state.num_agents)
    inc_enemy = jnp.sum(jnp.where(enemy_seats[None, :, None], arrive, 0.0), axis=1)

    nodes = jnp.concatenate([
        mine[:, None].astype(jnp.float32),
        enemy[:, None].astype(jnp.float32),
        neutral[:, None].astype(jnp.float32),
        log_ships[:, None],
        prod[:, None],
        radius[:, None],
        (dx / 50.0)[:, None],
        (dy / 50.0)[:, None],
        (r_orb / 50.0)[:, None],
        jnp.cos(theta)[:, None],
        jnp.sin(theta)[:, None],
        is_rot[:, None].astype(jnp.float32),
        state.is_comet[:, None].astype(jnp.float32),
        remain[:, None],
        jnp.log1p(inc_mine) / 5.0,
        jnp.log1p(inc_enemy) / 5.0,
    ], axis=-1)
    nodes = jnp.where(alive[:, None], nodes, 0.0)

    # --- Edges ---
    aim = tables["aim"]
    eta = aim["eta"]                       # (P,P)
    eta_n = jnp.clip(eta, 0.0, 60.0) / 40.0
    sun = aim["sun_hit"].astype(jnp.float32)
    valid = aim["valid"].astype(jnp.float32)

    # Target garrison projection at arrival: ships + prod*eta (if owned)
    # + own-side arrivals before eta − enemy(of target)-side arrivals.
    # Approximate with cumulative bucket at eta's bucket.
    bucket_e = jnp.clip(jnp.searchsorted(ETA_BUCKETS, eta), 0, N_BUCKETS - 1)
    cum = tables["cum"]  # (P, A, B)
    cum_mine_t = cum[:, seat, :]  # (P,B) — arrivals from MY fleets at target
    cum_enemy_t = jnp.sum(jnp.where(enemy_seats[None, :, None], cum, 0.0), axis=1)
    # gather per (s,t): cum at bucket_e[s,t] for target t
    tgt_idx = jnp.broadcast_to(jnp.arange(P)[None, :], (P, P))
    my_before = cum_mine_t[tgt_idx, bucket_e]      # (P,P)
    en_before = cum_enemy_t[tgt_idx, bucket_e]     # (P,P)

    owned_t = (state.planets_owner >= 0) & alive
    garr_proj = (ships[None, :]
                 + jnp.where(owned_t, state.planets_prod, 0)[None, :].astype(jnp.float32) * eta)
    # Defense seen by ME attacking target t: garrison + enemy reinforcement
    # − my inbound. Signed: negative means I already out-commit it.
    net_def = garr_proj + jnp.where(mine[None, :], my_before, en_before) \
        - jnp.where(mine[None, :], en_before, my_before)
    net_def_n = jnp.sign(net_def) * jnp.log1p(jnp.abs(net_def)) / 5.0

    dist = eta * fleet_speed(jnp.maximum(state.planets_ships // 2, 1))[:, None]
    edges = jnp.stack([
        eta_n, sun, valid, net_def_n,
        jnp.clip(dist, 0.0, 150.0) / 100.0,
        jnp.eye(P),
    ], axis=-1)

    # --- Globals ---
    my_ships_tot = jnp.sum(jnp.where(mine, ships, 0.0))
    en_ships_tot = jnp.sum(jnp.where(enemy, ships, 0.0))
    my_prod_tot = jnp.sum(jnp.where(mine, state.planets_prod, 0).astype(jnp.float32))
    en_prod_tot = jnp.sum(jnp.where(enemy, state.planets_prod, 0).astype(jnp.float32))
    # Fleet ships count toward material.
    fl_mine = jnp.sum(jnp.where(
        (state.fleets_owner == seat) & state.fleets_alive,
        state.fleets_ships, 0).astype(jnp.float32))
    fl_enemy = jnp.sum(jnp.where(
        (state.fleets_owner != seat) & (state.fleets_owner >= 0) & state.fleets_alive,
        state.fleets_ships, 0).astype(jnp.float32))
    my_mat = my_ships_tot + fl_mine
    en_mat = en_ships_tot + fl_enemy
    tot = my_mat + en_mat + 1e-6
    step_f = state.step.astype(jnp.float32)
    next_spawn = jnp.min(jnp.where(
        state.comet_step.astype(jnp.float32) > step_f,
        state.comet_step.astype(jnp.float32) - step_f, 999.0))
    globals_ = jnp.stack([
        step_f / 500.0,
        (state.num_agents == 2).astype(jnp.float32),
        (state.num_agents == 4).astype(jnp.float32),
        my_mat / tot,
        en_mat / tot,
        (my_prod_tot - en_prod_tot) / (my_prod_tot + en_prod_tot + 1e-6),
        jnp.sum(mine.astype(jnp.float32)) / 10.0,
        jnp.sum(enemy.astype(jnp.float32)) / 10.0,
        jnp.sum(neutral.astype(jnp.float32)) / 10.0,
        jnp.clip(next_spawn, 0.0, 200.0) / 100.0,
        state.angular_velocity * 20.0,
        jnp.log1p(my_mat) / 8.0,
    ])

    # --- Masks ---
    src_mask = mine & (state.planets_ships >= 1)
    pair_ok = (aim["valid"] & ~aim["sun_hit"]
               & alive[None, :] & (~jnp.eye(P, dtype=bool)))
    pair_ok = pair_ok & src_mask[:, None]
    hold_col = jnp.ones((P, 1), dtype=bool)
    tgt_mask = jnp.concatenate([pair_ok, hold_col], axis=1)  # (P, P+1)

    return nodes, edges, globals_, src_mask, tgt_mask


def material_potential(state: GameState):
    """Per-seat shaping potential Φ ∈ [-1, 1]: (mine − best rival) /
    total, where material = planet ships + fleet ships + 20·production.
    Returns (MAX_AGENTS,) float32; inactive seats get 0.
    """
    W_PROD = 20.0
    seat_ids = jnp.arange(MAX_AGENTS)

    def per_seat(seat):
        p_mask = (state.planets_owner == seat) & state.planets_alive
        mat = jnp.sum(jnp.where(p_mask, state.planets_ships, 0).astype(jnp.float32))
        mat += W_PROD * jnp.sum(jnp.where(p_mask, state.planets_prod, 0).astype(jnp.float32))
        f_mask = (state.fleets_owner == seat) & state.fleets_alive
        mat += jnp.sum(jnp.where(f_mask, state.fleets_ships, 0).astype(jnp.float32))
        return mat

    mats = jax.vmap(per_seat)(seat_ids)  # (A,)
    active = seat_ids < state.num_agents
    mats = jnp.where(active, mats, 0.0)

    def phi(seat):
        my = mats[seat]
        rival = jnp.max(jnp.where((seat_ids != seat) & active, mats, -1.0))
        return (my - rival) / (my + rival + 1e-6)

    phis = jax.vmap(phi)(seat_ids)
    return jnp.where(active, phis, 0.0)
