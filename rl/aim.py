"""JAX lead-aim intercept solver for orbit_wars.

Given a source planet, target planet, and fleet size, find the launch
angle such that the straight-line fleet path intercepts the target's
predicted future position. Mirrors the iterative approach in lib/aim.py
but fully vectorised/jittable over (source, target) pairs.

Conventions (match lib/game/interpreter.py):
- fleet speed = 1 + (max_speed-1) * (log(ships)/log(1000))**1.5, capped
- rotating planet position at absolute step S:
    theta = atan2(iy-C, ix-C) + omega * S;  pos = C + r_orb*(cos,sin)
- comet position at path index i: comet_paths_xy[spawn, path, i]
  (path index advances by 1 per step; comet expires when idx >= len)
- the fleet spawns at source_pos + (radius+0.1)*(cos a, sin a) and moves
  speed*(cos a, sin a) per step. Planets move AFTER fleet movement in
  the same tick; the collision check uses the planet's swept old->new
  segment, so "arrival at predicted position at t" is the right target.

All functions operate on a single GameState (vmap for batches).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.game.interpreter import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS
from lib.game.jax.jax_types import GameState, MAX_PLANETS

AIM_ITERS = 4


def fleet_speed(ships, max_speed: float = 6.0):
    """Per-fleet speed from ship count (engine formula)."""
    ships_f = jnp.maximum(ships.astype(jnp.float32), 1.0)
    s = 1.0 + (max_speed - 1.0) * (jnp.log(ships_f) / jnp.log(1000.0)) ** 1.5
    return jnp.minimum(s, max_speed)


def planet_pos_at(state: GameState, t_rel):
    """Predicted positions of ALL planet slots at `t_rel` steps in the
    future (relative to state.step). Returns (P,2) float32.

    - rotating planets: closed-form orbit at absolute step = step + t_rel
    - static planets: current position
    - comets: path lookup at comet_path_index[spawn] + t_rel (clamped;
      callers should mask expired via `comet_steps_remaining`).
    """
    dx = state.initial_x - CENTER
    dy = state.initial_y - CENTER
    r = jnp.sqrt(dx * dx + dy * dy)
    theta0 = jnp.arctan2(dy, dx)
    # Engine's planet_path_compute uses the PRE-increment step counter:
    # after t more ticks from a state at step S, planets sit at angle
    # theta0 + omega*(S + t - 1). (Verified vs jax_step; off-by-one
    # otherwise — see tests/test_rl_aim.py.)
    abs_step = state.step.astype(jnp.float32) + t_rel - 1.0
    theta = theta0 + state.angular_velocity * abs_step
    is_rot = (r + state.planets_radius < ROTATION_RADIUS_LIMIT) & ~state.is_comet
    rx = CENTER + r * jnp.cos(theta)
    ry = CENTER + r * jnp.sin(theta)

    # Comets: gather along their path.
    spawn_k = jnp.maximum(state.planet_comet_spawn, 0)
    path_j = jnp.maximum(state.planet_comet_path, 0)
    cur_idx = state.comet_path_index[spawn_k]
    fut_idx = (cur_idx.astype(jnp.float32) + t_rel).astype(jnp.int32)
    plen = state.comet_paths_len[spawn_k, path_j]
    safe_idx = jnp.clip(fut_idx, 0, jnp.maximum(plen - 1, 0))
    cx = state.comet_paths_xy[spawn_k, path_j, safe_idx, 0]
    cy = state.comet_paths_xy[spawn_k, path_j, safe_idx, 1]

    px = jnp.where(state.is_comet, cx, jnp.where(is_rot, rx, state.planets_x))
    py = jnp.where(state.is_comet, cy, jnp.where(is_rot, ry, state.planets_y))
    return jnp.stack([px, py], axis=-1)


def comet_steps_remaining(state: GameState):
    """Steps until each planet slot expires (large number for non-comets)."""
    spawn_k = jnp.maximum(state.planet_comet_spawn, 0)
    path_j = jnp.maximum(state.planet_comet_path, 0)
    cur_idx = state.comet_path_index[spawn_k]
    plen = state.comet_paths_len[spawn_k, path_j]
    remain = (plen - 1 - cur_idx).astype(jnp.float32)
    return jnp.where(state.is_comet, jnp.maximum(remain, 0.0), 1e6)


def solve_intercept(state: GameState, ships_grid, max_speed: float = 6.0):
    """Solve lead-aim for every (source, target) planet pair.

    ships_grid: (P, P) int32 — ships that source s would send to target t
      (affects speed). Use a representative value for feature-building;
      exact value at launch time.

    Returns dict of (P, P) arrays:
      angle    — launch angle from source
      eta      — predicted arrival step count (float)
      sun_hit  — straight path passes through the sun
      valid    — target reachable (comet alive at arrival, intercept on board)
    """
    src_pos = jnp.stack([state.planets_x, state.planets_y], axis=-1)  # (P,2)
    speed = fleet_speed(ships_grid, max_speed)  # (P,P)

    # Initial guess: distance to target's current position.
    tgt_now = planet_pos_at(state, jnp.float32(0.0))  # (P,2)

    d0 = jnp.linalg.norm(tgt_now[None, :, :] - src_pos[:, None, :], axis=-1)
    eta = d0 / speed

    def body(_, eta):
        # Predict target position at eta (per-pair → vmap over targets).
        # planet_pos_at over a (P,P) t_rel grid: compute per unique target
        # column. t_rel differs per (s,t) so compute full grid.
        # Vectorise: positions depend on target index t and time eta[s,t].
        def pos_for_src_row(eta_row):
            return planet_pos_at(state, eta_row)  # eta_row (P,) → (P,2)
        tgt_fut = jax.vmap(pos_for_src_row)(eta)  # (P, P, 2)
        d = jnp.linalg.norm(tgt_fut - src_pos[:, None, :], axis=-1)
        return d / speed

    eta = jax.lax.fori_loop(0, AIM_ITERS, body, eta)

    def pos_for_src_row(eta_row):
        return planet_pos_at(state, eta_row)
    tgt_fut = jax.vmap(pos_for_src_row)(eta)  # (P,P,2)

    delta = tgt_fut - src_pos[:, None, :]
    angle = jnp.arctan2(delta[..., 1], delta[..., 0])  # (P,P)

    # Sun check: segment from src to intercept point vs sun disc.
    sx = src_pos[:, None, 0];  sy = src_pos[:, None, 1]
    txx = tgt_fut[..., 0];     tyy = tgt_fut[..., 1]
    vx = txx - sx;             vy = tyy - sy
    l2 = vx * vx + vy * vy
    t_par = ((CENTER - sx) * vx + (CENTER - sy) * vy) / jnp.maximum(l2, 1e-9)
    t_par = jnp.clip(t_par, 0.0, 1.0)
    cx = sx + t_par * vx - CENTER
    cy = sy + t_par * vy - CENTER
    sun_dist = jnp.sqrt(cx * cx + cy * cy)
    sun_hit = sun_dist < (SUN_RADIUS + 0.5)  # small margin

    # Validity: intercept point on board; comet still alive at arrival.
    on_board = (
        (txx >= 1.0) & (txx <= BOARD_SIZE - 1.0)
        & (tyy >= 1.0) & (tyy <= BOARD_SIZE - 1.0)
    )
    remain = comet_steps_remaining(state)  # (P,)
    comet_ok = eta <= remain[None, :] - 1.0
    valid = on_board & comet_ok & state.planets_alive[None, :] & state.planets_alive[:, None]

    return {"angle": angle, "eta": eta, "sun_hit": sun_hit, "valid": valid}


def solve_intercept_rows(state: GameState, tgt_idx, ships,
                         max_speed: float = 6.0):
    """Lead-aim for ONE chosen target per source planet.

    tgt_idx (P,) int32 — chosen target slot per source (clip-safe)
    ships   (P,) int32 — actual ships to send per source

    Returns angle (P,) for launches source_slot -> tgt_idx[source_slot].
    """
    src_pos = jnp.stack([state.planets_x, state.planets_y], axis=-1)  # (P,2)
    speed = fleet_speed(ships, max_speed)  # (P,)
    safe_t = jnp.clip(tgt_idx, 0, MAX_PLANETS - 1)

    tgt_now = planet_pos_at(state, jnp.float32(0.0))[safe_t]  # (P,2)
    eta = jnp.linalg.norm(tgt_now - src_pos, axis=-1) / speed

    def body(_, eta):
        tgt_fut = planet_pos_at_idx(state, safe_t, eta)  # (P,2)
        return jnp.linalg.norm(tgt_fut - src_pos, axis=-1) / speed

    eta = jax.lax.fori_loop(0, AIM_ITERS, body, eta)
    tgt_fut = planet_pos_at_idx(state, safe_t, eta)
    delta = tgt_fut - src_pos
    return jnp.arctan2(delta[:, 1], delta[:, 0])


def planet_pos_at_idx(state: GameState, idx, t_rel):
    """Positions of planets `idx` (K,) at per-entry future times t_rel
    (K,). Same math as planet_pos_at but gathered."""
    ix = state.initial_x[idx]
    iy = state.initial_y[idx]
    dx = ix - CENTER
    dy = iy - CENTER
    r = jnp.sqrt(dx * dx + dy * dy)
    theta0 = jnp.arctan2(dy, dx)
    # Same pre-increment step convention as planet_pos_at.
    abs_step = state.step.astype(jnp.float32) + t_rel - 1.0
    theta = theta0 + state.angular_velocity * abs_step
    radius = state.planets_radius[idx]
    is_comet = state.is_comet[idx]
    is_rot = (r + radius < ROTATION_RADIUS_LIMIT) & ~is_comet
    rx = CENTER + r * jnp.cos(theta)
    ry = CENTER + r * jnp.sin(theta)

    spawn_k = jnp.maximum(state.planet_comet_spawn[idx], 0)
    path_j = jnp.maximum(state.planet_comet_path[idx], 0)
    cur_idx = state.comet_path_index[spawn_k]
    fut_idx = (cur_idx.astype(jnp.float32) + t_rel).astype(jnp.int32)
    plen = state.comet_paths_len[spawn_k, path_j]
    safe_i = jnp.clip(fut_idx, 0, jnp.maximum(plen - 1, 0))
    cx = state.comet_paths_xy[spawn_k, path_j, safe_i, 0]
    cy = state.comet_paths_xy[spawn_k, path_j, safe_i, 1]

    px = jnp.where(is_comet, cx,
                   jnp.where(is_rot, rx, state.planets_x[idx]))
    py = jnp.where(is_comet, cy,
                   jnp.where(is_rot, ry, state.planets_y[idx]))
    return jnp.stack([px, py], axis=-1)
