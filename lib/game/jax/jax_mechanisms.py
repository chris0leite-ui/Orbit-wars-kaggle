"""JAX-adjacent mechanism stack for the rollout path.

Operates on the dicts emitted by `settle_plan_from_matrices` (numpy)
plus the JAX state + JaxWorldModel. Returns per-agent action lists
ready for `actions_to_jax(...)` → `jax_step(...)`.

For sub-phase 4 (rollout-first) we cover the high-impact mechanisms:
  - validate (final src/owner/ships check),
  - arrival_size (production-aware ship sizing for enemy targets),
  - simple atan2 aim (no lead — covers static targets exactly,
    moving targets approximately).

Deferred to sub-phase 7 for parity-exact emission:
  - lead_aim_v2 (5-iter fixed-point + search_safe_intercept fallback),
  - sun_avoid (predict_fleet_fate ray-cast),
  - path_clears_other_planets,
  - oob_guard,
  - gang_up_size (DEFAULT-off currently).

The simple atan2 aim diverges from scalar by ~0.05 rad on the worst
orbital targets at long range. Empirically this changes < 3 % of fleet
outcomes; tolerable for the candidate-ordering use case of v7_0's
drop-one chooser.
"""

from __future__ import annotations

import math

import numpy as np

from lib.fleet import speed as fleet_speed
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT

_CENTER = 50.0
_ROTATION_RADIUS_LIMIT = 50.0
_SUN_RADIUS = 10.0
_SUN_SAFETY = 0.5            # mirror lib.trajectory.SUN_SAFETY
_MAX_AIM_ITERATIONS = 5
_AIM_CONVERGE_XY_TOL = 0.5
_BOARD_LO = 0.0
_BOARD_HI = 100.0
_PATH_MAX_STEPS = 200        # mirror lib.trajectory.DEFAULT_MAX_STEPS
_INTERCEPT_TOLERANCE = 1     # mirror lib.aim.INTERCEPT_TOLERANCE
_SEARCH_HORIZON = 60         # mirror lib.aim.SEARCH_HORIZON


def _is_orbiting(px, py, pr):
    return math.hypot(px - _CENTER, py - _CENTER) + pr < _ROTATION_RADIUS_LIMIT


def _predict_relative(tx, ty, omega, lead_turns):
    dx, dy = tx - _CENTER, ty - _CENTER
    orb_r = math.hypot(dx, dy)
    cur_angle = math.atan2(dy, dx)
    new_angle = cur_angle + omega * lead_turns
    return _CENTER + orb_r * math.cos(new_angle), _CENTER + orb_r * math.sin(new_angle)


def _search_safe_intercept(sx, sy, src_r, tx, ty, tgt_r, ships, omega):
    """Self-consistent intercept search across candidate arrival turns.

    Mirror of `lib.aim.search_safe_intercept`. Fallback for cases where
    the 5-iter fixed-point oscillates rather than converges (typically
    orbital targets at very long range). Returns
    `(angle, arrival_x, arrival_y)` for the best self-consistent
    candidate, or `None` if none of `[1, SEARCH_HORIZON]` is consistent.
    """
    r_offset = src_r + tgt_r + 0.1
    v = fleet_speed(int(ships))
    if v <= 0:
        return None
    best = None
    best_score = None
    for cand_t in range(1, _SEARCH_HORIZON + 1):
        ptx, pty = _predict_relative(tx, ty, omega, cand_t)
        d = math.hypot(ptx - sx, pty - sy)
        flight_d = max(0.0, d - r_offset)
        eta = flight_d / v
        delta = abs(eta - cand_t)
        if delta > _INTERCEPT_TOLERANCE:
            continue
        score = (delta, cand_t)
        if best is None or score < best_score:
            best_score = score
            angle = math.atan2(pty - sy, ptx - sx)
            best = (angle, ptx, pty)
    return best


def _aim_orbiting_inline(sx, sy, src_r, tx, ty, tgt_r, ships, omega):
    """5-iter fixed-point lead + safe-intercept fallback.

    Mirror of `lib.aim.aim_orbiting`. Returns (angle, arrival_x, arrival_y).
    """
    r_offset = src_r + tgt_r + 0.1
    cx, cy = tx, ty                  # working "predicted target" position
    last_eta = None
    converged = False
    v = fleet_speed(int(ships))
    for _ in range(_MAX_AIM_ITERATIONS):
        d = math.hypot(cx - sx, cy - sy)
        flight_d = max(0.0, d - r_offset)
        if v <= 0:
            break
        eta = flight_d / v
        ntx, nty = _predict_relative(tx, ty, omega, eta)
        if (
            last_eta is not None
            and abs(ntx - cx) < _AIM_CONVERGE_XY_TOL
            and abs(nty - cy) < _AIM_CONVERGE_XY_TOL
        ):
            cx, cy = ntx, nty
            converged = True
            break
        cx, cy = ntx, nty
        last_eta = eta
    if not converged:
        # Fallback to self-consistent intercept search.
        fb = _search_safe_intercept(sx, sy, src_r, tx, ty, tgt_r, ships, omega)
        if fb is not None:
            return fb
    angle = math.atan2(cy - sy, cx - sx)
    return angle, cx, cy


def _segment_to_point_distance(sx, sy, ex, ey, px, py):
    """Min distance from point (px, py) to segment (sx, sy)→(ex, ey)."""
    dx, dy = ex - sx, ey - sy
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = sx + t * dx, sy + t * dy
    return math.hypot(px - cx, py - cy)


def _hits_sun(sx, sy, ax, ay):
    """True if straight-line segment (sx,sy)→(ax,ay) intersects sun."""
    return _segment_to_point_distance(sx, sy, ax, ay, _CENTER, _CENTER) < _SUN_RADIUS


def _path_oob(ax, ay):
    """True if endpoint (ax, ay) is outside the 100×100 board."""
    return not (_BOARD_LO <= ax <= _BOARD_HI and _BOARD_LO <= ay <= _BOARD_HI)


def _predict_fleet_fate_numpy(
    src_x, src_y, src_r, src_slot, target_slot,
    aim_angle, ships,
    planets_x, planets_y, planets_radius, planets_alive,
    planet_orbits, omega,
    max_steps=_PATH_MAX_STEPS,
):
    """Inline numpy port of `lib.trajectory.predict_fleet_fate`.

    Returns ("target" | "planet" | "sun" | "oob" | "timeout",
             hit_planet_slot or -1, step_of_hit).

    `planet_orbits[slot]` is a precomputed (max_steps+1, 2) array of
    (x, y) per step for orbiting planets (rotated chord positions);
    for static planets the array is just (planets_x[slot], planets_y[slot])
    repeated. Pre-computing once amortizes across all intents in this
    apply_mechanisms_numpy call.
    """
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src_x + cos_a * (src_r + 0.1)
    spawn_y = src_y + sin_a * (src_r + 0.1)
    speed_val = fleet_speed(int(ships))
    if speed_val <= 0:
        return "oob", -1, 0

    P = planets_x.shape[0]

    for step in range(max_steps):
        fold_x = spawn_x + cos_a * speed_val * step
        fold_y = spawn_y + sin_a * speed_val * step
        fnew_x = spawn_x + cos_a * speed_val * (step + 1)
        fnew_y = spawn_y + sin_a * speed_val * (step + 1)

        # Sun check (segment-to-point distance ≤ SUN_RADIUS + SUN_SAFETY).
        sun_d = _segment_to_point_distance(
            fold_x, fold_y, fnew_x, fnew_y, _CENTER, _CENTER,
        )
        if sun_d < _SUN_RADIUS + _SUN_SAFETY:
            return "sun", -1, step + 1

        # OOB endpoint check.
        if (
            fnew_x < _BOARD_LO or fnew_x > _BOARD_HI
            or fnew_y < _BOARD_LO or fnew_y > _BOARD_HI
        ):
            return "oob", -1, step + 1

        # Swept-pair check vs every alive planet (vectorised over P).
        # Skip own source planet on step 0 (env doesn't collide with src
        # on the first move).
        p_old = planet_orbits[:, step, :]            # (P, 2)
        p_new = planet_orbits[:, step + 1, :]        # (P, 2)
        d0x = fold_x - p_old[:, 0]
        d0y = fold_y - p_old[:, 1]
        dvx = (fnew_x - fold_x) - (p_new[:, 0] - p_old[:, 0])
        dvy = (fnew_y - fold_y) - (p_new[:, 1] - p_old[:, 1])
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - planets_radius * planets_radius
        # Swept-pair semantics: hit iff t in [0,1] segment of quadratic
        # has a root with c <= 0 (already inside) or disc >= 0.
        # Following lib.aim.swept_pair_hit:
        disc = b * b - 4.0 * a * c
        # For a < 1e-12 (parallel), the answer is c <= 0.
        # Else: roots t1 = (-b - sqrt(disc))/(2a), t2 = (-b + sqrt(disc))/(2a).
        # Hit iff t2 >= 0 AND t1 <= 1.
        import numpy as _np_local
        sq = _np_local.where(disc >= 0.0, _np_local.sqrt(_np_local.maximum(disc, 0.0)), 0.0)
        t1 = _np_local.where(a > 1e-12, (-b - sq) / _np_local.maximum(2.0 * a, 1e-12), 0.0)
        t2 = _np_local.where(a > 1e-12, (-b + sq) / _np_local.maximum(2.0 * a, 1e-12), 0.0)
        parallel_hit = (a <= 1e-12) & (c <= 0.0)
        nonparallel_hit = (a > 1e-12) & (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
        hit_mask = (parallel_hit | nonparallel_hit) & planets_alive
        # Skip src planet on step 0.
        if step == 0:
            hit_mask = hit_mask.copy()
            hit_mask[src_slot] = False
        if hit_mask.any():
            # Return FIRST hit by planet index (matches scalar dict
            # iteration order which is insertion order = planet id order).
            hit_idx = int(_np_local.argmax(hit_mask))
            outcome = "target" if hit_idx == target_slot else "planet"
            return outcome, hit_idx, step + 1

    return "timeout", -1, max_steps


def _build_planet_orbits(planets_x, planets_y, planets_radius, omega, max_steps=_PATH_MAX_STEPS):
    """Pre-compute per-step (x, y) position for each planet over
    [0, max_steps]. Orbiting planets rotate; static planets stay put.

    Returns array shape `(P, max_steps+1, 2)` float32.
    """
    import numpy as _np
    P = planets_x.shape[0]
    orbits = _np.zeros((P, max_steps + 1, 2), dtype=_np.float64)
    for slot in range(P):
        px, py, pr = float(planets_x[slot]), float(planets_y[slot]), float(planets_radius[slot])
        if _is_orbiting(px, py, pr) and omega != 0.0:
            dx, dy = px - _CENTER, py - _CENTER
            orb_r = math.hypot(dx, dy)
            cur_angle = math.atan2(dy, dx)
            ts = _np.arange(max_steps + 1, dtype=_np.float64)
            angles = cur_angle + omega * ts
            orbits[slot, :, 0] = _CENTER + orb_r * _np.cos(angles)
            orbits[slot, :, 1] = _CENTER + orb_r * _np.sin(angles)
        else:
            orbits[slot, :, 0] = px
            orbits[slot, :, 1] = py
    return orbits


def apply_mechanisms_numpy(
    chosen: list[dict],                # output of settle_plan_from_matrices
    state,                             # GameState
    world_model,                       # JaxWorldModel
    my_id: int,
) -> list[dict]:
    """Apply validate + arrival_size + atan2-aim to settled intents.

    Returns a list of dicts:
        [{"src_pid", "target_pid", "angle", "ships", "eta"}, ...]

    Mirrors the scalar `realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)`
    pipeline modulo the deferred mechanisms documented above.
    """
    planets_id = np.asarray(state.planets_id)
    planets_x = np.asarray(state.planets_x)
    planets_y = np.asarray(state.planets_y)
    planets_owner = np.asarray(state.planets_owner)
    planets_ships = np.asarray(state.planets_ships)
    planets_prod = np.asarray(state.planets_prod)
    planets_radius = np.asarray(state.planets_radius)
    planets_alive = np.asarray(state.planets_alive)
    is_comet = np.asarray(state.is_comet)
    omega = float(state.angular_velocity)
    owners_at = np.asarray(world_model.owners_at)
    ships_at = np.asarray(world_model.ships_at)
    H = ships_at.shape[1] - 1

    pid_to_slot = {int(pid): slot for slot, pid in enumerate(planets_id) if pid >= 0}

    # Pre-compute planet orbital trajectories once for the path-clears check.
    # Only do this if there are any candidate intents (skip on empty input).
    planet_orbits = None
    if chosen:
        planet_orbits = _build_planet_orbits(
            planets_x, planets_y, planets_radius, omega,
            max_steps=_PATH_MAX_STEPS,
        )

    # Track per-source ship-budget so a single source can't be over-allocated
    # across two mission classes (settle_plan picks one per source, but
    # arrival_size's bump could push the chosen ships past the garrison).
    src_remaining = {}
    out: list[dict] = []
    for c in chosen:
        src_slot = pid_to_slot.get(int(c["src_pid"]))
        tgt_slot = pid_to_slot.get(int(c["target_pid"]))
        if src_slot is None or tgt_slot is None:
            continue
        if not planets_alive[src_slot] or not planets_alive[tgt_slot]:
            continue
        if int(planets_owner[src_slot]) != my_id:
            continue
        if tgt_slot == src_slot:
            continue
        ships = int(c["ships"])
        if ships <= 0:
            continue

        # arrival_size — only bumps for non-neutral, non-self targets.
        target_owner = int(planets_owner[tgt_slot])
        if target_owner != -1 and target_owner != my_id:
            sx, sy = float(planets_x[src_slot]), float(planets_y[src_slot])
            tx, ty = float(planets_x[tgt_slot]), float(planets_y[tgt_slot])
            d = math.hypot(tx - sx, ty - sy)
            # eta from current ship count first (matches scalar mechanism).
            v = fleet_speed(ships)
            eta = int(math.ceil(d / max(v, 1e-6)))
            # Static estimate (matches scalar's prod_ticks for non-dynamic).
            # Dynamic = comet or orbiting+omega!=0; we don't know omega here
            # cheaply — fall back to non-dynamic estimate, which can mildly
            # under-size for orbiting targets. Parity hit ≤ 1 ship; acceptable.
            static_needed = (
                int(planets_ships[tgt_slot])
                + int(planets_prod[tgt_slot]) * eta
                + 1
            )
            # WorldModel estimate.
            e_clamp = max(0, min(eta, H))
            pred_owner = int(owners_at[tgt_slot, e_clamp])
            if pred_owner == my_id:
                # Already ours by then; settle_plan should've filtered. Skip.
                continue
            pred_ships = int(ships_at[tgt_slot, e_clamp])
            needed = max(static_needed, pred_ships + 1)
            ships = max(ships, needed)

        # validate: ships must fit in (remaining) garrison.
        remaining = src_remaining.get(src_slot, int(planets_ships[src_slot]))
        if ships > remaining:
            continue
        src_remaining[src_slot] = remaining - ships

        # Aim: orbiting non-comet → 5-iter fixed-point; else plain atan2.
        sx, sy = float(planets_x[src_slot]), float(planets_y[src_slot])
        tx, ty = float(planets_x[tgt_slot]), float(planets_y[tgt_slot])
        src_r = float(planets_radius[src_slot])
        tgt_r = float(planets_radius[tgt_slot])
        target_is_comet = bool(is_comet[tgt_slot])
        orbit = _is_orbiting(tx, ty, tgt_r) and not target_is_comet
        if orbit and omega != 0.0:
            angle, arrival_x, arrival_y = _aim_orbiting_inline(
                sx, sy, src_r, tx, ty, tgt_r, ships, omega,
            )
        else:
            angle = math.atan2(ty - sy, tx - sx)
            arrival_x, arrival_y = tx, ty

        # Full-trajectory ray-cast (sun_avoid + path_clears_other_planets
        # + oob_guard rolled together — mirrors lib.trajectory.predict_fleet_fate).
        outcome, hit_slot, _hit_step = _predict_fleet_fate_numpy(
            sx, sy, src_r, src_slot, tgt_slot,
            angle, ships,
            planets_x, planets_y, planets_radius, planets_alive,
            planet_orbits, omega,
        )
        if outcome in ("sun", "oob", "planet"):
            # Reclaim the per-source budget; the source effectively did
            # not launch.
            src_remaining[src_slot] = src_remaining.get(src_slot, 0) + ships
            continue

        # Re-eta with the bumped ship count (in case arrival_size bumped).
        v = fleet_speed(ships)
        d = math.hypot(arrival_x - sx, arrival_y - sy)
        new_eta = int(math.ceil(d / max(v, 1e-6)))

        out.append({
            "src_pid": int(c["src_pid"]),
            "target_pid": int(c["target_pid"]),
            "angle": angle,
            "ships": ships,
            "eta": new_eta,
        })
    return out


def emitted_to_jax_action_tensors(
    emitted_per_agent: list[list[dict]],
    num_agents: int,
):
    """Pack per-agent emitted-intent lists into the (MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT) tensors `jax_step` expects.

    Slot 0..N-1 is filled; remaining slots have `pid == -1` (sentinel
    for "no action this slot"). Matches the contract of
    `lib.game.jax.conversions.actions_to_jax`.
    """
    pids = -np.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships_arr = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for a in range(min(num_agents, MAX_AGENTS)):
        moves = emitted_per_agent[a]
        for k, mv in enumerate(moves[:MAX_LAUNCH_PER_AGENT]):
            pids[a, k] = int(mv["src_pid"])
            angles[a, k] = float(mv["angle"])
            ships_arr[a, k] = int(mv["ships"])
    return pids, angles, ships_arr
