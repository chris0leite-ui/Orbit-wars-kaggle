"""Lead-prediction aim helpers — 5-iter fixed-point + safe-intercept fallback.

Adapted from public-kernel patterns (Roman 1224 §Physics) into our pipeline.
The Mechanism layer (`lib/mechanism.lead_aim_v2`) calls these functions;
keeping them in this module makes them easy to unit-test in isolation and
to reuse from a future `search_safe_intercept` mechanism / planner layer.

Conventions:
- Coordinates are floats in [0, 100] (board units); see `lib/geometry`.
- Fleet speed obeys `lib/fleet.speed` (log-curve, clamp at 1000 ships).
- Orbiting planet position at `lead_turns` is predicted by
  `lib/orbit.predict_relative` (no step counter needed; rotates current
  obs forward by `omega * lead_turns`).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.orbit import predict_relative, predict_relative_smart

# Tolerance bands tuned from public kernel patterns (Roman §K).
INTERCEPT_TOLERANCE = 1        # +/- step delta between predicted and candidate
SEARCH_HORIZON = 60            # max future steps to scan in safe-intercept
CONVERGENCE_XY_TOL = 0.3       # |dx|, |dy| convergence threshold in fixed-point
MAX_ITERATIONS = 5             # fixed-point iter cap (was 1 in v1, 2 in lead_aim)


def flight_distance(src_xy, src_radius, target_xy, target_radius):
    """Center-to-center distance minus launch offset minus capture radius.

    The env spawns the fleet just outside the source at `r_src + 0.1`, and
    captures when the fleet enters `target.radius`. So the actual *flight*
    distance is `dist(src, target) - r_src - r_target - 0.1`, clamped at 0
    to avoid negative ETA for degenerate launches at the source.
    """
    d = math.hypot(target_xy[0] - src_xy[0], target_xy[1] - src_xy[1])
    return max(0.0, d - src_radius - target_radius - 0.1)


def estimate_eta(src_xy, src_radius, target_xy, target_radius, ships):
    """Floating-point ETA in steps. None if speed is degenerate.

    Returns the FLOAT step count to traverse the flight distance at the
    fleet's speed (a function of ships via the log-curve). Callers ceil
    or floor as their step semantics require.
    """
    flight = flight_distance(src_xy, src_radius, target_xy, target_radius)
    v = fleet_speed(ships)
    if v <= 0:
        return None
    return flight / v


def search_safe_intercept(
    src_xy,
    src_radius,
    target_tuple,
    target_radius,
    ships,
    omega,
    horizon=SEARCH_HORIZON,
):
    """Self-consistent intercept search over candidate arrival turns.

    For each `candidate_turns` in `[1, horizon]`:
    1. Predict target position at `candidate_turns` (orbital projection).
    2. Estimate ETA from current source to that predicted position.
    3. If |ETA - candidate_turns| <= INTERCEPT_TOLERANCE, the candidate is
       self-consistent — i.e. firing at it now would actually intersect
       the target at the predicted step.

    Returns (aim_angle, arrival_xy, eta) for the best (smallest delta,
    then smallest turn count) candidate, or None if no candidate is
    self-consistent.

    This is the fallback for cases where 5-iter fixed-point doesn't
    converge — typically orbital targets at very long distances where
    `eta` oscillates between two values.
    """
    best = None
    best_score = None
    for cand_t in range(1, horizon + 1):
        pred_xy = predict_relative_smart(target_tuple, omega, cand_t)
        eta = estimate_eta(src_xy, src_radius, pred_xy, target_radius, ships)
        if eta is None:
            continue
        delta = abs(eta - cand_t)
        if delta > INTERCEPT_TOLERANCE:
            continue
        score = (delta, cand_t)
        if best is None or score < best_score:
            best_score = score
            angle = math.atan2(pred_xy[1] - src_xy[1], pred_xy[0] - src_xy[0])
            best = (angle, pred_xy, eta)
    return best


def aim_orbiting(src_xy, src_radius, target_tuple, target_radius, ships, omega):
    """5-iter fixed-point lead for orbiting non-comet targets, with
    safe-intercept fallback when iteration doesn't converge.

    Returns (aim_angle, arrival_xy, eta) or None if no valid intercept.

    Algorithm:
    1. Start with target's current position.
    2. Estimate ETA to current target position.
    3. Predict target position at that ETA via orbital projection.
    4. Re-estimate ETA to the predicted position.
    5. Repeat up to MAX_ITERATIONS. Convergence = |dx|, |dy| < TOL.
    6. If converged, return; else fall back to search_safe_intercept.

    The fallback exists because at long ranges / fast orbital motion,
    the fixed-point can oscillate between two estimates rather than
    converge. Roman 1224 uses 5 iter + this fallback; we follow the
    pattern.
    """
    tx, ty = target_tuple[2], target_tuple[3]
    last_eta = None
    for _ in range(MAX_ITERATIONS):
        eta = estimate_eta(src_xy, src_radius, (tx, ty), target_radius, ships)
        if eta is None:
            return search_safe_intercept(
                src_xy, src_radius, target_tuple, target_radius, ships, omega,
            )
        ntx, nty = predict_relative_smart(target_tuple, omega, eta)
        if (
            last_eta is not None
            and abs(ntx - tx) < CONVERGENCE_XY_TOL
            and abs(nty - ty) < CONVERGENCE_XY_TOL
        ):
            angle = math.atan2(nty - src_xy[1], ntx - src_xy[0])
            return angle, (ntx, nty), eta
        tx, ty = ntx, nty
        last_eta = eta

    # Non-convergence → safe-intercept fallback.
    fb = search_safe_intercept(
        src_xy, src_radius, target_tuple, target_radius, ships, omega,
    )
    if fb is not None:
        return fb

    # Last resort: return final iteration's guess (better than nothing —
    # the fleet still launches; physics decides the outcome).
    angle = math.atan2(ty - src_xy[1], tx - src_xy[0])
    return angle, (tx, ty), last_eta or 0.0


def aim_comet(src_xy, src_radius, target_tuple, target_radius, ships,
              comet_path, comet_path_index):
    """5-iter fixed-point lead for COMET targets — path-indexed, NOT orbital.

    Sibling to `aim_orbiting`. Comets follow pre-computed polynomial paths
    at `cometSpeed=4` board-units/turn, not orbital rotation around the
    sun. Using `predict_relative` for comets mis-aims by 20-40 board
    units within ~7 turns (ep 77087563 / sub 52811320: 40 ships from
    planet 12 → comet 31, fleet OOB).

    Algorithm (mirrors `aim_orbiting`):
    1. Start with target's CURRENT position (path[index]).
    2. Estimate ETA to current target position.
    3. Predict target position at that ETA via path[index + ceil(eta)].
       Returns None if the comet exits the path before arrival.
    4. Re-estimate ETA to the predicted position.
    5. Repeat up to MAX_ITERATIONS. Convergence = |dx|, |dy| < TOL.
    6. If converged, return; else fall back to last-iteration guess.

    Returns (aim_angle, arrival_xy, eta) or None if the comet exits
    before any reachable intercept.

    `comet_path` is a list of `[x, y]` pairs; `comet_path_index` is the
    current path position (advances by 1 per turn in the env, see
    `orbit_wars.py:550`). Caller is responsible for fetching these via
    `lib.world_model.comet_position_at` or `_comet_paths_by_id`.
    """
    path_len = len(comet_path)
    base_idx = int(comet_path_index)
    if base_idx < 0 or base_idx >= path_len:
        return None

    # Start at the comet's current position (path[base_idx]).
    cur_pt = comet_path[base_idx]
    tx, ty = float(cur_pt[0]), float(cur_pt[1])
    last_eta: float | None = None

    for _ in range(MAX_ITERATIONS):
        eta = estimate_eta(src_xy, src_radius, (tx, ty), target_radius, ships)
        if eta is None:
            return None
        # Path-indexed position lookup at the predicted arrival step.
        lead = int(math.ceil(eta))
        future_idx = base_idx + lead
        if future_idx < 0 or future_idx >= path_len:
            # Comet has exited the board before we'd arrive — abort.
            return None
        future_pt = comet_path[future_idx]
        ntx, nty = float(future_pt[0]), float(future_pt[1])
        if (
            last_eta is not None
            and abs(ntx - tx) < CONVERGENCE_XY_TOL
            and abs(nty - ty) < CONVERGENCE_XY_TOL
        ):
            angle = math.atan2(nty - src_xy[1], ntx - src_xy[0])
            return angle, (ntx, nty), eta
        tx, ty = ntx, nty
        last_eta = eta

    # Non-convergence: return the last iteration's guess. The trajectory
    # filter / cost-parity filter will catch downstream misfires.
    angle = math.atan2(ty - src_xy[1], tx - src_xy[0])
    return angle, (tx, ty), last_eta or 0.0


def swept_pair_hit(A, B, P0, P1, r):
    """Mirror of the env's swept-pair collision check (orbit_wars.py:46-67).

    True iff a fleet moving A->B and a planet moving P0->P1 come within
    `r` of each other for some t in [0, 1]. Treats both motions as
    linear over the tick (planet rotation is linearised to its chord).

    Used by `path_clears_other_planets` to detect mid-flight collisions
    with non-target planets — the largest physics-loss bucket from the
    capture probe (10.7%).
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0
