"""Vectorized lite feature encoder for the distilled-Tier-2 opp policy.

Computes the 34 *cheap* features from `lib.shot_features.encode_features` —
those that depend ONLY on the raw `obs` planet/fleet arrays, NOT on
`lib.intent.World` / `lib.world_model.WorldModel`. The 11 features that
*do* need WorldModel (indices 6-8, 24, 25-28, 31-33 — F3, F2, F6, F8, F4)
are skipped entirely.

Why: `trained_logreg_policy` benched at 24.6 ms median (TOP_K=8) and 7 ms
median (TOP_K=2) because each per-candidate `encode_shot_features` call
internally invoked `predict_fleet_fate`, `pv_horizon`, `expected_hold`,
and `predict_garrison_at`. Even with the World+WorldModel cached and
shared across candidates, the per-candidate Tier-2 feature math was
~0.3 ms; at 64 candidates that's 19 ms.

This encoder ALWAYS skips that work. The booster is retrained on the
sliced 34-d corpus (`X[:, LITE_KEEP]` from the existing 45-d labels).

Target: ≤ 1 ms median per opp-policy call — comparable to Tier 0
`lite_greedy` (0.02 ms) without the on-policy quality cliff.

Layout of the 34 kept indices (preserving the original 45-d slot
positions for traceability; the *output* of `encode_lite_batch` is
the dense 34-d vector in the order below):

  in-corpus 45-d  →  output 34-d  feature
       0,1,2       →   0,1,2     sps_ships, sps_prod, sps_rad
       3,4,5       →   3,4,5     tgt_ships, tgt_prod, tgt_rad
       9,10,11     →   6,7,8     shot_ships, shot_frac, shot_dist
       12,13       →   9,10      shot_eta, shot_fs
       14,15,16,17 →  11,12,13,14 in-flight n/ships allied/enemy
       18-23       →  15-20      meta (turn, my/enemy ships, diff, my/enemy pc)
       29,30       →  21,22      F10 friendly inflight n/ships
       34          →  23         F11 joint arrival count
       35          →  24         F7 intercept enemy eta (norm)
       36          →  25         F13 growth-field diff (signed)
       37,38       →  26,27      F9 source threat (n, frontier)
       39,40       →  28,29      enemy inflight n/ships to target
       41,42       →  30,31      post-capture nearest-enemy dist/prod
       43,44       →  32,33      src/tgt orbital flags
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

LITE_FEATURE_DIM = 30

# Slice mapping from the 45-d corpus to the 30-d lite corpus. Used at
# training time: `X_lite = X_full[:, LITE_KEEP_INDICES]`.
#
# Dropped from the 45-d schema:
#  - 11 WorldModel-dependent features (F2/F3/F4/F6/F8): same as before.
#  - F9 src_threat (37, 38): uses world_model.time_to_enemy_threat → WM.
#  - Post-capture nearest geometry (41, 42): uses world's predict_relative
#    for orbital planets → also WM.
# Feature-parity check vs slow encoder showed these had mean-abs-diff
# 0.1-1.0 (binary frontier flag flipped), unusable at inference.
LITE_KEEP_INDICES = np.asarray([
    0, 1, 2, 3, 4, 5,            # planet-static  → 0-5
    9, 10, 11, 12, 13,           # shot-static    → 6-10
    14, 15, 16, 17,              # in-flight totals → 11-14
    18, 19, 20, 21, 22, 23,      # meta           → 15-20
    29, 30,                      # F10 friendly inflight (n, ships) → 21-22
    34,                          # F11 joint arrival count at eta   → 23
    35,                          # F7 intercept enemy eta           → 24
    36,                          # F13 growth-field diff (signed)   → 25
    39, 40,                      # enemy inflight to target (n, ships) → 26-27
    43, 44,                      # orbital flags (src, tgt)         → 28-29
], dtype=np.int64)

assert LITE_KEEP_INDICES.size == LITE_FEATURE_DIM, (
    f"LITE_KEEP_INDICES has {LITE_KEEP_INDICES.size} entries, expected {LITE_FEATURE_DIM}"
)

# Normalization constants — must match `lib.shot_features.NORM` exactly so
# the slice from the existing corpus is comparable to lite-encoded inputs.
_MAX_SHIPS = 2000.0
_MAX_PROD = 5.0
_MAX_RADIUS = 3.0
_MAX_FLEET_SPEED = 6.0
_MAX_ETA = 200.0
_BOARD_DIAG = 141.42
_MAX_PLANETS = 40.0
_EP_STEPS = 500.0
_GROWTH_MAX = 5.0


def _fleet_speed(ships: float) -> float:
    """Mirror `lib.fleet.fleet_speed`."""
    if ships <= 0:
        return 0.0
    return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def _fleet_destinations(
    fleets_arr: np.ndarray,
    planets_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute each fleet's destination planet id + eta to that planet.

    Mirrors `lib.shot_features.infer_target_pid` (ray-cast from fleet
    position along the fleet's angle, pick the planet with smallest
    perpendicular distance among forward candidates).

    Args:
        fleets_arr: (F, 7) [id, owner, x, y, angle, src, ships]
        planets_arr: (P, 7) [id, owner, x, y, r, ships, prod]

    Returns:
        dest_pid: (F,) int64 — destination planet id, or -1 if no
                  forward candidate matched.
        dest_eta: (F,) float32 — turns until arrival (ceil(d / speed)),
                  or 0 if dest_pid == -1.
    """
    F = fleets_arr.shape[0]
    if F == 0 or planets_arr.shape[0] == 0:
        return (np.full(F, -1, dtype=np.int64),
                np.zeros(F, dtype=np.float32))

    fx = fleets_arr[:, 2].astype(np.float32)
    fy = fleets_arr[:, 3].astype(np.float32)
    fang = fleets_arr[:, 4].astype(np.float32)
    fships = fleets_arr[:, 6].astype(np.float32)
    cos_a = np.cos(fang)
    sin_a = np.sin(fang)

    pids = planets_arr[:, 0].astype(np.int64)
    px = planets_arr[:, 2].astype(np.float32)
    py = planets_arr[:, 3].astype(np.float32)

    # Pairwise displacements: (F, P)
    rx = px[None, :] - fx[:, None]
    ry = py[None, :] - fy[:, None]
    fwd = rx * cos_a[:, None] + ry * sin_a[:, None]
    # perpendicular = |r - fwd * d_hat|
    perp_x = rx - fwd * cos_a[:, None]
    perp_y = ry - fwd * sin_a[:, None]
    perp = np.sqrt(perp_x * perp_x + perp_y * perp_y)
    # Score = perp + 0.001 * fwd (matches lib.shot_features.infer_target_pid)
    score = perp + 0.001 * fwd
    # Mask out non-forward candidates (fwd <= 0) AND zero-distance (self).
    score[fwd <= 0] = np.inf
    score[(rx ** 2 + ry ** 2) < 1e-12] = np.inf

    # Best target per fleet.
    best_idx = score.argmin(axis=1)
    best_score = score[np.arange(F), best_idx]
    matched = best_score < np.inf

    dest_pid = np.where(matched, pids[best_idx], -1).astype(np.int64)

    # ETA: forward distance from fleet position to chosen target / fleet speed.
    chosen_rx = rx[np.arange(F), best_idx]
    chosen_ry = ry[np.arange(F), best_idx]
    chosen_d = np.sqrt(chosen_rx * chosen_rx + chosen_ry * chosen_ry)
    # speed = 1 + 5 * (log(ships) / log(1000)) ^ 1.5 ; zero when ships <= 0
    log_ships = np.log(np.maximum(fships, 1e-6))
    speed = np.where(
        fships > 0,
        1.0 + 5.0 * (log_ships / math.log(1000.0)) ** 1.5,
        np.zeros(F, dtype=np.float32),
    ).astype(np.float32)
    eta = np.where(
        (speed > 0) & matched,
        np.ceil(chosen_d / np.maximum(speed, 1e-6)),
        np.zeros(F, dtype=np.float32),
    ).astype(np.float32)
    return dest_pid, eta


def encode_lite_batch(
    planets_arr: np.ndarray,
    fleets_arr: np.ndarray,
    focal_seat: int,
    step: int,
    candidates: list[tuple[int, int, float, int]],
    *,
    angular_velocity: float = 0.0,
    initial_planets: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized per-candidate feature batch encoder.

    Args:
        planets_arr: (P, 7) np.ndarray of [id, owner, x, y, r, ships, prod]
        fleets_arr: (F, 7) np.ndarray of [id, owner, x, y, angle, src, ships]
                    or empty (0, 7) array.
        focal_seat: int — the seat the opp policy is acting on behalf of.
        step: int — current game step (for `meta_turn`).
        candidates: list of (src_pid, tgt_pid, angle, ships) tuples.
        angular_velocity: optional, for orbital flag computation.
        initial_planets: optional (P, 7), for orbital flag detection.

    Returns:
        (N, LITE_FEATURE_DIM) float32 array. N = len(candidates).
    """
    N = len(candidates)
    out = np.zeros((N, LITE_FEATURE_DIM), dtype=np.float32)
    if N == 0:
        return out

    P = planets_arr.shape[0]
    F = fleets_arr.shape[0] if fleets_arr.size else 0

    # Build pid → row-index dict for indexing candidates.
    by_id = {int(planets_arr[i, 0]): i for i in range(P)}

    # Per-candidate index arrays (vectorized lookup into planets_arr).
    src_pids = np.fromiter((c[0] for c in candidates), dtype=np.int64, count=N)
    tgt_pids = np.fromiter((c[1] for c in candidates), dtype=np.int64, count=N)
    cand_ships = np.fromiter(
        (c[3] for c in candidates), dtype=np.float32, count=N,
    )
    # `angle` from the candidate (straight-line aim) is currently unused
    # in the lite feature set; the encoder relies on (src, tgt) positions
    # instead of explicit angle. Kept in the signature for symmetry with
    # the training-time enumerate_candidates which propagates angle to
    # the bundler's policy call.

    src_idx = np.fromiter(
        (by_id.get(int(p), 0) for p in src_pids), dtype=np.int64, count=N,
    )
    tgt_idx = np.fromiter(
        (by_id.get(int(p), 0) for p in tgt_pids), dtype=np.int64, count=N,
    )

    src_rows = planets_arr[src_idx]   # (N, 7)
    tgt_rows = planets_arr[tgt_idx]   # (N, 7)

    sx = src_rows[:, 2].astype(np.float32)
    sy = src_rows[:, 3].astype(np.float32)
    tx = tgt_rows[:, 2].astype(np.float32)
    ty = tgt_rows[:, 3].astype(np.float32)
    s_rad = src_rows[:, 4].astype(np.float32)
    t_rad = tgt_rows[:, 4].astype(np.float32)
    s_ships = src_rows[:, 5].astype(np.float32)
    s_prod = src_rows[:, 6].astype(np.float32)
    t_ships = tgt_rows[:, 5].astype(np.float32)
    t_prod = tgt_rows[:, 6].astype(np.float32)
    t_owner = tgt_rows[:, 1].astype(np.float32)

    # 0-5 planet-static
    out[:, 0] = s_ships / _MAX_SHIPS
    out[:, 1] = s_prod / _MAX_PROD
    out[:, 2] = s_rad / _MAX_RADIUS
    out[:, 3] = t_ships / _MAX_SHIPS
    out[:, 4] = t_prod / _MAX_PROD
    out[:, 5] = t_rad / _MAX_RADIUS

    # 6-10 shot-static
    src_garrison = np.maximum(1.0, s_ships)
    out[:, 6] = np.minimum(1.0, cand_ships / _MAX_SHIPS)
    out[:, 7] = np.minimum(1.0, cand_ships / src_garrison)
    dx = tx - sx
    dy = ty - sy
    dist = np.sqrt(dx * dx + dy * dy)
    out[:, 8] = np.minimum(1.0, dist / _BOARD_DIAG)
    # fleet_speed is per-candidate, vectorize
    fs_vec = np.zeros(N, dtype=np.float32)
    positive_mask = cand_ships > 0
    if positive_mask.any():
        log_ships = np.log(np.maximum(cand_ships[positive_mask], 1e-6))
        fs_vec[positive_mask] = (
            1.0 + 5.0 * (log_ships / math.log(1000.0)) ** 1.5
        )
    eta_vec = np.where(
        fs_vec > 0,
        np.ceil(dist / np.maximum(fs_vec, 1e-6)).astype(np.float32),
        np.zeros(N, dtype=np.float32),
    )
    out[:, 9] = np.minimum(1.0, eta_vec / _MAX_ETA)
    out[:, 10] = np.minimum(1.0, fs_vec / _MAX_FLEET_SPEED)

    # 11-14 in-flight totals (per-call constants, broadcast)
    if F > 0:
        f_owner = fleets_arr[:, 1].astype(np.int64)
        f_ships = fleets_arr[:, 6].astype(np.float32)
        allied_mask = f_owner == focal_seat
        enemy_mask = (f_owner != focal_seat) & (f_owner != -1)
        n_allied = int(allied_mask.sum())
        n_enemy = int(enemy_mask.sum())
        ship_allied = float(f_ships[allied_mask].sum())
        ship_enemy = float(f_ships[enemy_mask].sum())
    else:
        n_allied = n_enemy = 0
        ship_allied = ship_enemy = 0.0
        f_owner = np.empty(0, dtype=np.int64)
        f_ships = np.empty(0, dtype=np.float32)
    out[:, 11] = min(1.0, n_allied / _MAX_PLANETS)
    out[:, 12] = min(1.0, ship_allied / _MAX_SHIPS)
    out[:, 13] = min(1.0, n_enemy / _MAX_PLANETS)
    out[:, 14] = min(1.0, ship_enemy / _MAX_SHIPS)

    # 15-20 meta — partly per-call (constants), one per-candidate slot (turn)
    p_owner = planets_arr[:, 1].astype(np.int64)
    p_ships = planets_arr[:, 5].astype(np.float32)
    my_planet_ships = float(p_ships[p_owner == focal_seat].sum())
    enemy_planet_ships = float(p_ships[(p_owner != focal_seat) & (p_owner != -1)].sum())
    my_total = my_planet_ships + ship_allied
    enemy_total = enemy_planet_ships + ship_enemy
    my_pc = int((p_owner == focal_seat).sum())
    enemy_pc = int(((p_owner != focal_seat) & (p_owner != -1)).sum())

    out[:, 15] = min(1.0, step / _EP_STEPS)
    # Slow encoder uses _MAX_SHIPS=2000 (not *4) for these norms.
    out[:, 16] = min(1.0, my_total / _MAX_SHIPS)
    out[:, 17] = min(1.0, enemy_total / _MAX_SHIPS)
    out[:, 18] = max(-1.0, min(1.0,
        (my_total - enemy_total) / _MAX_SHIPS
    ))
    out[:, 19] = min(1.0, my_pc / _MAX_PLANETS)
    out[:, 20] = min(1.0, enemy_pc / _MAX_PLANETS)

    # ----- Fleet-destination preprocessing (used by F10, F11, F7, enemy
    # inflight features). One ray-cast pass over all fleets; results
    # broadcast to per-candidate aggregates below.
    if F > 0:
        fleet_dest_pid, fleet_eta = _fleet_destinations(fleets_arr, planets_arr)
    else:
        fleet_dest_pid = np.empty(0, dtype=np.int64)
        fleet_eta = np.empty(0, dtype=np.float32)

    # 21-22 F10 friendly inflight to target (n, ships).
    if F > 0:
        friendly_to_tgt_n = np.zeros(N, dtype=np.float32)
        friendly_to_tgt_ships = np.zeros(N, dtype=np.float32)
        for i in range(N):
            tgt_pid_i = int(tgt_pids[i])
            mask = (fleet_dest_pid == tgt_pid_i) & allied_mask
            friendly_to_tgt_n[i] = float(mask.sum())
            friendly_to_tgt_ships[i] = float(f_ships[mask].sum())
        out[:, 21] = np.minimum(1.0, friendly_to_tgt_n / _MAX_PLANETS)
        out[:, 22] = np.minimum(1.0, friendly_to_tgt_ships / _MAX_SHIPS)

    # 23 F11 joint arrival count: number of fleets arriving at target at
    # the same eta as the candidate emit. Cheap window: |delta_eta| <= 2.
    if F > 0:
        joint_arr = np.zeros(N, dtype=np.float32)
        for i in range(N):
            tgt_pid_i = int(tgt_pids[i])
            cand_eta_i = float(eta_vec[i])
            same_target = fleet_dest_pid == tgt_pid_i
            close_eta = np.abs(fleet_eta - cand_eta_i) <= 2.0
            joint_arr[i] = float((same_target & close_eta).sum())
        out[:, 23] = np.minimum(1.0, joint_arr / _MAX_PLANETS)

    # 24 F7 intercept enemy eta: for enemy fleets heading to the candidate
    # target, the min arrival eta (normalized). 1.0 if no such fleet.
    if F > 0:
        intercept_norm = np.ones(N, dtype=np.float32)
        for i in range(N):
            tgt_pid_i = int(tgt_pids[i])
            mask = (fleet_dest_pid == tgt_pid_i) & enemy_mask
            if mask.any():
                min_eta = float(fleet_eta[mask].min())
                intercept_norm[i] = min(1.0, min_eta / _MAX_ETA)
        out[:, 24] = intercept_norm
    else:
        out[:, 24] = 1.0

    # 25 F13 growth-field diff — per-call constant, broadcast.
    if P > 1:
        px = planets_arr[:, 2].astype(np.float32)
        py = planets_arr[:, 3].astype(np.float32)
        pp = planets_arr[:, 6].astype(np.float32)
        dx_pp = px[:, None] - px[None, :]
        dy_pp = py[:, None] - py[None, :]
        d2 = dx_pp * dx_pp + dy_pp * dy_pp + 1.0
        field_per_planet = (pp[None, :] / d2).sum(axis=1) - pp / 1.0
        my_field = float(field_per_planet[p_owner == focal_seat].sum())
        enemy_field = float(
            field_per_planet[(p_owner != focal_seat) & (p_owner != -1)].sum()
        )
        diff = (my_field - enemy_field) / _GROWTH_MAX
        out[:, 25] = max(-1.0, min(1.0, float(diff)))

    # 26-27 enemy inflight to target (n, ships).
    if F > 0:
        enemy_to_tgt_n = np.zeros(N, dtype=np.float32)
        enemy_to_tgt_ships = np.zeros(N, dtype=np.float32)
        for i in range(N):
            tgt_pid_i = int(tgt_pids[i])
            mask = (fleet_dest_pid == tgt_pid_i) & enemy_mask
            enemy_to_tgt_n[i] = float(mask.sum())
            enemy_to_tgt_ships[i] = float(f_ships[mask].sum())
        out[:, 26] = np.minimum(1.0, enemy_to_tgt_n / _MAX_PLANETS)
        out[:, 27] = np.minimum(1.0, enemy_to_tgt_ships / _MAX_SHIPS)

    # 28-29 orbital flags. The slow encoder uses lib.orbit.is_orbiting
    # which checks the planet's `angular_velocity` (only set for orbital
    # bodies). When `initial_planets` is provided, position drift is a
    # reasonable proxy; without it, the slot stays 0.
    if initial_planets is not None and initial_planets.shape[0] == P:
        init_x = initial_planets[:, 2].astype(np.float32)
        init_y = initial_planets[:, 3].astype(np.float32)
        is_orb_per_pid = ((planets_arr[:, 2] - init_x) ** 2 +
                          (planets_arr[:, 3] - init_y) ** 2) > 1e-4
        out[:, 28] = is_orb_per_pid[src_idx].astype(np.float32)
        out[:, 29] = is_orb_per_pid[tgt_idx].astype(np.float32)

    return out


def planets_to_array(planets: list) -> np.ndarray:
    """Convert obs.planets (list of 7-tuples) to (P, 7) float32 numpy."""
    if not planets:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(planets, dtype=np.float32)


def fleets_to_array(fleets: list) -> np.ndarray:
    """Convert obs.fleets (list of 7-tuples) to (F, 7) float32 numpy."""
    if not fleets:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(fleets, dtype=np.float32)
