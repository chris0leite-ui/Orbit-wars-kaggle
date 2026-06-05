"""Recapture-penalty leaf-scorer term for producer_plus.

For each candidate ``c`` that would capture target ``T`` at arrival tick
``e_c`` with ``s_c`` ships, compute a non-negative penalty in ship units
proportional to the opponent's plausible recapture-and-hold ability. The
penalty is subtracted from ``competitive_score`` to discount thin
captures the opponent can punish before the scorer horizon ends.

Composes additively with the existing scorer. The multi-tick opp
projection already debits us for opp launches in its projection window;
to avoid double-counting, the caller passes ``K_opp`` and we restrict
the recapture window to ticks past the projection (``K_recap_eff =
max(1, K_recap - K_opp)``).

Math summary, per candidate ``c`` with target short index ``t``,
absolute target slot ``T``, send size ``s_c``, arrival tick ``e_c``:

    floor_c    = capture_floor_TK[t, e_c - 1]
    captures_c = (s_c >= floor_c) & cand_valid & ~cand_is_def
                 & (we don't already own T at e_c)
    defender_c = max(0, s_c - floor_c)

For each enemy-owned alive planet ``p`` with distance ``d = cross_dist[1, p, T]``:
    reach_p_T  = ceil(d / fleet_speed(ships[p]))                  # turns to recapture
    can_reach  = reach_p_T <= K_recap_eff
    threat_p_T = (1 - safety_reserve) * ships[p] + prod[p] * reach_p_T
    threat_T   = sum_p (threat_p_T where can_reach)

    reach_recap_T = min_p reach_p_T where can_reach   (or +inf if none)
    deficit_c     = max(0, threat_T - defender_c)
    turns_lost_c  = max(0, H - e_c - reach_recap_T)
    penalty_c     = captures_c * (deficit_c > 0) * prod[T] * turns_lost_c

Default OFF behaviour at the caller (see ``_recapture_penalty_enabled``
in ``producer_plus/main.py``); this module is import-safe in the
single-pass byte-identical path.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .distance_cache import DistanceCache
from .geometry import fleet_speed
from .movement import PlanetGarrisonStatus


def recapture_penalty(
    *,
    obs,
    cache: DistanceCache,
    garrison_status: PlanetGarrisonStatus,
    cand_tgt_slot: Tensor,       # [C] long — absolute planet index
    cand_tgt_short: Tensor,      # [C] long — short index into target_idx
    cand_send: Tensor,           # [C, L] float — read [:, 0]
    cand_eta: Tensor,            # [C, L] float — read [:, 0]
    cand_valid: Tensor,          # [C] bool
    cand_is_def: Tensor,         # [C] bool — own-planet reinforcement
    target_idx: Tensor,          # [T] long — short -> absolute slot
    capture_floor_TK: Tensor,    # [T, K] from caller (planner_core.capture_floor)
    prod: Tensor,                # [P] float — per-planet production
    H: int,
    K_recap: int,
    K_opp: int,
    safety_reserve: float,
    player_id: int,
) -> Tensor:
    """Return per-candidate recapture penalty in ship units. Shape ``[C]``, all >= 0."""
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)

    if C == 0 or P == 0:
        return torch.zeros(C, dtype=dtype, device=device)

    # Effective recap window: only count ticks past what multi-tick opp_proj
    # already modeled (the scorer already saw those via the background launchset).
    K_recap_eff = max(1, int(K_recap) - max(0, int(K_opp)))
    K_recap_eff = min(K_recap_eff, int(cache.K))
    if K_recap_eff <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    # ----- per-(opp_planet, target) reach + threat -----
    # cross_dist at k=1 as immediate cross-time distance (exact for static
    # planets; conservative for orbitals — overestimates distance, so
    # underestimates threat → penalty leans safe).
    d_immediate = cache.cross_dist[1].to(dtype)                # [P_src, P_tgt]
    speeds = fleet_speed(obs.ships.clamp(min=1.0)).to(dtype)   # [P]
    # Reach time = ceil(distance / speed) per (src=p, tgt). At least 1 turn.
    reach_pT_f = (d_immediate / speeds.view(P, 1).clamp(min=1e-6)).clamp(min=1.0)
    reach_pT = torch.ceil(reach_pT_f).clamp(min=1.0)            # [P_src, P_tgt]

    enemy_alive = (obs.is_enemy & obs.alive).to(device)         # [P]
    self_mask = torch.eye(P, dtype=torch.bool, device=device)
    can_reach = (
        (reach_pT <= float(K_recap_eff))
        & enemy_alive.view(P, 1)
        & obs.alive.view(1, P).to(device)
        & ~self_mask
    )                                                            # [P_src, P_tgt]

    ships_f = obs.ships.to(dtype)
    prod_f = prod.to(dtype)
    safety = float(max(0.0, min(1.0, safety_reserve)))
    # threat = (1 - safety_reserve) * ships + prod * reach_time, masked.
    threat_pT_raw = (
        (1.0 - safety) * ships_f.view(P, 1)
        + prod_f.view(P, 1) * reach_pT
    )                                                            # [P_src, P_tgt]
    threat_pT = torch.where(can_reach, threat_pT_raw, torch.zeros_like(threat_pT_raw))
    threat_P = threat_pT.sum(dim=0)                              # [P] per planet

    inf_v = torch.full_like(reach_pT, float("inf"))
    reach_pT_masked = torch.where(can_reach, reach_pT, inf_v)
    reach_recap_P = reach_pT_masked.amin(dim=0)                  # [P]

    # ----- gather to per-candidate target -----
    # cand_tgt_short indexes into target_idx; threat / reach are per-planet,
    # so resolve via the absolute slot.
    tgt_abs = cand_tgt_slot.clamp(0, P - 1).to(device)           # [C]
    threat_c = threat_P[tgt_abs]                                 # [C]
    reach_recap_c = reach_recap_P[tgt_abs]                       # [C]
    prod_c = prod_f[tgt_abs]                                     # [C]

    # ----- per-candidate floor + capture detection -----
    K_floor = int(capture_floor_TK.shape[-1])
    if K_floor <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    e_send = cand_send[:, 0].to(dtype)                            # [C] ships
    e_eta = cand_eta[:, 0].to(dtype)
    e_idx = (torch.ceil(e_eta).long() - 1).clamp(0, K_floor - 1)  # [C] eta -> K index

    tshort = cand_tgt_short.clamp(0, max(int(capture_floor_TK.shape[0]) - 1, 0)).to(device)
    floor_c = capture_floor_TK[tshort, e_idx]                     # [C]
    defender_c = (e_send - floor_c).clamp(min=0.0)                # [C]

    # We capture only if send >= floor AND we don't already own at arrival tick.
    # garrison_status.owner is the do-nothing trajectory; if we already own T
    # at e_c there's no capture happening (it's reinforcement, handled by
    # cand_is_def). Be defensive and gate on it explicitly.
    owner_axis_H = int(garrison_status.owner.shape[-1])
    own_idx = (torch.ceil(e_eta).long()).clamp(0, max(owner_axis_H - 1, 0))
    own_at_arrival = (
        garrison_status.owner[tgt_abs, own_idx] == int(player_id)
    )                                                              # [C]

    captures_c = (
        (e_send >= floor_c)
        & cand_valid.to(device)
        & ~cand_is_def.to(device)
        & ~own_at_arrival
    )                                                              # [C]

    # ----- penalty -----
    deficit_c = (threat_c - defender_c).clamp(min=0.0)
    # When no enemy can reach (reach_recap_c is +inf), turns_lost falls to 0.
    e_eta_ceil = torch.ceil(e_eta)
    turns_lost_c = (float(H) - e_eta_ceil - reach_recap_c).clamp(min=0.0)
    # Replace +inf*0 = NaN with 0 explicitly.
    turns_lost_c = torch.where(
        torch.isfinite(reach_recap_c), turns_lost_c, torch.zeros_like(turns_lost_c)
    )

    deficit_signal = (deficit_c > 0).to(dtype)
    penalty_c = captures_c.to(dtype) * deficit_signal * prod_c * turns_lost_c
    return penalty_c.clamp(min=0.0)
