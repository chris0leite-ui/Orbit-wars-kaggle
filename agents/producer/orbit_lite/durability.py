"""Tenure / durability leaf-scorer term for producer_plus.

Discounts a capture by how long we can actually HOLD the target — the
positional-game *durability* factor. The champion values a capture's
production stream as if tenure were always full; this term prices the
**net force balance** at the target so captures we cannot keep (the
collapse/churn loss driver) score lower, conserving force for keepable
positions.

It is the closest sibling of ``recapture.recapture_penalty`` and shares
its reach/threat machinery, with one addition that makes it a *tenure*
term rather than a pure threat penalty: it subtracts **our own
reinforcement reach** from the enemy threat, so a contested planet a
friendly neighbour can defend is NOT penalised, while a thin capture of a
contested planet with no reinforcement in reach is.

This is capture-SELECTION shaping (which captures to make), not global
defense — the distinction from the ``garval`` direction that over-defended
and lost on the ladder. Do NOT enable alongside ``recapture_penalty``
(double-penalty); tenure is the more complete form.

Math, per candidate ``c`` capturing target ``T`` at arrival tick ``e_c``
with send ``s_c``, over a hold window ``W`` turns:

    defender_c    = max(0, s_c - capture_floor_TK[t, e_c - 1])
    captures_c    = (s_c >= floor) & valid & ~is_def & ~own_at_arrival
    reach(p,T)    = ceil(cross_dist[1, p, T] / fleet_speed(ships[p]))

    enemy_force_T = sum_p in (enemy reaching T within W) of
                    (1 - safety) * ships_p + prod_p * reach(p, T)
    friend_reach_T= sum_q in (ours  reaching T within W, q != source) of
                    hold_fraction * ships_q
    exposure_c    = max(0, enemy_force_T - defender_c - friend_reach_T)

    enemy_min_T   = min reach(p, T) over enemy reaching T within W
    turns_lost_c  = max(0, H - e_c - enemy_min_T)
    penalty_c     = captures_c * (exposure_c > 0) * prod[T] * turns_lost_c * weight

``penalty_c >= 0`` is SUBTRACTED from the candidate score (same sign as
recapture_penalty). Default-OFF at the caller; import-safe in the
byte-identical path.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .distance_cache import DistanceCache
from .geometry import fleet_speed
from .movement import PlanetGarrisonStatus


def tenure_penalty(
    *,
    obs,
    cache: DistanceCache,
    garrison_status: PlanetGarrisonStatus,
    cand_tgt_slot: Tensor,       # [C] long — absolute planet index
    cand_tgt_short: Tensor,      # [C] long — short index into capture_floor_TK
    cand_send: Tensor,           # [C, L] float — read [:, 0]
    cand_eta: Tensor,            # [C, L] float — read [:, 0]
    cand_valid: Tensor,          # [C] bool
    cand_is_def: Tensor,         # [C] bool — own-planet reinforcement
    capture_floor_TK: Tensor,    # [T, K] from planner_core.capture_floor
    prod: Tensor,                # [P] float — per-planet production
    H: int,
    W: int,
    hold_fraction: float,
    safety_reserve: float,
    weight: float,
    player_id: int,
) -> Tensor:
    """Return per-candidate tenure penalty in ship units. ``[C]``, all >= 0."""
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    assert garrison_status.owner.device == device, (
        f"garrison_status.owner on {garrison_status.owner.device}, expected {device}"
    )

    W_eff = min(int(W), int(cache.K))
    if W_eff <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    # ----- per-(planet, target) reach time -----
    # k=1 immediate cross-time distance (exact for static; conservative for
    # orbitals — over-estimates distance => under-estimates threat => leans safe).
    d_immediate = cache.cross_dist[1].to(dtype)                  # [P_src, P_tgt]
    speeds = fleet_speed(obs.ships.clamp(min=1.0)).to(dtype)     # [P]
    reach_pT = torch.ceil(
        (d_immediate / speeds.view(P, 1).clamp(min=1e-6)).clamp(min=1.0)
    )                                                            # [P_src, P_tgt]

    self_mask = torch.eye(P, dtype=torch.bool, device=device)
    alive = obs.alive.to(device)
    enemy_alive = (obs.is_enemy & obs.alive).to(device)
    ours_alive = (obs.owned & obs.alive).to(device)
    within = reach_pT <= float(W_eff)

    can_enemy = within & enemy_alive.view(P, 1) & alive.view(1, P) & ~self_mask
    can_friend = within & ours_alive.view(P, 1) & alive.view(1, P) & ~self_mask

    ships_f = obs.ships.to(dtype)
    prod_f = prod.to(dtype)
    safety = float(max(0.0, min(1.0, safety_reserve)))
    hf = float(max(0.0, min(1.0, hold_fraction)))

    # Enemy attacking force that can reach T within the window.
    enemy_force_pT = torch.where(
        can_enemy,
        (1.0 - safety) * ships_f.view(P, 1) + prod_f.view(P, 1) * reach_pT,
        torch.zeros_like(reach_pT),
    )
    enemy_force_P = enemy_force_pT.sum(dim=0)                    # [P]

    # Our reinforcement force that can reach T within the window (the tenure
    # addition: a contested planet we can defend is not penalised).
    friend_reach_pT = torch.where(
        can_friend,
        hf * ships_f.view(P, 1),
        torch.zeros_like(reach_pT),
    )
    friend_reach_P = friend_reach_pT.sum(dim=0)                  # [P]

    inf_v = torch.full_like(reach_pT, float("inf"))
    enemy_min_P = torch.where(can_enemy, reach_pT, inf_v).amin(dim=0)   # [P]

    # ----- per-candidate capture detection + defender (mirrors recapture) -----
    K_floor = int(capture_floor_TK.shape[-1])
    if K_floor <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    e_send = cand_send[:, 0].to(dtype)
    e_eta = cand_eta[:, 0].to(dtype)
    e_idx = (torch.ceil(e_eta).long() - 1).clamp(0, K_floor - 1)
    tshort = cand_tgt_short.clamp(0, max(int(capture_floor_TK.shape[0]) - 1, 0)).to(device)
    floor_c = capture_floor_TK[tshort, e_idx]
    defender_c = (e_send - floor_c).clamp(min=0.0)

    tgt_abs = cand_tgt_slot.clamp(0, P - 1).to(device)
    owner_axis_H = int(garrison_status.owner.shape[-1])
    own_idx = torch.ceil(e_eta).long().clamp(0, max(owner_axis_H - 1, 0))
    own_at_arrival = garrison_status.owner[tgt_abs, own_idx] == int(player_id)
    captures_c = (
        (e_send >= floor_c)
        & cand_valid.to(device)
        & ~cand_is_def.to(device)
        & ~own_at_arrival
    )

    # ----- exposure + penalty -----
    enemy_force_c = enemy_force_P[tgt_abs]
    friend_reach_c = friend_reach_P[tgt_abs]
    enemy_min_c = enemy_min_P[tgt_abs]
    prod_c = prod_f[tgt_abs]

    exposure_c = (enemy_force_c - defender_c - friend_reach_c).clamp(min=0.0)
    # When no enemy reaches, enemy_min_c is +inf so H - e - inf = -inf -> clamp 0.
    turns_lost_c = (float(H) - torch.ceil(e_eta) - enemy_min_c).clamp(min=0.0)

    penalty_c = (
        captures_c.to(dtype)
        * (exposure_c > 0).to(dtype)
        * prod_c
        * turns_lost_c
        * float(weight)
    )
    return penalty_c.clamp(min=0.0)
