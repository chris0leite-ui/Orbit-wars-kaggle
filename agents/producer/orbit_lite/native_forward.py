"""Dropout-NATIVE forward model — mean-field flip-hazard ownership (Phase A).

The dropout BOLT-ON (producer_plus) values a capture by a 2-point blend (clean
score vs one reflipped score) over a static one-ply scorer. The continuous-score
A/B (state/DROPOUT_PLAN.md) showed that graft is SATURATED: its perturbation has
no distribution to refine. This module is the design's answer
(state/DROPOUT_NATIVE_DESIGN.md, Phase A): the forward model itself is a
distribution over ownership futures.

THE MODEL (mean-field, deterministic, NO RNG)
---------------------------------------------
1. Deterministic dynamics, unchanged: production accrues, our launches land,
   in-flight fleets resolve, combat at arrivals — exactly the engine recurrence
   (`_run_exact_recurrence`, reused). This yields, per candidate, the
   owner/garrison trajectory [C, P, H+1].
2. Flip HAZARD overlay: at each (planet p, step k) a planet I own is held with
   probability `1 - flip(atk_reach, garrison)`, where `atk_reach(p,k)` is the
   enemy's physically-routable mass that can reach p by step k and `garrison` is
   my projected ship count there. `flip = sigmoid(steepness·(atk-def)/(atk+def))`
   is steep near parity. This is the per-step Markov-ownership probability
   `P(I own p at k)` — a genuine continuous distribution over futures, not a
   2-point blend. (Phase B will calibrate `flip` to observed flip rates.)
3. Value functional = expected production-weighted ownership margin over the
   horizon:  `Σ_k discount^k Σ_p prod_p · (P_mine(p,k) − P_opp(p,k))`.
   A capture I cannot hold (high atk_reach, thin garrison) leaks its ownership
   probability to the opponent and earns little; a holdable capture earns the
   full production stream. This subsumes the bolt-on's reflip and the
   terminal-prod / hold-value hacks.

Everything is one deterministic pass over the batch axis — CPU/GPU bit-identical,
the codebase's hard requirement.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .garrison_launch import _run_exact_recurrence
from .geometry import fleet_speed


def build_candidate_trajectories(
    *,
    init_owner: Tensor,          # [P] long
    init_ships: Tensor,          # [P] float (current garrison)
    prod: Tensor,                # [P] float
    alive_by_step: Tensor,       # [H+1, P] bool
    background_arrivals: Tensor,  # [P, H, A] float (do-nothing arrivals)
    src: Tensor,                 # [C, L] long  source slot (-1 = none)
    tgt: Tensor,                 # [C, L] long  target slot
    ships: Tensor,               # [C, L] float
    eta: Tensor,                 # [C, L] float
    owner: Tensor,               # [C, L] long  arriving owner
    valid: Tensor,               # [C, L] bool
) -> tuple[Tensor, Tensor]:
    """Per-candidate post-combat owner/garrison trajectories ``[C, P, H+1]``.

    Applies each candidate's launches (source debit + target credit at the eta
    bucket) on top of the shared background arrivals, then walks the SAME engine
    recurrence the producer trusts. Dense across candidates (the shortlist is
    small; the freed opponent-mirror budget covers it).
    """
    P = int(init_owner.shape[0])
    H = int(background_arrivals.shape[1])
    A = int(background_arrivals.shape[2])
    C = int(src.shape[0])
    device = init_ships.device
    fdtype = init_ships.dtype

    h_idx = torch.ceil(eta.to(fdtype)).to(torch.long) - 1                 # [C, L]
    valid_t = (valid & (ships > 0) & (tgt >= 0) & (tgt < P)
               & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H))
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))

    # Source debit -> per-candidate init ships.
    init_ships_c = init_ships.view(1, P).expand(C, P).clone()
    debit = torch.zeros(C, P, dtype=fdtype, device=device)
    debit.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
    init_ships_c = init_ships_c - debit

    # Per-candidate arrivals = background + candidate credits.
    arrivals_c = background_arrivals.view(1, P, H, A).expand(C, P, H, A).clone()
    c_idx = torch.arange(C, device=device).view(C, 1).expand_as(tgt)
    m = valid_t
    if bool(m.any()):
        arrivals_c.index_put_(
            (c_idx[m], tgt_safe[m], h_idx[m], owner[m]),
            ships[m], accumulate=True,
        )

    init_owner_c = init_owner.view(1, P).expand(C, P).contiguous()
    prod_c = prod.view(1, P).expand(C, P).contiguous()
    alive_c = alive_by_step.permute(1, 0).view(1, P, H + 1).expand(C, P, H + 1).contiguous()

    owner_out, ships_out, _, _ = _run_exact_recurrence(
        init_owner=init_owner_c, init_ships=init_ships_c, prod=prod_c,
        alive=alive_c, arrivals=arrivals_c,
    )
    return owner_out, ships_out


def reachable_enemy_mass(
    *,
    cross_dist: Tensor,   # [K+1, P, P]  cross_dist[k, s, t] = dist(s@0, t@k)
    ships: Tensor,        # [P] float (current garrison)
    is_enemy: Tensor,     # [P] bool
    H: int,
) -> Tensor:
    """Enemy mass physically routable to each planet by step k -> ``[P, H+1]``.

    Enemy planet q can reach target p arriving at step k by launching now if
    ``dist(q@0, p@k) <= k · speed(q)``. Once reachable it stays reachable, so the
    result is cumulative (non-decreasing) in k. The attacker reservoir is the
    enemy's CURRENT ships (a conservative mean-field reservoir; Phase B/C can
    grow it with production)."""
    K = int(cross_dist.shape[0]) - 1
    P = int(ships.shape[0])
    device = ships.device
    fdtype = ships.dtype
    Hc = min(H, K)

    speed = fleet_speed(ships).to(fdtype)                       # [P] per source
    enemy_mass = torch.where(is_enemy, ships, torch.zeros_like(ships))  # [P]

    out = torch.zeros(P, H + 1, dtype=fdtype, device=device)
    # k = 0: nothing can have arrived yet.
    reachable_any = torch.zeros(P, P, dtype=torch.bool, device=device)  # [src, tgt]
    for k in range(1, Hc + 1):
        d_k = cross_dist[k]                                    # [src, tgt] dist(s@0, t@k)
        reach_k = d_k <= (float(k) * speed.view(P, 1))         # [src, tgt]
        reachable_any = reachable_any | reach_k
        # mass at tgt = sum over enemy sources that can reach tgt by now
        mass_t = (reachable_any.to(fdtype) * enemy_mass.view(P, 1)).sum(dim=0)  # [tgt]
        out[:, k] = mass_t
    for k in range(Hc + 1, H + 1):
        out[:, k] = out[:, Hc]  # horizon beyond the distance cache: hold last
    return out


def flip_prob(atk: Tensor, deff: Tensor, *, steepness: float, eps: float = 1.0
              ) -> Tensor:
    """Probability the opponent flips a planet I hold, from local force balance.
    Steep near parity; -> 1 when out-massed, -> 0 when dominant. In [0, 1)."""
    bal = (atk - deff) / (atk + deff + eps)
    return torch.sigmoid(steepness * bal)


def hazard_ownership_value(
    *,
    owner: Tensor,         # [C, P, H+1] long  (post-combat deterministic owner)
    ships: Tensor,         # [C, P, H+1] float (my projected garrison)
    prod: Tensor,          # [P] float
    atk_reach: Tensor,     # [P, H+1] float (enemy routable mass by step k)
    me: int,
    steepness: float = 5.0,
    discount: float = 1.0,
) -> Tensor:
    """Expected production-weighted ownership margin over the horizon -> ``[C]``.

    `P_mine(p,k) = [owner==me] · (1 - flip(atk_reach, garrison))`; the leaked
    `flip` mass and any planet the deterministic owner is an opponent both go to
    `P_opp`. Value = Σ_k discount^k Σ_p prod_p (P_mine − P_opp)."""
    C, P, H1 = owner.shape
    device = ships.device
    fdtype = ships.dtype

    atk = atk_reach.view(1, P, H1).to(fdtype)
    garrison = ships.clamp(min=0.0)
    leak = flip_prob(atk, garrison, steepness=steepness)        # [C, P, H+1]

    is_mine = (owner == me)
    is_opp = (owner >= 0) & (~is_mine)
    p_mine = is_mine.to(fdtype) * (1.0 - leak)
    # opponent ownership mass: planets they already hold + the leaked share of
    # mine. (Neutral planets contribute to neither.)
    p_opp = is_opp.to(fdtype) + is_mine.to(fdtype) * leak

    margin = (p_mine - p_opp) * prod.view(1, P, 1)              # [C, P, H+1]
    # discount^k over steps 1..H (step 0 is the present, weight 1).
    k = torch.arange(H1, device=device, dtype=fdtype)
    disc = torch.pow(torch.tensor(float(discount), dtype=fdtype, device=device), k)
    return (margin.sum(dim=1) * disc.view(1, H1)).sum(dim=1)    # [C]


def score_candidates_native(
    *,
    init_owner: Tensor, init_ships: Tensor, prod: Tensor,
    alive_by_step: Tensor, background_arrivals: Tensor,
    src: Tensor, tgt: Tensor, ships: Tensor, eta: Tensor,
    owner: Tensor, valid: Tensor,
    cross_dist: Tensor, cur_ships: Tensor, is_enemy: Tensor,
    me: int, steepness: float = 5.0, discount: float = 1.0,
) -> Tensor:
    """End-to-end native value per candidate ``[C]`` (the Phase-A scorer)."""
    H = int(background_arrivals.shape[1])
    owner_traj, ships_traj = build_candidate_trajectories(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive_by_step, background_arrivals=background_arrivals,
        src=src, tgt=tgt, ships=ships, eta=eta, owner=owner, valid=valid,
    )
    atk_reach = reachable_enemy_mass(
        cross_dist=cross_dist, ships=cur_ships, is_enemy=is_enemy, H=H,
    )
    return hazard_ownership_value(
        owner=owner_traj, ships=ships_traj, prod=prod, atk_reach=atk_reach,
        me=me, steepness=steepness, discount=discount,
    )
