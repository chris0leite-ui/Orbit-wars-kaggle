"""Long-term production value bonuses for the producer_plus scorer.

Two leaf-scorer terms that add per-candidate bonuses (in ship units)
reflecting production value past the scorer's H-tick horizon:

1. ``denial_bonus`` — opp-aware. Rewards captures of targets the
   opponent values (currently owns OR predicted to attack via
   opp_proj's background ``LaunchSet``). Encodes the intuition that
   blocking the opponent's biggest bet is itself a winning move.

2. ``opening_bonus`` — opp-agnostic. Rewards captures during the early
   game phase, when the H=18 scorer most under-values compounded
   production from a long-held planet. Linearly decays from full at
   step 0 to zero at the configured opening window.

3. ``frontier_bonus`` — reach/option-aware. Rewards capturing a planet
   for the *new options it unlocks*: the production-weighted set of
   neutral planets that become newly reachable (within a turn budget)
   once we own it and can launch from it. This is the "gateway" value
   the H-tick own-income scorer cannot see — a far corner is valued for
   the cluster it opens, not its own production alone. See
   ``audit/2026-06-15-frontier-gateway-value-spec.md``.

All three bonuses ADD to the candidate score (recapture_penalty
SUBTRACTS). All gated default-OFF; byte-identical when the gates are unset.

Math, per candidate ``c`` capturing target ``T`` with ``s_c`` ships at
arrival tick ``e_c``:

    future_h   = max(0, game_length_est - current_step - H)
    captures_c = (s_c >= capture_floor_TK[t, e_c-1]) & cand_valid
                 & ~cand_is_def & ~own_already_at_e_c

    # Denial:
    opp_values_T = (opp_owns_T & alive) | (sum_of_opp_proj_ships_at_T > 0)
    denial_c     = captures_c & opp_values_T
    denial_bonus = denial_c * prod[T] * future_h * weight

    # Opening:
    phase = max(0, 1 - current_step / opening_window)
    opening_bonus = captures_c * phase * prod[T] * future_h * weight

The shared ``_compute_captures()`` helper centralizes the capture-gate
logic so both bonuses (and recapture_penalty in a future refactor)
agree on what "we actually capture" means.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .distance_cache import DistanceCache
from .garrison_launch import LaunchSet
from .movement import PlanetGarrisonStatus


def _future_value_horizon(current_step: int, H: int, game_length_est: int) -> int:
    """Estimated turns of production beyond the scorer's H-tick window.

    The scorer already values production over ticks [1, H]. The bonuses
    here value the *additional* compound production from holding the
    planet for the rest of the game. Default ``game_length_est=200`` is
    a rough average orbit-wars game length; tune via env knob.
    """
    return max(0, int(game_length_est) - int(current_step) - int(H))


def _compute_captures(
    *,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    capture_floor_TK: Tensor,
    garrison_status: PlanetGarrisonStatus,
    player_id: int,
    P: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor] | None:
    """Return ``(captures_c [C] bool, tgt_abs [C] long)`` or ``None`` if
    the shortlist is empty (no candidates can be captures).

    A candidate "captures" iff:
    - it sends >= the per-target capture_floor at the arrival tick
    - it is marked valid
    - it is not a defensive (own-planet) reinforcement
    - the post-do-nothing trajectory does not already show us owning the
      target at the arrival tick (else it's a reinforcement, not a capture)
    """
    K_floor = int(capture_floor_TK.shape[-1])
    T_floor = int(capture_floor_TK.shape[0])
    if K_floor <= 0 or T_floor <= 0:
        return None
    e_send = cand_send[:, 0].to(dtype)
    e_eta = cand_eta[:, 0].to(dtype)
    e_idx = (torch.ceil(e_eta).long() - 1).clamp(0, K_floor - 1)
    tshort = cand_tgt_short.clamp(0, T_floor - 1).to(device)
    floor_c = capture_floor_TK[tshort, e_idx]

    tgt_abs = cand_tgt_slot.clamp(0, P - 1).to(device)
    owner_axis_H = int(garrison_status.owner.shape[-1])
    own_idx = torch.ceil(e_eta).long().clamp(0, max(owner_axis_H - 1, 0))
    own_at_arrival = (
        garrison_status.owner[tgt_abs, own_idx] == int(player_id)
    )

    captures_c = (
        (e_send >= floor_c)
        & cand_valid.to(device)
        & ~cand_is_def.to(device)
        & ~own_at_arrival
    )
    return captures_c, tgt_abs


def denial_bonus(
    *,
    obs,
    background: LaunchSet | None,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    capture_floor_TK: Tensor,
    prod: Tensor,
    garrison_status: PlanetGarrisonStatus,
    H: int,
    current_step: int,
    game_length_est: int,
    weight: float,
    player_id: int,
) -> Tensor:
    """Return ``[C]`` non-negative denial bonus in ship units.

    Triggers when (a) we actually capture target ``T`` and (b) opp values
    ``T`` (owns it OR opp_proj's background LaunchSet contains a launch
    targeting it). The bonus reflects the production we deny opp by
    getting there first, summed over the post-horizon game length.
    """
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    future_h = _future_value_horizon(current_step, H, game_length_est)
    if future_h <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    capture_info = _compute_captures(
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        capture_floor_TK=capture_floor_TK,
        garrison_status=garrison_status,
        player_id=player_id, P=P, device=device, dtype=dtype,
    )
    if capture_info is None:
        return torch.zeros(C, dtype=dtype, device=device)
    captures_c, tgt_abs = capture_info

    # opp_values_T[P]: opp currently owns it, OR opp_proj predicted at
    # least one launch targeting it. The "owns" arm covers attacks on
    # enemy planets (denying their production); the "intent" arm covers
    # races for neutrals opp planned to expand to.
    opp_owned_alive = (obs.is_enemy & obs.alive).to(device)
    opp_intent = torch.zeros(P, dtype=dtype, device=device)
    if background is not None and int(background.source_slots.shape[-1]) > 0:
        bg_valid = background.valid.to(device)
        bg_tgt = background.target_slots.clamp(0, P - 1).to(device)
        bg_ships = background.ships.to(device=device, dtype=dtype)
        ships_masked = torch.where(bg_valid, bg_ships, torch.zeros_like(bg_ships))
        opp_intent.scatter_add_(0, bg_tgt, ships_masked)
    opp_values_T = opp_owned_alive | (opp_intent > 0)               # [P] bool

    opp_values_c = opp_values_T[tgt_abs].to(dtype)                  # [C]
    prod_c = prod.to(device=device, dtype=dtype)[tgt_abs]           # [C]

    bonus = (
        captures_c.to(dtype)
        * opp_values_c
        * float(weight)
        * prod_c
        * float(future_h)
    )
    return bonus.clamp(min=0.0)


def opening_bonus(
    *,
    obs,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    capture_floor_TK: Tensor,
    prod: Tensor,
    garrison_status: PlanetGarrisonStatus,
    H: int,
    current_step: int,
    game_length_est: int,
    opening_window: int,
    weight: float,
    player_id: int,
) -> Tensor:
    """Return ``[C]`` non-negative opening-phase bonus in ship units.

    Linearly decays from a full ``weight × prod × future_horizon`` bonus
    at step 0 to zero at ``opening_window`` (default 30). Opp-agnostic:
    encodes the H-too-short defect at the opening, when long-held
    captures compound the most.
    """
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    if opening_window <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    phase_factor = max(0.0, 1.0 - float(current_step) / float(opening_window))
    if phase_factor <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    future_h = _future_value_horizon(current_step, H, game_length_est)
    if future_h <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    capture_info = _compute_captures(
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        capture_floor_TK=capture_floor_TK,
        garrison_status=garrison_status,
        player_id=player_id, P=P, device=device, dtype=dtype,
    )
    if capture_info is None:
        return torch.zeros(C, dtype=dtype, device=device)
    captures_c, tgt_abs = capture_info

    prod_c = prod.to(device=device, dtype=dtype)[tgt_abs]            # [C]
    bonus = (
        captures_c.to(dtype)
        * float(weight)
        * float(phase_factor)
        * prod_c
        * float(future_h)
    )
    return bonus.clamp(min=0.0)


def _reach_eta_matrix(
    cache: DistanceCache,
    *,
    reach_turns: int,
    nominal_speed: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Per-(source, target) best arrival tick under a turn budget. ``[P, P]``.

    ``eta[u, v]`` is the smallest ``k in [1, R]`` for which a fleet of nominal
    speed ``c`` launched from ``u@0`` can intercept ``v@k`` —
    ``cross_dist[k, u, v] <= c * k`` — with ``v`` alive at tick ``k``; ``+inf``
    where no such ``k`` exists. ``reach[u, v] = isfinite(eta)``.

    ``R`` is clamped to the distance cache's horizon ``cache.K``. Pure tensor
    arithmetic over the precomputed cross-time cache → no new geometry, no
    host/device sync, CPU/CUDA agree. Mirrors the planner's own ``surf/k <=
    speed`` reachability shape (``planner_core.reachable_mask``).
    """
    P = int(cache.P)
    R = max(1, min(int(reach_turns), int(cache.K)))
    # cross[k-1, u, v] = dist(u@0, v@k) for k in [1, R].
    cross = cache.cross_dist[1 : R + 1].to(device=device, dtype=dtype)   # [R, P, P]
    alive_at_k = cache.alive_by_step[1 : R + 1].to(device=device)        # [R, P]
    k_grid = torch.arange(1, R + 1, device=device, dtype=dtype).view(R, 1, 1)
    feasible = cross <= float(nominal_speed) * k_grid                    # [R, P_src, P_tgt]
    # Target must be alive at the arrival tick (drops departing comets etc.).
    feasible = feasible & alive_at_k.unsqueeze(1)                        # [R, 1, P_tgt] broadcast
    inf = torch.full_like(cross, float("inf"))
    eta_kuv = torch.where(feasible, k_grid.expand_as(cross), inf)        # [R, P, P]
    eta = eta_kuv.amin(dim=0)                                            # [P_src, P_tgt]
    reach = torch.isfinite(eta)
    return eta, reach


def frontier_bonus(
    *,
    obs,
    cache: DistanceCache,
    garrison_status: PlanetGarrisonStatus,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    capture_floor_TK: Tensor,
    prod: Tensor,
    H: int,
    current_step: int,
    game_length_est: int,
    weight: float,
    reach_turns: int,
    nominal_speed: float,
    contest_weight: float,
    include_enemy: bool,
    player_id: int,
    comet_mask: Tensor | None = None,
) -> Tensor:
    """Return ``[C]`` non-negative gateway/option bonus in ship units.

    For each candidate ``c`` that actually captures target ``a`` (its
    ``cand_tgt_slot``), credit the *new options* owning ``a`` unlocks: the
    production-weighted set of neutral planets reachable from ``a`` (as a launch
    base, within ``reach_turns``) that are NOT already reachable from our
    currently-owned set. Closer-unlocked targets count more (proximity
    discount); the whole thing is scaled by the post-horizon turns remaining so
    a gateway is worth most early and zero at game end.

    See ``audit/2026-06-15-frontier-gateway-value-spec.md`` for the full math.
    """
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    future_h = _future_value_horizon(current_step, H, game_length_est)
    if future_h <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    capture_info = _compute_captures(
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        capture_floor_TK=capture_floor_TK,
        garrison_status=garrison_status,
        player_id=player_id, P=P, device=device, dtype=dtype,
    )
    if capture_info is None:
        return torch.zeros(C, dtype=dtype, device=device)
    captures_c, tgt_abs = capture_info

    # ----- pairwise reach (one [P, P] eta/reach matrix for the turn) -----
    R = max(1, min(int(reach_turns), int(cache.K)))
    eta, reach = _reach_eta_matrix(
        cache, reach_turns=R, nominal_speed=nominal_speed,
        device=device, dtype=dtype,
    )                                                                    # [P_src, P_tgt]

    # Frontier targets V: alive neutrals (optionally enemies), non-comet.
    V = (obs.is_neutral & obs.alive).to(device)
    if include_enemy:
        V = V | (obs.is_enemy & obs.alive).to(device)
    if comet_mask is not None:
        V = V & ~comet_mask.to(device=device, dtype=torch.bool)

    # reach_from_A[v]: already reachable today from some owned, alive source.
    A = (obs.owned & obs.alive).to(device)                              # [P]
    reach_from_A = (reach & A.view(P, 1)).any(dim=0)                    # [P_tgt]
    newly = reach & (~reach_from_A).view(1, P)                          # [P_src=a, P_tgt=v]
    # A base is not its own frontier.
    newly = newly & ~torch.eye(P, dtype=torch.bool, device=device)

    # Proximity discount: 1 when adjacent (eta=1), → 0 at the budget edge.
    disc = (1.0 - eta / float(R + 1)).clamp(0.0, 1.0)
    disc = torch.where(reach, disc, torch.zeros_like(disc))             # finite where reach

    # Contest down-weight (default off): a newly-unlocked target an enemy base
    # reaches no later than `a` is contested → worth less (we may lose the race).
    contest_w = float(max(0.0, min(1.0, contest_weight)))
    if contest_w > 0.0:
        E = (obs.is_enemy & obs.alive).to(device)
        inf = torch.full_like(eta, float("inf"))
        eta_enemy = torch.where(E.view(P, 1), eta, inf).amin(dim=0)     # [P_tgt]
        contested = eta_enemy.view(1, P) <= eta                         # [a, v]
        contest_factor = torch.where(
            contested & newly,
            torch.full_like(eta, 1.0 - contest_w),
            torch.ones_like(eta),
        )
    else:
        contest_factor = torch.ones_like(eta)

    prod_row = prod.to(device=device, dtype=dtype).view(1, P)           # [1, P_tgt]
    contrib = (
        newly.to(dtype)
        * V.view(1, P).to(dtype)
        * prod_row
        * disc
        * contest_factor
    )                                                                   # [P_src=a, P_tgt=v]
    fv_per_base = contrib.sum(dim=1)                                    # [P] value per base

    fv_c = fv_per_base[tgt_abs]                                         # [C]
    bonus = (
        captures_c.to(dtype)
        * float(weight)
        * float(future_h)
        * fv_c
    )
    return bonus.clamp(min=0.0)
