"""Opponent multi-launch projection for the producer_plus scorer.

`predict_opp_multi_launch_lite` returns a `LaunchSet [L_opp]` describing
the opponent's projected launches over the next ``horizon`` ticks,
slot-indexed and ready to inject as background launches in
``score_candidates``. Native rewrite of
``lib/joint_solver/opp_projection.predict_opp_multi_launch`` using
orbit_lite primitives — no dependency on ``lib/intent.World``,
``lib/trajectory``, ``lib/world_model``, etc.

The algorithm mirrors the original ROI-greedy:

  For each opp seat, for each opp-owned source:
    - simulate production accumulation tick-by-tick over the horizon
    - if budget >= min_ships, pick the highest-ROI not-yet-taken target
    - validate the shot via `intercept_angle`'s viable mask
    - check the capture-floor at arrival + 1 (defender-grew-by-one tick)
    - record the launch, debit the budget, cap at max_per_source launches

The returned LaunchSet has owner per-launch (opp_id), with `valid=False`
slots padded to a fixed `MAX_L_OPP` so the scorer-level concat shape is
deterministic per turn. The scorer reads `valid` to gate inclusion in
the flow-diff sim; padded slots contribute nothing.

Wallclock budget: <30 ms per turn (8 opp sources × 8 ticks scalar loop +
one batched intercept_angle call). Compare to the original lib version
(~30 ms with heavier per-launch ray-cast trajectory validation).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .garrison_launch import LaunchSet
from .intercept_aim import intercept_angle
from .movement import PlanetMovement, PlanetGarrisonStatus
from .obs import ParsedObs
from .planner_core import safe_drain, capture_floor


# Defaults match `lib/joint_solver/opp_projection.py` constants.
DEFAULT_HORIZON = 8
DEFAULT_MAX_PER_SOURCE = 3
DEFAULT_MIN_SHIPS = 5
DEFAULT_SHIP_FRACTION = 0.7
DEFAULT_MAX_OPP_SOURCES = 12
DEFAULT_MAX_OPP_TARGETS = 12
MAX_L_OPP = 24  # 8 opp sources × 3 launches; bounds the scorer L-axis cost


def _empty_launch_set(*, opp_id: int, dtype: torch.dtype, device: torch.device) -> LaunchSet:
    """Return a LaunchSet with L=0 (no projected launches)."""
    return LaunchSet(
        source_slots=torch.zeros(0, dtype=torch.long, device=device),
        target_slots=torch.zeros(0, dtype=torch.long, device=device),
        ships=torch.zeros(0, dtype=dtype, device=device),
        eta=torch.ones(0, dtype=dtype, device=device),
        owner=torch.full((0,), int(opp_id), dtype=torch.long, device=device),
        valid=torch.zeros(0, dtype=torch.bool, device=device),
    )


def _pack_records_to_launch_set(
    records: list[tuple[int, int, float, float, int]],
    *,
    pad_to: int,
    default_opp_id: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LaunchSet:
    """Pack a list of (src_slot, tgt_slot, ships, eta, opp_id) records.

    Pads with valid=False slots up to ``pad_to`` for a deterministic L axis.
    """
    L = max(int(pad_to), 0)
    src = torch.zeros(L, dtype=torch.long, device=device)
    tgt = torch.zeros(L, dtype=torch.long, device=device)
    ships = torch.zeros(L, dtype=dtype, device=device)
    eta = torch.ones(L, dtype=dtype, device=device)
    owner = torch.full((L,), int(default_opp_id), dtype=torch.long, device=device)
    valid = torch.zeros(L, dtype=torch.bool, device=device)
    n = min(len(records), L)
    for i in range(n):
        s, t, sh, et, op = records[i]
        src[i] = int(s)
        tgt[i] = int(t)
        ships[i] = float(sh)
        eta[i] = float(et)
        owner[i] = int(op)
        valid[i] = True
    return LaunchSet(source_slots=src, target_slots=tgt, ships=ships, eta=eta,
                     owner=owner, valid=valid)


def predict_opp_multi_launch_lite(
    *,
    obs: ParsedObs,
    movement: PlanetMovement,
    garrison_status: PlanetGarrisonStatus,
    cache,                     # DistanceCache; only `cross_dist` used
    opp_ids: list[int],
    horizon: int = DEFAULT_HORIZON,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
    min_ships: int = DEFAULT_MIN_SHIPS,
    ship_fraction: float = DEFAULT_SHIP_FRACTION,
    max_opp_sources: int = DEFAULT_MAX_OPP_SOURCES,
    max_opp_targets: int = DEFAULT_MAX_OPP_TARGETS,
    pad_to: int = MAX_L_OPP,
) -> LaunchSet:
    """Project the opponents' next ``horizon`` ticks of launches.

    Returns a `LaunchSet [pad_to]` with the first N slots filled by
    projected opp launches and the remaining slots padded with
    ``valid=False`` (so the scorer skips them).
    """
    device = obs.device
    dtype = obs.ships.dtype
    P = int(obs.P)
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    horizon_eff = max(1, min(int(horizon), H))

    if P == 0 or not opp_ids:
        return _pack_records_to_launch_set(
            [], pad_to=pad_to,
            default_opp_id=opp_ids[0] if opp_ids else 0,
            dtype=dtype, device=device,
        )

    # cross_dist[0, s, t] = current centre distance from s to t.
    cross0 = cache.cross_dist[0].to(dtype)                                  # [P, P]
    prod = obs.prod.to(dtype)
    ships_now = obs.ships.to(dtype).clone()

    records: list[tuple[int, int, float, float, int]] = []
    # `taken_targets` ensures one opp doesn't project two launches at the same
    # target (the original algorithm uses this to avoid double-counting opp
    # captures within a single turn's projection).
    taken_targets: set[int] = set()

    for opp_id in opp_ids:
        opp_id = int(opp_id)
        # Sources: opp-owned alive planets with >= min_ships.
        opp_src_mask = obs.alive & (obs.owner_abs == opp_id) & (obs.ships >= float(min_ships))
        if not bool(opp_src_mask.any()):
            continue
        opp_src_slots_all = torch.nonzero(opp_src_mask, as_tuple=False).flatten()        # [N_src]
        # Cap to top-K by current ships (matches `_candidate_indices` style).
        if int(opp_src_slots_all.numel()) > int(max_opp_sources):
            ships_at_src = obs.ships[opp_src_slots_all]
            top_k = torch.topk(ships_at_src, k=int(max_opp_sources), largest=True).indices
            opp_src_slots = opp_src_slots_all[top_k]
        else:
            opp_src_slots = opp_src_slots_all
        S_opp = int(opp_src_slots.numel())
        if S_opp == 0:
            continue

        # Targets: NOT opp-owned, alive planets. Cap by top-K by ROI proxy.
        opp_tgt_mask = obs.alive & (obs.owner_abs != opp_id)
        if not bool(opp_tgt_mask.any()):
            continue
        opp_tgt_slots_all = torch.nonzero(opp_tgt_mask, as_tuple=False).flatten()        # [N_tgt]
        # ROI proxy per target: prod[t] / (min over opp sources of dist + 1).
        prod_t_all = prod[opp_tgt_slots_all]                                              # [N_tgt]
        dist_st_all = cross0[opp_src_slots][:, opp_tgt_slots_all]                         # [S_opp, N_tgt]
        roi_per_target = prod_t_all / (dist_st_all.min(dim=0).values + 1.0)               # [N_tgt]
        if int(opp_tgt_slots_all.numel()) > int(max_opp_targets):
            top_t = torch.topk(roi_per_target, k=int(max_opp_targets), largest=True).indices
            opp_tgt_slots = opp_tgt_slots_all[top_t]
        else:
            opp_tgt_slots = opp_tgt_slots_all
        T_opp = int(opp_tgt_slots.numel())
        if T_opp == 0:
            continue
        # Recompute distance matrix on the CAPPED target set.
        dist_st = cross0[opp_src_slots][:, opp_tgt_slots]                                # [S_opp, T_opp]

        # Per-source safe_drain (max safely-shippable from each opp source over H).
        src_ships = obs.ships[opp_src_slots].to(dtype)                                   # [S_opp]
        H_eff = torch.full((), float(H), dtype=dtype, device=device)
        drain = safe_drain(
            garrison_status, source_idx=opp_src_slots,
            source_ships=src_ships, H_eff=H_eff, player_id=opp_id,
        )                                                                                # [S_opp]

        # Batched intercept aim at sizes_hi = drain.floor (one size per pair).
        sizes = drain.view(S_opp, 1).expand(S_opp, T_opp).floor().clamp(min=1.0)         # [S_opp, T_opp]
        aim = intercept_angle(
            movement,
            opp_src_slots.view(S_opp, 1),
            opp_tgt_slots.view(1, T_opp),
            sizes,
        )
        eta_pair = aim["eta"]                                                            # [S_opp, T_opp]
        viable_pair = aim["viable"] & (eta_pair <= float(horizon_eff))                   # [S_opp, T_opp]

        # Capture-floor at arrival tick for the opp's POV.
        K_floor = int(horizon_eff)
        floor_opp = capture_floor(
            garrison_status, target_idx=opp_tgt_slots, k_max=K_floor,
            capture_overhead=1.0, player_id=opp_id,
        )                                                                                # [T_opp, K_floor]
        # eta-bucket index per pair (clamped to [0, K_floor-1]).
        k_arr = (
            eta_pair.clamp(min=1.0, max=float(K_floor)).ceil().long() - 1
        ).clamp(0, K_floor - 1)                                                          # [S_opp, T_opp]
        # Look up floor[t, k_arr+1] for the "defender grew by one tick" check;
        # clamp to last index when at horizon edge.
        k_arr_plus = (k_arr + 1).clamp(0, K_floor - 1)
        floor_at_arr_plus = floor_opp.unsqueeze(0).expand(S_opp, T_opp, K_floor).gather(
            -1, k_arr_plus.unsqueeze(-1)).squeeze(-1)                                    # [S_opp, T_opp]

        # ROI per (s, t) for greedy target ranking (same proxy as original).
        prod_t_b = prod[opp_tgt_slots].view(1, T_opp).expand(S_opp, T_opp)
        roi_st = prod_t_b / (dist_st + 1.0)                                              # [S_opp, T_opp]

        # Per-source scalar greedy loop with budget tracking.
        # Pull tensors to CPU once for the small inner loops.
        eta_cpu = eta_pair.cpu().tolist()
        viable_cpu = viable_pair.cpu().tolist()
        floor_cpu = floor_at_arr_plus.cpu().tolist()
        roi_cpu = roi_st.cpu().tolist()
        drain_cpu = drain.cpu().tolist()
        tgt_slot_cpu = opp_tgt_slots.cpu().tolist()
        src_slot_cpu = opp_src_slots.cpu().tolist()
        prod_cpu = prod.cpu().tolist()
        ships_cpu = ships_now.cpu().tolist()

        for s_i in range(S_opp):
            src_slot = int(src_slot_cpu[s_i])
            budget = float(ships_cpu[src_slot])
            launches_from = 0
            for _tick in range(horizon_eff):
                budget += float(prod_cpu[src_slot])
                if launches_from >= int(max_per_source) or budget < float(min_ships):
                    continue
                # Pick best viable, not-taken, capture-clearable target.
                best_t = -1
                best_roi = -1.0
                for t_i in range(T_opp):
                    if not bool(viable_cpu[s_i][t_i]):
                        continue
                    t_slot = int(tgt_slot_cpu[t_i])
                    if t_slot in taken_targets:
                        continue
                    # Send proposal: max(min, fraction × budget), capped by drain + budget.
                    proposal = max(float(min_ships), float(ship_fraction) * budget)
                    proposal = min(proposal, float(drain_cpu[s_i]), budget)
                    if proposal < float(min_ships):
                        continue
                    # Defender check at arrival + 1: must clear `floor_at_arr_plus`.
                    if proposal < float(floor_cpu[s_i][t_i]):
                        continue
                    if float(roi_cpu[s_i][t_i]) > best_roi:
                        best_roi = float(roi_cpu[s_i][t_i])
                        best_t = t_i
                if best_t < 0:
                    continue
                t_slot = int(tgt_slot_cpu[best_t])
                send = max(float(min_ships), float(ship_fraction) * budget)
                send = min(send, float(drain_cpu[s_i]), budget)
                eta_val = max(1.0, float(eta_cpu[s_i][best_t]))
                records.append((src_slot, t_slot, send, eta_val, opp_id))
                budget -= send
                taken_targets.add(t_slot)
                launches_from += 1
                if len(records) >= int(pad_to):
                    break
            if len(records) >= int(pad_to):
                break
        if len(records) >= int(pad_to):
            break

    return _pack_records_to_launch_set(
        records,
        pad_to=pad_to,
        default_opp_id=opp_ids[0] if opp_ids else 0,
        dtype=dtype, device=device,
    )
