"""Producer-mirror opponent projection for the producer_plus scorer.

`predict_opp_launches_via_mirror` runs Producer's own planner once per
opponent seat (with ``background=None`` to avoid recursion) and returns
the launches that planner would fire this turn as a padded `LaunchSet`,
ready to inject as background launches in our scorer.

The earlier ROI-greedy projector (ported from
``lib/joint_solver/opp_projection``) modeled the wrong agent: ROI-greedy
target selection, ``0.7 * budget`` send size, and up to 3 launches per
source over 8 ticks. The real opponent distribution is dominated by the
public Producer agent itself, whose target ranking, send sizes, and
launch counts differ materially. Using Producer's own planner as the
opponent model tracks the real opponent automatically.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .garrison_launch import LaunchSet
from .movement import PlanetMovement, PlanetGarrisonStatus
from .obs import parse_obs


# Padded L axis for the projected opp LaunchSet. Producer typically fires
# 0-3 launches per turn per seat; 24 slots is generous headroom.
MAX_L_OPP = 24


def _pack_records_to_launch_set(
    records: list[tuple[int, int, float, float, int]],
    *,
    pad_to: int,
    default_opp_id: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LaunchSet:
    """Pack ``(src_slot, tgt_slot, ships, eta, opp_id)`` records into a
    padded `LaunchSet[pad_to]`. Unused slots have ``valid=False``."""
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
    return LaunchSet(
        source_slots=src, target_slots=tgt, ships=ships,
        eta=eta, owner=owner, valid=valid,
    )


def predict_opp_launches_via_mirror(
    *,
    plan_fn,
    obs_tensors: dict,
    movement: PlanetMovement,
    cache,
    garrison_status: PlanetGarrisonStatus,
    prod: Tensor,
    alive_by_step: Tensor,
    opp_ids: list[int],
    config,
    player_count: int,
    K_eta_override: int | None = None,
    pad_to: int = MAX_L_OPP,
    K: int = 1,
    H: int | None = None,
) -> LaunchSet:
    """For each opponent seat, run ``plan_fn`` (Producer's planner) with
    the seat swapped to their POV.

    With ``K=1`` (default), passes ``background=None`` (one-step best
    response, opp assumes we do nothing this turn). Byte-identical to the
    original single-pass behaviour.

    With ``K>1``, runs K successive projection rounds. Round k passes
    ``background=cumulative_bg`` (the union of all previously-projected
    opp launches, with absolute-frame etas) so opp's planner accounts for
    its own prior commitments via its internal forward-simulator. Each
    round's launches have etas shifted by ``+k`` turns before being
    appended to the cumulative records — round k is conceptually opp's
    decision at game-tick ``k``, so an eta of ``e`` from round k lands at
    absolute tick ``k + e``. Launches whose shifted eta exceeds ``H``
    (the scorer horizon) are dropped — invalid arrivals are worse than
    no arrivals.

    ``plan_fn`` is passed as a callback (rather than imported) so this
    module has no cross-package dependency -- works the same in the source
    tree and in the bundled submission.
    """
    device = obs_tensors["planets"].device
    # Ships dtype matches obs.ships used inside plan_lite_waves; obs.ships is
    # derived from obs_tensors["planets"] in parse_obs.
    sample = parse_obs(obs_tensors, player_id=int(opp_ids[0]) if opp_ids else 0)
    dtype = sample.ships.dtype

    if not opp_ids:
        return _pack_records_to_launch_set(
            [], pad_to=pad_to, default_opp_id=0,
            dtype=dtype, device=device,
        )

    K_clamped = max(1, int(K))

    if K_clamped == 1:
        # Single-pass path: bit-identical to the original implementation.
        records: list[tuple[int, int, float, float, int]] = []
        for opp_id in opp_ids:
            opp_id = int(opp_id)
            obs_opp = parse_obs(obs_tensors, player_id=opp_id)
            opp_entries = plan_fn(
                movement=movement,
                obs=obs_opp,
                obs_tensors=obs_tensors,
                cache=cache,
                garrison_status=garrison_status,
                prod=prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=int(player_count),
                K_eta_override=K_eta_override,
                background=None,
            )
            # Walk the flat [L] entry table; emit one record per valid slot.
            src_cpu = opp_entries.source_slots.cpu().tolist()
            tgt_cpu = opp_entries.target_slots.cpu().tolist()
            ships_cpu = opp_entries.ships.cpu().tolist()
            eta_cpu = opp_entries.eta.cpu().tolist()
            valid_cpu = opp_entries.valid.cpu().tolist()
            for i in range(len(src_cpu)):
                if not bool(valid_cpu[i]):
                    continue
                records.append((
                    int(src_cpu[i]),
                    int(tgt_cpu[i]),
                    float(ships_cpu[i]),
                    float(eta_cpu[i]),
                    opp_id,
                ))
                if len(records) >= int(pad_to):
                    break
            if len(records) >= int(pad_to):
                break

        return _pack_records_to_launch_set(
            records, pad_to=pad_to,
            default_opp_id=int(opp_ids[0]),
            dtype=dtype, device=device,
        )

    # K > 1: multi-round projection. Eta cap = scorer horizon H if
    # provided; otherwise K_eta_override; otherwise no cap. The garrison
    # status only has entries for ticks <= H, so launches with shifted
    # eta > H cannot be scored and would corrupt the planner if packed.
    if H is not None:
        eta_cap = float(H)
    elif K_eta_override is not None:
        eta_cap = float(K_eta_override)
    else:
        eta_cap = float("inf")

    # NB: the scorer's per-candidate broadcast (plan_lite_waves: concat of
    # background onto launches' L axis) costs O(C × L_total). Inflating the
    # background's pad slot count past ``pad_to`` blows up wallclock by ~2x
    # per doubling for *every* downstream scorer call -- the per-round opp
    # planner AND the main planner. Cap pad at the original ``pad_to``;
    # records beyond ``pad_to`` are silently dropped at pack time (matches
    # the K=1 path's len(records) >= pad_to early break). With K=2 in 2P
    # and 1 opp at ~3 launches/round, total records ~ 6 — well under cap.
    records: list[tuple[int, int, float, float, int]] = []
    cumulative_bg: LaunchSet | None = None
    final_pad = int(pad_to)
    for k in range(K_clamped):
        round_records: list[tuple[int, int, float, float, int]] = []
        for opp_id in opp_ids:
            opp_id = int(opp_id)
            obs_opp = parse_obs(obs_tensors, player_id=opp_id)
            opp_entries = plan_fn(
                movement=movement,
                obs=obs_opp,
                obs_tensors=obs_tensors,
                cache=cache,
                garrison_status=garrison_status,
                prod=prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=int(player_count),
                K_eta_override=K_eta_override,
                background=cumulative_bg,
            )
            src_cpu = opp_entries.source_slots.cpu().tolist()
            tgt_cpu = opp_entries.target_slots.cpu().tolist()
            ships_cpu = opp_entries.ships.cpu().tolist()
            eta_cpu = opp_entries.eta.cpu().tolist()
            valid_cpu = opp_entries.valid.cpu().tolist()
            for i in range(len(src_cpu)):
                if not bool(valid_cpu[i]):
                    continue
                shifted_eta = float(eta_cpu[i]) + float(k)
                if shifted_eta > eta_cap:
                    continue
                round_records.append((
                    int(src_cpu[i]),
                    int(tgt_cpu[i]),
                    float(ships_cpu[i]),
                    shifted_eta,
                    opp_id,
                ))
        records.extend(round_records)
        # Rebuild cumulative bg for the next round (skip on final round).
        if k + 1 < K_clamped:
            cumulative_bg = _pack_records_to_launch_set(
                records, pad_to=final_pad,
                default_opp_id=int(opp_ids[0]),
                dtype=dtype, device=device,
            )

    return _pack_records_to_launch_set(
        records, pad_to=final_pad,
        default_opp_id=int(opp_ids[0]),
        dtype=dtype, device=device,
    )
