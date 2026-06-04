"""Opponent-projection registry for ``producer_plus``.

Producer's `sparse_launch_flow_delta` scorer treats opponent garrisons as
frozen for the full 18-tick scoring window. Champion's
``lib.opp_model.lite_greedy_policy`` projects opponent launches; this
module exposes a swappable interface so we can layer the lite-greedy
projection (and, later, the condensed-Producer projector being built on
a sister branch) into Producer's per-planet/per-tick/per-player arrival
ledger before scoring runs.

See ``state/MIGRATION_PLAN.md`` Step 3.

Default registration is ``"none"`` — bit-identical to today's Producer.
Picked at runtime via ``PRODUCER_PLUS_OPP_PROJECTOR`` env var.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


ArrivalTuple = tuple[int, int, int, int]  # (target_planet_id, eta_abs, owner, ships)
Projector = Callable[..., list[ArrivalTuple]]


def _debug_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_DEBUG", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _debug_log(where: str, exc: BaseException) -> None:
    """Single-line stderr log gated on ``PRODUCER_PLUS_DEBUG``.

    Silent by default so production runs match today's behaviour. Turning
    the flag on surfaces genuine bugs that the broad ``except Exception``
    would otherwise swallow as "no projection".
    """
    if _debug_enabled():
        sys.stderr.write(
            f"[producer_plus] {where}: {type(exc).__name__}: {exc}\n"
        )


def _none_projector(
    obs: Any, *, my_id: int, num_seats: int, horizon: int
) -> list[ArrivalTuple]:
    """Bit-identical default: project no opponent launches."""
    return []


def _lite_greedy_projector(
    obs: Any, *, my_id: int, num_seats: int, horizon: int
) -> list[ArrivalTuple]:
    """ROI-greedy multi-tick projection — mirror of champion lite-greedy.

    Builds a champion-style ``World`` from raw obs and delegates to
    ``lib.joint_solver.opp_projection.predict_opp_multi_launch``. Cost is
    ~1-2 ms per turn (one-shot, NOT per candidate). Returns an empty
    list on any failure so a misbehaving projector cannot brick the
    agent; set ``PRODUCER_PLUS_DEBUG=1`` to surface the swallowed
    exception on stderr.
    """
    try:
        from lib.intent import World
        from lib.joint_solver.opp_projection import predict_opp_multi_launch

        world = World.from_obs(obs)
        return predict_opp_multi_launch(
            world, int(my_id), int(num_seats), horizon=int(horizon)
        )
    except Exception as exc:
        _debug_log("lite_greedy_projector", exc)
        return []


PROJECTORS: dict[str, Projector] = {
    "none": _none_projector,
    "lite_greedy": _lite_greedy_projector,
}


def get_projector(name: str | None) -> Projector:
    """Lookup a projector by name; unknown names fall back to ``_none_projector``."""
    if name is None:
        return _none_projector
    return PROJECTORS.get(str(name).strip().lower(), _none_projector)


def arrivals_tuples_to_buckets_delta(
    tuples: Sequence[ArrivalTuple],
    planet_ids: Tensor,
    *,
    A: int,
    H: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Translate champion-format arrivals into Producer's ``fleet_buckets`` delta.

    Producer indexes ``fleet_buckets[p_slot, k, owner]`` where ``k`` is the
    bucket index for arrivals landing at tick ``k+1`` (status pads a
    zero step-0 frame in ``arrivals_by_owner``). Champion's tuples carry
    absolute eta starting at 1 (= next tick), so ``bucket_idx = eta - 1``.

    Skips silently:
      * tuples with ``eta < 1`` (already-landed; impossible from projector but defensive),
      * tuples with ``eta > H`` (outside scoring window),
      * tuples whose ``target_planet_id`` is not present in the current obs,
      * tuples with ``owner < 0`` or ``owner >= A``,
      * tuples with non-positive ``ships``.

    Duplicate ``(slot, bucket, owner)`` triples accumulate (sum).
    """
    P = int(planet_ids.shape[0])
    delta = torch.zeros((P, max(H, 0), int(A)), dtype=dtype, device=device)
    if H <= 0 or not tuples:
        return delta

    # Map planet_id -> slot. planet_ids is a CPU-cheap tensor; resolve via dict
    # for unknown-id rejection.
    ids_cpu = planet_ids.detach().to("cpu").tolist()
    id_to_slot = {int(pid): slot for slot, pid in enumerate(ids_cpu) if int(pid) >= 0}

    for tgt_id, eta, owner, ships in tuples:
        eta_i = int(eta)
        if eta_i < 1 or eta_i > H:
            continue
        owner_i = int(owner)
        if owner_i < 0 or owner_i >= int(A):
            continue
        ships_f = float(ships)
        if ships_f <= 0.0:
            continue
        slot = id_to_slot.get(int(tgt_id))
        if slot is None:
            continue
        delta[slot, eta_i - 1, owner_i] += ships_f
    return delta


def affected_slots(
    tuples: Sequence[ArrivalTuple],
    planet_ids: Tensor,
    *,
    H: int,
    A: int,
) -> Tensor:
    """Return the unique planet slot indices that ``tuples`` would write to.

    Applies the same skip rules as ``arrivals_tuples_to_buckets_delta``
    (out-of-window eta, unknown planet, bad owner, non-positive ships)
    so the returned slots exactly match the planets whose
    ``fleet_buckets`` rows are mutated by the delta. Used to narrow the
    garrison-cache invalidation footprint.

    Returns an empty ``long`` tensor on this projector module's CPU
    when nothing is affected. Caller is responsible for moving the
    tensor onto the movement device.
    """
    if H <= 0 or not tuples:
        return torch.empty((0,), dtype=torch.long)
    ids_cpu = planet_ids.detach().to("cpu").tolist()
    id_to_slot = {int(pid): slot for slot, pid in enumerate(ids_cpu) if int(pid) >= 0}
    slots: set[int] = set()
    A_int = int(A)
    H_int = int(H)
    for tgt_id, eta, owner, ships in tuples:
        eta_i = int(eta)
        if eta_i < 1 or eta_i > H_int:
            continue
        owner_i = int(owner)
        if owner_i < 0 or owner_i >= A_int:
            continue
        if float(ships) <= 0.0:
            continue
        slot = id_to_slot.get(int(tgt_id))
        if slot is None:
            continue
        slots.add(slot)
    if not slots:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(sorted(slots), dtype=torch.long)
