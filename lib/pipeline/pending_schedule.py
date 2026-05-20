"""Persistent schedule state for `commit_persistent`.

Module-level container so the kaggle harness's per-call agent invocation
can read/write between turns. Keyed by `(my_id, game_id)` to keep
parallel games independent.

Game-id is the env's `episode_seed` if available, else a synthesized
fallback. For test isolation, call `clear()` between independent test
runs (the pipeline composer doesn't share a game_id key between fresh
env.run invocations because seeds differ; but tests on the same seed
benefit from explicit clear).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScheduledFire:
    """A wait_N>0 commitment held until its fire_step arrives."""
    src_id: int
    tgt_id: int
    ships: int
    angle: float
    fire_step: int   # absolute env step; decant when ctx.step_now == fire_step
    committed_at_step: int   # for diagnostics
    wait_N_original: int     # the wait_N from the column at commit time


# Per-(my_id, game_id) state. Lives at module level; kaggle worker shares.
_PENDING: dict[tuple, list[ScheduledFire]] = {}


def _key(my_id: int, game_id: int) -> tuple:
    return (int(my_id), int(game_id))


def clear() -> None:
    """Reset all pending state. Call between independent test runs."""
    _PENDING.clear()


def get_pending(my_id: int, game_id: int) -> list[ScheduledFire]:
    """Return (a copy of) the pending fires for this (my_id, game_id)."""
    return list(_PENDING.get(_key(my_id, game_id), []))


def set_pending(my_id: int, game_id: int, fires: list[ScheduledFire]) -> None:
    """Overwrite the pending fires for this (my_id, game_id)."""
    _PENDING[_key(my_id, game_id)] = list(fires)


def commit(my_id: int, game_id: int, new_fires: list[ScheduledFire]) -> None:
    """Append new wait_N>0 commitments to the pending list."""
    key = _key(my_id, game_id)
    existing = _PENDING.get(key, [])
    _PENDING[key] = existing + list(new_fires)


def decant_due(my_id: int, game_id: int, step_now: int) -> list[ScheduledFire]:
    """Pop pending fires whose `fire_step == step_now`.

    Removes them from the pending list and returns them. Caller is
    responsible for emitting them as moves this turn.
    """
    key = _key(my_id, game_id)
    pending = _PENDING.get(key, [])
    due = [f for f in pending if int(f.fire_step) == int(step_now)]
    if due:
        _PENDING[key] = [f for f in pending if int(f.fire_step) != int(step_now)]
    return due


def prune_stale(my_id: int, game_id: int, world) -> int:
    """Drop pending fires whose source no longer exists or no longer has
    enough ships. Returns the number pruned.
    """
    key = _key(my_id, game_id)
    pending = _PENDING.get(key, [])
    if not pending:
        return 0
    kept: list[ScheduledFire] = []
    pruned = 0
    for fire in pending:
        src = world.planets_by_id.get(int(fire.src_id))
        if src is None:
            pruned += 1
            continue
        if int(src.owner) != int(my_id):
            pruned += 1
            continue
        # Ship-feasibility: skip aggressive feasibility check here. The
        # decant step double-checks at emit time against the live source.
        # Pruning over-eagerly here can drop fires that production will
        # have replenished by the actual fire_step.
        kept.append(fire)
    if pruned > 0:
        _PENDING[key] = kept
    return pruned


def prune_past(my_id: int, game_id: int, step_now: int) -> int:
    """Drop pending fires whose fire_step is in the past (somehow missed).

    Returns the number pruned. This is a defensive cleanup; under normal
    operation decant_due processes fires at their fire_step.
    """
    key = _key(my_id, game_id)
    pending = _PENDING.get(key, [])
    if not pending:
        return 0
    kept = [f for f in pending if int(f.fire_step) >= int(step_now)]
    pruned = len(pending) - len(kept)
    if pruned > 0:
        _PENDING[key] = kept
    return pruned


def stats(my_id: int, game_id: int) -> dict:
    """Return diagnostics for the current pending state."""
    pending = _PENDING.get(_key(my_id, game_id), [])
    if not pending:
        return {"n_pending": 0}
    return {
        "n_pending": len(pending),
        "min_fire_step": min(int(f.fire_step) for f in pending),
        "max_fire_step": max(int(f.fire_step) for f in pending),
        "src_ids": sorted({int(f.src_id) for f in pending}),
    }
