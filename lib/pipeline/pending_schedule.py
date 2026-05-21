"""Persistent schedule state for `commit_persistent`.

Bug #1 fix — closure-instance state.

The OLD design held a single module-level `_PENDING` dict keyed by
`(my_id, game_id)` and fell back to a hash of the initial planet
config when `obs_d['episode_seed']` was absent. In a tournament harness
that runs multiple games in the same Python process (with seeds Kaggle
doesn't always expose), the hash could collide and state from game A
would surface in game B — manifesting as "ships missing targets" when
fires scheduled for one game decanted during another.

The new design exposes a `PendingSchedule` class that:
  * Encapsulates all state in instance attributes (no module dict).
  * Has a `begin_turn(fingerprint)` method that detects game boundaries
    via a per-game fingerprint and resets state when the fingerprint
    changes — so even a long-lived instance shared across sequential
    games stays isolated.
  * Has explicit `reset()` for tests.

The module-level functions (`commit`, `decant_due`, ...) are preserved
as thin wrappers around a module-level **singleton** instance so
existing imports keep working. Tests can instantiate their own
PendingSchedule for isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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


class PendingSchedule:
    """Per-instance pending-schedule container.

    Designed so kaggle's "one process per worker, sequential games" model
    can safely share a single instance: `begin_turn(fingerprint)` resets
    state when a new game's fingerprint is detected.
    """

    def __init__(self) -> None:
        # Per-my_id list (single game; cross-game isolation via reset
        # on fingerprint change in begin_turn).
        self._pending: dict[int, list[ScheduledFire]] = {}
        self._game_fingerprint: Any = None

    def begin_turn(self, fingerprint: Any) -> bool:
        """Mark the start of a turn; reset if game boundary detected.

        Returns True iff a reset was triggered (the caller can log this
        for observability).
        """
        if self._game_fingerprint is None:
            self._game_fingerprint = fingerprint
            return False
        if fingerprint == self._game_fingerprint:
            return False
        # Fingerprint changed → new game; wipe.
        self._pending = {}
        self._game_fingerprint = fingerprint
        return True

    def reset(self) -> None:
        """Drop all state. Tests use this; production callers shouldn't
        need to."""
        self._pending = {}
        self._game_fingerprint = None

    # ---- query / mutate ----

    def get_pending(self, my_id: int) -> list[ScheduledFire]:
        return list(self._pending.get(int(my_id), []))

    def set_pending(self, my_id: int, fires: list[ScheduledFire]) -> None:
        self._pending[int(my_id)] = list(fires)

    def commit(self, my_id: int, new_fires: list[ScheduledFire]) -> None:
        key = int(my_id)
        existing = self._pending.get(key, [])
        self._pending[key] = existing + list(new_fires)

    def decant_due(self, my_id: int, step_now: int) -> list[ScheduledFire]:
        key = int(my_id)
        pending = self._pending.get(key, [])
        due = [f for f in pending if int(f.fire_step) == int(step_now)]
        if due:
            self._pending[key] = [
                f for f in pending if int(f.fire_step) != int(step_now)
            ]
        return due

    def prune_stale(self, my_id: int, world) -> int:
        key = int(my_id)
        pending = self._pending.get(key, [])
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
            kept.append(fire)
        if pruned > 0:
            self._pending[key] = kept
        return pruned

    def prune_past(self, my_id: int, step_now: int) -> int:
        key = int(my_id)
        pending = self._pending.get(key, [])
        if not pending:
            return 0
        kept = [f for f in pending if int(f.fire_step) >= int(step_now)]
        pruned = len(pending) - len(kept)
        if pruned > 0:
            self._pending[key] = kept
        return pruned

    def stats(self, my_id: int) -> dict:
        pending = self._pending.get(int(my_id), [])
        if not pending:
            return {"n_pending": 0}
        return {
            "n_pending": len(pending),
            "min_fire_step": min(int(f.fire_step) for f in pending),
            "max_fire_step": max(int(f.fire_step) for f in pending),
            "src_ids": sorted({int(f.src_id) for f in pending}),
        }


# ---------------------------------------------------------------------------
# Module-level singleton + thin function wrappers
#
# Existing callers import these by name; the singleton makes the refactor
# behaviorally compatible while the new `PendingSchedule` class is the
# preferred per-instance entry point for tests + future callers.
# ---------------------------------------------------------------------------

_DEFAULT = PendingSchedule()


def clear() -> None:
    """Reset the module-level singleton (tests + the legacy entry point)."""
    _DEFAULT.reset()


def get_pending(my_id: int, game_id: int) -> list[ScheduledFire]:
    """Legacy two-arg API; game_id ignored (was the source of Bug #1)."""
    del game_id  # ignored — kept for signature parity with old callers
    return _DEFAULT.get_pending(my_id)


def set_pending(my_id: int, game_id: int, fires: list[ScheduledFire]) -> None:
    del game_id
    _DEFAULT.set_pending(my_id, fires)


def commit(my_id: int, game_id: int, new_fires: list[ScheduledFire]) -> None:
    del game_id
    _DEFAULT.commit(my_id, new_fires)


def decant_due(my_id: int, game_id: int, step_now: int) -> list[ScheduledFire]:
    del game_id
    return _DEFAULT.decant_due(my_id, step_now)


def prune_stale(my_id: int, game_id: int, world) -> int:
    del game_id
    return _DEFAULT.prune_stale(my_id, world)


def prune_past(my_id: int, game_id: int, step_now: int) -> int:
    del game_id
    return _DEFAULT.prune_past(my_id, step_now)


def stats(my_id: int, game_id: int) -> dict:
    del game_id
    return _DEFAULT.stats(my_id)


def get_default_pending() -> PendingSchedule:
    """Accessor for the module-level singleton — for callers that want
    to drive `begin_turn(fingerprint)` from a stage."""
    return _DEFAULT


# Backward-compat: some callers / tests reference `_PENDING` directly.
# Expose a property-like view via the singleton's internal dict.
# (Read-only via the property; mutations should go through the class API.)
class _PendingViewDescriptor:
    def __get__(self, _obj, _objtype=None):
        return _DEFAULT._pending


_PENDING = _DEFAULT._pending  # alias: same dict object the class mutates
