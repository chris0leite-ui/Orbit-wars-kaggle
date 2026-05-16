"""Mission-book — TTL-based persistence for committed (src, tgt) intents.

Avoids the per-turn churn pathology: today's chooser re-enumerates
candidates from scratch every turn, so a target picked at step 50 may
be dropped at step 51 in favour of a marginally-higher-scoring candidate,
even though a one-turn switch wastes both the new and old fleet's
travel time.

PI's directive: "missions … continue set or multistep strategies."
MVP shape (per Phase-1 question): missions carry a short TTL (default 3
turns), get a small carryforward bonus on the next scoring pass, and
expire OR drop if their preconditions invalidate (src lost, target
captured by us, target captured by enemy, etc.).

Same module-level state pattern as `lib/missions/recapture.py`'s
`_RecaptureState`: a single MissionBook instance with `reset_if_new_game`
called on every turn so a fresh game wipes state cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


CARRYFORWARD_WEIGHT = 0.10
DEFAULT_TTL = 3


@dataclass
class CommittedMission:
    src_id: int
    target_id: int
    score_at_commit: float
    committed_step: int
    ttl: int  # turns remaining
    target_owner_at_commit: int


class MissionBook:
    """Per-game persistent commitments. Single global instance — see bottom."""

    def __init__(self) -> None:
        self._book: dict[tuple[int, int], CommittedMission] = {}
        self._last_step: int = -1

    def reset_if_new_game(self, step: int) -> None:
        """Wipe state at game start. Detection: step==0, OR step decreased
        vs last call (different game in same process)."""
        if step == 0 or step < self._last_step:
            self._book.clear()
        self._last_step = step

    def carryforward(self, world, model, me: int) -> dict[tuple[int, int], CommittedMission]:
        """Return the SUBSET of committed missions whose preconditions still
        hold this turn. Drops invalidated commits IN PLACE so they don't
        reappear next turn either.

        Validity gates:
        - src still owned by `me`
        - target ownership changed from commit-time (handled separately:
          if target is now ours → mission fulfilled; if enemy captured it
          → re-target may apply, but this commit is stale).
        """
        valid: dict[tuple[int, int], CommittedMission] = {}
        my_planet_ids = {int(p.id) for p in world.planets_by_id.values() if int(p.owner) == me}
        for key, cm in list(self._book.items()):
            src_id, target_id = key
            if src_id not in my_planet_ids:
                # We lost the source planet — commit is invalid.
                del self._book[key]
                continue
            tgt = world.planets_by_id.get(target_id)
            if tgt is None:
                # Target somehow vanished (comet expiration?). Drop.
                del self._book[key]
                continue
            current_owner = int(tgt.owner)
            commit_owner = cm.target_owner_at_commit
            if current_owner == me and commit_owner != me:
                # Capture/reinforce fulfilled — drop.
                del self._book[key]
                continue
            valid[key] = cm
        return valid

    def carryforward_bonus(self, src_id: int, target_id: int) -> float:
        """Bonus to add to a re-proposed candidate that matches a live commit.
        Returns 0.0 if no commit exists."""
        cm = self._book.get((int(src_id), int(target_id)))
        if cm is None:
            return 0.0
        # Decay by remaining TTL fraction so freshly-committed missions
        # get the full bonus and near-expiry ones taper off.
        decay = max(0.0, min(1.0, cm.ttl / float(DEFAULT_TTL)))
        return CARRYFORWARD_WEIGHT * abs(cm.score_at_commit) * decay

    def commit(self, src_id: int, target_id: int, score: float,
               step: int, target_owner: int, ttl: int = DEFAULT_TTL) -> None:
        """Register or refresh a commit. If the (src, tgt) pair already
        had a commit, the TTL is reset (we're re-confirming the same plan)."""
        key = (int(src_id), int(target_id))
        self._book[key] = CommittedMission(
            src_id=int(src_id),
            target_id=int(target_id),
            score_at_commit=float(score),
            committed_step=int(step),
            ttl=int(ttl),
            target_owner_at_commit=int(target_owner),
        )

    def decay_ttls(self) -> None:
        """Decrement all TTLs by 1 and drop those that hit 0. Call once
        per turn at the END of agent() so this turn's emits are scored
        against an undecayed TTL."""
        for key in list(self._book.keys()):
            self._book[key].ttl -= 1
            if self._book[key].ttl <= 0:
                del self._book[key]

    def size(self) -> int:
        return len(self._book)


# Single per-process instance. The agent module imports this directly.
BOOK = MissionBook()
