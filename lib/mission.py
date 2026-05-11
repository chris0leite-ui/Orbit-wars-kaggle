"""Mission dataclass — the unit a planner reasons over.

A *mission* is a typed candidate for a single fleet launch: src planet,
target planet, ships, score, and a mission class (`snipe` for v3.0;
`reinforce` / `recapture` / `gang_up` queued for v3.1+).

Why a separate dataclass rather than just Intent + score:
- Intent is the strategy/mechanism contract — pure (src, target, ships,
  aim_angle, arrival_xy). Adding score / class / eta there pollutes a
  data path that every mechanism depends on.
- The planner enumerates *many* candidates per source and ranks them; a
  Mission carries the metadata the planner needs while preserving the
  one-fleet-launch atomicity that downstream code expects.

Lifecycle: missions are produced by `lib/missions/<class>.py` builders,
ranked + filtered by `lib/planner.settle_plan`, then converted to Intent
at the planner boundary. The mechanism layer never sees Mission.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.intent import Intent


@dataclass
class Mission:
    """A typed fleet-launch candidate."""

    mission_class: str
    src_id: int
    target_id: int
    ships: int
    score: float
    eta: int               # estimated turns to arrival; for the planner's
                           # this-turn arrival-ledger tracking
    note: str = ""

    def to_intent(self) -> Intent:
        """Drop to the strategy/mechanism contract."""
        note = (
            f"{self.mission_class}:{self.note}" if self.note
            else self.mission_class
        )
        return Intent(
            src_id=self.src_id,
            target_id=self.target_id,
            ships=self.ships,
            note=note,
        )
