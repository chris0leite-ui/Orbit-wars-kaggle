"""Tunables for the holdgrab agent — single source of truth.

Every magic number the strategy depends on lives here as a field on the
frozen ``Config`` dataclass, so iterating the agent is "edit one struct,
re-run the panel". No logic lives in this module.

Design: ``knowledge-base/concepts/play-against-the-board-research.md`` and
the plan in ``/root/.claude/plans/i-want-you-to-steady-snowglobe.md``.
The one idea: *grab the production you can hold* — value each planet by
production x hold-time, gated by a binary "would I still hold it under the
nearest worst-case attack" check, greedily, re-solved every turn.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- horizon ---
    game_horizon: int = 500            # env hard step cap; ceiling on hold_time.

    # --- offense hold gate (force-based, full-attack-future) ---
    contest_window: int = 30           # reaction window over which the enemy's
                                       # full reachable counter to a capture is
                                       # summed. A capture is allowed iff we can
                                       # afford to beat that counter (so we attack
                                       # when we hold a force lead, and never
                                       # over-extend). The key offense/defense dial:
                                       # larger = more cautious.

    # --- defense floor ---
    defense_floor_horizon: int = 60    # only COMMITTED in-flight enemy fleets
                                       # arriving within this many ticks reserve
                                       # garrison on one of my planets. In-flight
                                       # (not hypothetical) is the guard against the
                                       # rf over-pessimism turtle (audit 2026-05-27).

    # --- value ---
    enemy_capture_weight: float = 2.0  # taking an enemy planet gains us P/turn AND
                                       # denies them P/turn (zero-sum) -> double.
    neutral_capture_weight: float = 1.0

    # --- 4-player ---
    weakest_opp_bias: float = 0.30     # multiplicative boost on captures from the
                                       # weakest opponent (4P only): attack-weakest.

    # --- selection ---
    order_by_roi: bool = True          # greedy by value-per-ship (knapsack-
                                       # correct; what the roi opponent does)
                                       # rather than raw value.

    # --- candidate enumeration ---
    max_targets_per_source: int = 12   # nearest-N target prune per source.
    max_arrival_lead: int = 200        # drop launches whose ETA exceeds this.
    validate_physics: bool = True      # drop sun / out-of-bounds / non-target
                                       # trajectories via predict_fleet_fate (Rule 47).

    # --- anytime budget ---
    hard_budget_ms: float = 950.0      # safety break inside the greedy commit loop;
                                       # the closed-form pass fits well under this.


DEFAULT = Config()
