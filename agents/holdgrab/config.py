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

    # --- value (denial-centric: optimize the differential, not own production) ---
    denial_weight: float = 1.0         # weight on the "production DENIED to the
                                       # opponent" term vs the "production GAINED"
                                       # term. 1.0 = the true differential swing
                                       # (enemy capture = 2x hold; contested
                                       # neutral rewarded for out-holding; safe
                                       # deep neutral = pure self-growth).

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

    # --- forward-rollout chooser (Plan v4) ---
    use_rollout: bool = True           # pick the turn whose SIMULATED end-state
                                       # wins, vs emitting the raw closed-form set.
    rollout_K: int = 20                # rollout horizon (ticks) per candidate turn.
    rollout_prefix_ks: tuple = (0, 1, 2, 3, 5, 8)
                                       # candidate turns = these prefix lengths of
                                       # the value-ordered committed launch set
                                       # (0 = idle .. full = spread). The prefix
                                       # that simulates best is emitted -> if
                                       # spreading churns, a short prefix wins.
    rollout_budget_ms: float = 800.0   # wallclock for the whole rollout pass;
                                       # falls back to the closed-form set if hit.

    # --- mass-to-HOLD consolidation (Plan v5; default OFF = behavior-preserving) ---
    # Pool ship budget across sources so a double-value ENEMY planet that no
    # single source can HOLD (only PRESSURE) can be captured-and-held. Linear
    # Lanchester means there is no concentration bonus -- this is purely
    # budget-aggregation, so it is a narrow band; the census (Plan v5 STEP 1)
    # gates whether it arises at all.
    consolidate_hold: bool = False         # master switch for the proposer.
    consolidate_max_targets: int = 3       # top-M enemy targets by planet_value.
    consolidate_max_legs: int = 2          # 2 = pairs only; 3 = allow triples.
    consolidate_max_eta_gap: int = 3       # faster source throttles to align ETA.
    consolidate_neutral: bool = False      # also consider contested neutrals.
    instrument_consolidation: bool = False  # census only; no behavior change.


DEFAULT = Config()
