"""chooser_roi — ROI-prior + opp-modifier chooser (Phase 1 stub).

Architectural pivot (2026-05-19, PI-directed): replace the trajectory rollout
foundation with a closed-form ROI prior + thin opp-vulnerability posterior.
This module is the new chooser, dispatched via BASELINE_CHOOSER=roi from
agents/baseline/main.py.

This is the Phase 1 stub: returns no moves, so the env-var toggle is wired
without changing default behaviour. The actual ROI implementation lands in
Phase 2 (solo_roi), Phase 3 (coalition_roi), Phase 4 (opp_modifier_check).

See /root/.claude/plans/okay-we-can-do-elegant-lampson.md for the full plan.
"""

from __future__ import annotations


def choose_roi(
    snap_base,
    prerank,
    me: int,
    num_seats: int,
    wallclock_ms: float,
    min_horizon: int,
    max_horizon: int,
    gamma: float,
    world,
    model,
    step: int,
) -> list:
    """Return final move list under the ROI-prior architecture.

    Phase 1: returns []. Subsequent phases will fill in solo_roi, coalition
    enumeration, and the opp-modifier posterior.
    """
    return []
