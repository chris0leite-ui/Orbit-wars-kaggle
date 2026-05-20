"""Fully-analytical agent (Phases 1-5).

Entry point for the multi-turn joint solver agent. Wraps
`lib.joint_solver.mpc.solve_turn` to expose the standard
`agent(obs, configuration) -> moves` signature used by kaggle_environments.

Design: substrate replacement for the trajectory-rollout chooser.
Composes closed-form per-planet subgames into a multi-turn LP optimized
via MPC. See /root/.claude/plans/you-are-a-mathematician-clever-lighthouse.md
and the joint_solver Phase 1-3 commits for the full architecture.
"""

from __future__ import annotations

import os

# Phase 5E (2026-05-20): turn off the proposer's "drain" and "hold-
# feasibility" filters. Both are designed for the trajectory baseline's
# conservative rollout (reject candidates that look hard to defend after
# capture). Our outcome-aware LP VALUES short-hold captures positively
# (each tick of ownership = production stream). Discovered at seed 42
# step 80: proposer emitted 3 candidates total (2 own→own migrations + 1
# wait_N=10 capture) despite 5 trajectory-feasible opp targets in nearest-8;
# DRAIN+HOLD filters dropped all 5 because src=0 had projected threat at
# eta=8 and the targets had "strong nearby opp" candidates.
#
# TRAJECTORY filter stays ON — it drops physics-impossible launches
# (sun-kill, OOB) which the LP can't recover.
os.environ.setdefault("PROPOSER_DRAIN_FILTER", "off")
os.environ.setdefault("PROPOSER_HOLD_FEASIBILITY", "off")

from lib.joint_solver.mpc import solve_turn


def agent(obs, configuration=None):
    """Return a list of [src_id, angle, ships] launch tuples for this turn."""
    return solve_turn(obs, configuration)
