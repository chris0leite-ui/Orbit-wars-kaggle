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

from lib.joint_solver.mpc import solve_turn


def agent(obs, configuration=None):
    """Return a list of [src_id, angle, ships] launch tuples for this turn.

    Phase 5E+: scope PROPOSER_DRAIN_FILTER / PROPOSER_HOLD_FEASIBILITY
    overrides to THIS CALL ONLY by saving/restoring the previous values.
    Why this matters: the tournament harness shares a worker process
    across multiple agents (importlib spec_from_file_location). A
    module-load-time `os.environ.setdefault` would leak our filter
    overrides to the baseline running in the same worker, breaking
    the A/B's symmetry. By scoping per-call, baseline always runs
    with ITS intended defaults.
    """
    prev_drain = os.environ.get("PROPOSER_DRAIN_FILTER")
    prev_hold = os.environ.get("PROPOSER_HOLD_FEASIBILITY")
    os.environ["PROPOSER_DRAIN_FILTER"] = "off"
    os.environ["PROPOSER_HOLD_FEASIBILITY"] = "off"
    try:
        return solve_turn(obs, configuration)
    finally:
        if prev_drain is None:
            os.environ.pop("PROPOSER_DRAIN_FILTER", None)
        else:
            os.environ["PROPOSER_DRAIN_FILTER"] = prev_drain
        if prev_hold is None:
            os.environ.pop("PROPOSER_HOLD_FEASIBILITY", None)
        else:
            os.environ["PROPOSER_HOLD_FEASIBILITY"] = prev_hold
