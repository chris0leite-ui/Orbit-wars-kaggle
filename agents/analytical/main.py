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

from lib.joint_solver.mpc import solve_turn


def agent(obs, configuration=None):
    """Return a list of [src_id, angle, ships] launch tuples for this turn."""
    return solve_turn(obs, configuration)
