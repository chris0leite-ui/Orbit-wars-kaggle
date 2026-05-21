"""Fully-analytical multi-turn joint solver (Phases 1-5).

Substrate replacement for the trajectory-rollout chooser. Composes
closed-form per-planet subgames into a multi-turn LP optimized via MPC.

See /root/.claude/plans/you-are-a-mathematician-clever-lighthouse.md
for the plan; built incrementally across phases with STOP gates at each.
"""
