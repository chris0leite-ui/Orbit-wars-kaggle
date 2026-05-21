"""Fully-analytical agent (Phases 1-5; pipeline-composed in Phase A).

Entry point for the multi-turn joint solver agent. Composes the seven
analytical-pipeline stages from `lib.pipeline` and exposes the standard
`agent(obs, configuration) -> moves` signature used by kaggle_environments.

Phase A (2026-05-21): rewired from `lib.joint_solver.mpc.solve_turn` to
`lib.pipeline.default_composition()`. The composition is bit-parity with
the prior solve_turn entry point — same stages, same order, same
primitives, same defaults. Future phases (B-D) swap individual stages
from the registry without touching this entry point.

See `/root/.claude/plans/spicy-marinating-token.md` for the architectural
intent and the seven-stage contract.
"""

from __future__ import annotations

import os

from lib.pipeline import default_composition


# Compose once at module load; the composition itself is stateless so this
# is safe even when the tournament harness shares a worker across agents.
_AGENT = default_composition()


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
        return _AGENT(obs, configuration)
    finally:
        if prev_drain is None:
            os.environ.pop("PROPOSER_DRAIN_FILTER", None)
        else:
            os.environ["PROPOSER_DRAIN_FILTER"] = prev_drain
        if prev_hold is None:
            os.environ.pop("PROPOSER_HOLD_FEASIBILITY", None)
        else:
            os.environ["PROPOSER_HOLD_FEASIBILITY"] = prev_hold
