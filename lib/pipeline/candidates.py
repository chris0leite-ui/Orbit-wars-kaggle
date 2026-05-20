"""Stage 2 — Candidate generation.

TurnContext → CandidateSet (prerank list).

The reference implementation invokes `agents.baseline.proposer.propose`
plus `agents.baseline.migration_solver.propose_migrations`, matching the
exact ordering and shape used by `mpc.solve_turn:244-252`.
"""

from __future__ import annotations

from agents.baseline.migration_solver import propose_migrations
from agents.baseline.proposer import MAX_HORIZON, propose

from lib.pipeline.types import CandidateSet, TurnContext


def candidates_default(ctx: TurnContext) -> CandidateSet:
    """Reference Stage-2 implementation (parity with mpc.solve_turn).

    The target pool is `other_planets` plus our own planets that have a
    predicted incoming enemy threat — this matches mpc.solve_turn:240-244.
    """
    threatened_mine = [
        p for p in ctx.my_planets
        if ctx.model.time_to_enemy_threat(int(p.id), ctx.me, ctx.world) is not None
    ]
    target_pool = ctx.other_planets + threatened_mine

    prerank = propose(
        ctx.my_planets, target_pool, ctx.world, ctx.model, ctx.me, ctx.omega,
        baseline_len=MAX_HORIZON + 1,
    )
    migrations = propose_migrations(ctx.world, ctx.model, ctx.me)
    prerank = list(prerank) + list(migrations)

    return CandidateSet(prerank=prerank)
