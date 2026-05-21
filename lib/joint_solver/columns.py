"""Column abstraction for the joint LP.

A "column" is one possible launch — a single (src, tgt, ships, wait_N,
angle, eta) tuple plus a closed-form value. The LP decides which columns
to fire by setting x_i ∈ {0,1} on each.

This module wraps the existing proposer / migration_solver / opp-projection
output (in the prerank tuple format used by chooser_trajectory and
chooser_lp) into a uniform Column dataclass that downstream LP / MPC
code can iterate over without caring about the upstream variant.

Phase 2 (this session): single-turn columns only (wait_N=0). The
column space is identical to chooser_lp's prerank-filtered space, which
is what enables the LP parity gate.

Phase 3 (next session): expands to a multi-turn t-indexed column space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Column:
    """One LP decision variable: "fire this specific launch."

    Fields mirror the prerank tuple convention `(cheap_delta, src, tgt,
    ships, angle, eta, horizon_hint, wait_N)` but with explicit names and
    a `value` slot for the LP cost coefficient (filled by value computation).
    """
    column_id: int          # unique LP index
    src_id: int
    tgt_id: int
    ships: int
    wait_N: int             # 0 for fire-now (Phase 2); >0 enabled in Phase 3+
    angle: float
    eta: int                # closed-form arrival tick from current snapshot
    owner: int              # owner emitting the launch (me for our cols; opp for theirs)
    value: float = 0.0      # closed-form value (cost coefficient = −value)
    horizon_hint: int = 0   # passed through for downstream consumers
    cheap_delta: float = 0.0  # passed through (proposer's cheap-ranked Δ)
    is_opp: bool = False    # True for opp-projected columns (Phase 3+)


def column_from_candidate(c, *, column_id: int, owner: int, value: float = 0.0,
                          is_opp: bool = False) -> Column:
    """Convert a prerank tuple into a Column.

    Prerank tuple shape (matches chooser_trajectory / chooser_lp):
      (cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N)
    """
    cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N = c
    return Column(
        column_id=int(column_id),
        src_id=int(src.id),
        tgt_id=int(tgt.id),
        ships=int(ships),
        wait_N=int(wait_N),
        angle=float(angle),
        eta=int(eta),
        owner=int(owner),
        value=float(value),
        horizon_hint=int(horizon_hint),
        cheap_delta=float(cheap_delta),
        is_opp=bool(is_opp),
    )


def columns_from_prerank(prerank, *, me: int, value_fn,
                         start_column_id: int = 0) -> list[Column]:
    """Build a column list from a prerank, computing value via `value_fn`.

    `value_fn(candidate, me) -> float` should return the closed-form
    value of firing that candidate (e.g., capture EV, migration EV,
    defensive reinforce value). Columns with value <= 0 are still
    returned but the LP build will route them to a noop column.

    Only emits columns with wait_N == 0 (Phase 2 single-turn convention).
    """
    columns: list[Column] = []
    next_id = int(start_column_id)
    for c in prerank:
        wait_N = int(c[7])
        if wait_N != 0:
            continue
        value = float(value_fn(c, me))
        columns.append(column_from_candidate(
            c, column_id=next_id, owner=int(me), value=value,
        ))
        next_id += 1
    return columns
