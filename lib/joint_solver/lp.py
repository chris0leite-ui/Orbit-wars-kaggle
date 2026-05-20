"""Linear-programming layer for the joint solver.

Phase 2: bipartite Hungarian assignment (rows=sources, cols=noop-per-source +
(src,tgt) pairs). Total-unimodularity of the assignment polytope guarantees
LP-relaxation extreme points are integer, so no rounding is needed.

The assignment math mirrors `agents/baseline/chooser_lp.py:_build_assignment_matrix`
and `_solve_and_extract`. The parity test `tests/test_joint_lp_parity.py`
verifies bit-for-bit equivalence on hand-built cost matrices.

Phase 3: extends to general LP with multi-turn decision variables x_{i,t}
linked via outcome-table subset constraints. The Hungarian path remains
as a fast warm-start / fallback.

Solver: scipy.optimize.linear_sum_assignment (HiGHS-equivalent for the
assignment polytope). Pure-Python greedy fallback if scipy is absent
(matches the Kaggle-bundle precedent in chooser_lp.py:33-38).
"""

from __future__ import annotations

from typing import Optional

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _hungarian = None  # type: ignore[assignment]

from agents.baseline.strategic_lp import _greedy_assignment as _greedy_assign
from lib.joint_solver.columns import Column


# Sentinel costs (mirror chooser_lp.py:50-54).
INFEASIBLE_COST: float = 1e9
NOOP_COST: float = 0.0


def build_assignment_matrix(columns: list[Column],
                            ) -> tuple[list[list[float]], list[int], dict[int, Column]]:
    """Construct the bipartite cost matrix from a column list.

    Layout:
      - Rows: unique source IDs (sorted asc).
      - Columns:
          [0, n_srcs)               — noop columns; only the diagonal
                                       cell (i, i) is feasible (cost NOOP_COST).
                                       Off-diagonal noop cells are infeasible
                                       so a source can't take another's noop.
          [n_srcs, n_srcs + n_pairs) — unique (src, tgt) pairs from columns
                                        with value > 0; each cell holds
                                        the BEST value across columns
                                        sharing that (src, tgt).
      - Infeasible cells default to INFEASIBLE_COST.

    Returns `(cost_matrix, src_ids, col_to_column)`:
      - `cost_matrix`: list of lists (N_srcs × N_cols), entries are
                       `-value` (negated so minimizing cost = maximizing value).
      - `src_ids`: ordered list of row labels.
      - `col_to_column`: dict[col_index → Column] for non-noop columns.
    """
    src_id_set: set[int] = set()
    pair_to_best: dict[tuple[int, int], Column] = {}

    for col in columns:
        src_id_set.add(int(col.src_id))
        if col.value <= 0.0:
            continue
        key = (int(col.src_id), int(col.tgt_id))
        prev = pair_to_best.get(key)
        if prev is None or float(col.value) > float(prev.value):
            pair_to_best[key] = col

    src_ids = sorted(src_id_set)
    if not src_ids:
        return [], [], {}

    n_srcs = len(src_ids)
    pair_keys = sorted(pair_to_best.keys())
    n_pairs = len(pair_keys)
    n_cols = n_srcs + n_pairs

    src_index = {sid: i for i, sid in enumerate(src_ids)}

    cost_matrix: list[list[float]] = [
        [INFEASIBLE_COST] * n_cols for _ in range(n_srcs)
    ]
    col_to_column: dict[int, Column] = {}

    # Noop columns: diagonal NOOP_COST, off-diagonal INFEASIBLE_COST.
    for i in range(n_srcs):
        cost_matrix[i][i] = NOOP_COST

    # Pair columns.
    for j_offset, key in enumerate(pair_keys):
        sid, _tid = key
        col_j = n_srcs + j_offset
        best_col = pair_to_best[key]
        row_i = src_index[sid]
        cost_matrix[row_i][col_j] = -float(best_col.value)
        col_to_column[col_j] = best_col

    return cost_matrix, src_ids, col_to_column


def solve_assignment(cost_matrix: list[list[float]]
                     ) -> tuple[list[int], list[int]]:
    """Solve the assignment LP. Returns `(row_ind, col_ind)` paired arrays.

    Uses scipy Hungarian when available; falls back to the pure-Python
    greedy from `strategic_lp` (matching the chooser_lp fallback path).
    """
    if not cost_matrix or not cost_matrix[0]:
        return [], []
    if _SCIPY_AVAILABLE:
        import numpy as np
        arr = np.array(cost_matrix, dtype=float)
        row_ind, col_ind = _hungarian(arr)
        return list(int(r) for r in row_ind), list(int(c) for c in col_ind)
    return _greedy_assign(cost_matrix)


def extract_moves(row_ind: list[int], col_ind: list[int],
                  col_to_column: dict[int, Column]) -> list[list]:
    """Read off the [src_id, angle, ships] launches from the assignment.

    Drops:
      - Noop assignments (j not in col_to_column).
      - wait_N != 0 columns (belt-and-suspenders; should be filtered at build).
      - Source/target dogpile conflicts (LP shouldn't produce these, but guard).
    """
    moves: list[list] = []
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        if int(j) not in col_to_column:
            continue
        col = col_to_column[int(j)]
        if int(col.wait_N) != 0:
            continue
        sid = int(col.src_id)
        tid = int(col.tgt_id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        moves.append([sid, float(col.angle), int(col.ships)])
    return moves


def solve_bipartite(columns: list[Column]) -> list[list]:
    """End-to-end: columns → cost matrix → assignment → moves."""
    cost_matrix, _src_ids, col_to_column = build_assignment_matrix(columns)
    row_ind, col_ind = solve_assignment(cost_matrix)
    return extract_moves(row_ind, col_ind, col_to_column)
