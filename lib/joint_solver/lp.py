"""Linear-programming layer for the joint solver.

Phase 2: bipartite Hungarian assignment (rows=sources, cols=noop-per-source +
(src,tgt) pairs). Total-unimodularity of the assignment polytope guarantees
LP-relaxation extreme points are integer, so no rounding is needed.

The assignment math mirrors `agents/baseline/chooser_lp.py:_build_assignment_matrix`
and `_solve_and_extract`. The parity test `tests/test_joint_lp_parity.py`
verifies bit-for-bit equivalence on hand-built cost matrices.

Phase 3: multi-turn binary LP via scipy.optimize.milp (HiGHS MILP). Drops
the wait_N==0 single-turn restriction; adds source-budget-over-time
constraints and per-target gang-up cap. See `solve_multi_turn` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _hungarian = None  # type: ignore[assignment]

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _MILP_AVAILABLE = True
except ImportError:
    _MILP_AVAILABLE = False
    milp = None  # type: ignore[assignment]
    LinearConstraint = None  # type: ignore[assignment]
    Bounds = None  # type: ignore[assignment]

from lib.joint_solver.columns import Column


def _greedy_assign(cost_matrix):
    """Lazy proxy for agents.baseline.strategic_lp._greedy_assignment.

    Lazy import avoids bundler module-ordering issues: the agent
    subpackage gets inlined AFTER lib/, so a module-level import
    here would fire before _greedy_assignment is defined in a
    flat bundle. Function-scoped import resolves at call time when
    everything is loaded.
    """
    from agents.baseline.strategic_lp import _greedy_assignment
    return _greedy_assignment(cost_matrix)


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


# ---------------------------------------------------------------------------
# Phase 3: multi-turn LP
# ---------------------------------------------------------------------------


# Per-target gang-up cap. Allows up to this many columns to fire at the
# same target (enabling multi-source gang-ups), but prevents runaway
# over-launching at one target. Tunable; 3 lets two sources team up plus
# one reinforce slot.
DEFAULT_MAX_CONTESTERS_PER_TARGET = 3

# Default planning horizon for source-budget constraints. Columns with
# wait_N beyond this are still allowed but the budget constraint binds
# at wait_N (not horizon), so the cap mainly limits per-turn LP size.
DEFAULT_MAX_WAIT_N = 5


@dataclass(frozen=True)
class MultiTurnResult:
    """Output of solve_multi_turn."""
    moves: list[list]                # [src_id, angle, ships], only wait_N==0 emits
    fired_columns: list[Column]      # full set of columns the LP fired (any wait_N)
    objective: float                 # achieved objective
    status: str                      # solver status string
    n_vars: int
    n_constraints: int


def _source_inventory(columns: list[Column], world, *, my_id: int
                      ) -> dict[int, tuple[int, int]]:
    """For each our-source-id present in columns, return
    (initial_ships, production)."""
    out: dict[int, tuple[int, int]] = {}
    for col in columns:
        if int(col.owner) != int(my_id):
            continue
        sid = int(col.src_id)
        if sid in out:
            continue
        p = world.planets_by_id.get(sid)
        if p is None:
            continue
        out[sid] = (int(p.ships), int(p.production))
    return out


def _greedy_multi_turn_fallback(columns: list[Column], world, *, my_id: int,
                                max_contesters_per_target: int,
                                ) -> MultiTurnResult:
    """Pure-Python multi-turn greedy: descending-value pass with running
    source-budget and per-target gang-up cap. Used when scipy.milp is
    absent or returns infeasible."""
    inv = _source_inventory(columns, world, my_id=int(my_id))
    # Track per-source per-time-window cumulative emission.
    emitted_by_src_wait: dict[tuple[int, int], int] = {}
    target_count: dict[int, int] = {}
    fired: list[Column] = []
    moves: list[list] = []
    for col in sorted(columns, key=lambda c: float(c.value), reverse=True):
        if col.value <= 0.0:
            break
        if int(col.owner) != int(my_id):
            continue
        sid = int(col.src_id)
        if sid not in inv:
            continue
        initial, prod = inv[sid]
        wait_N = int(col.wait_N)
        # Cumulative emissions from this src up to wait_N (inclusive).
        used = sum(v for (s, w), v in emitted_by_src_wait.items()
                   if s == sid and w <= wait_N)
        if used + int(col.ships) > initial + prod * wait_N:
            continue
        if target_count.get(int(col.tgt_id), 0) >= max_contesters_per_target:
            continue
        emitted_by_src_wait[(sid, wait_N)] = (
            emitted_by_src_wait.get((sid, wait_N), 0) + int(col.ships)
        )
        target_count[int(col.tgt_id)] = target_count.get(int(col.tgt_id), 0) + 1
        fired.append(col)
        if wait_N == 0:
            moves.append([sid, float(col.angle), int(col.ships)])
    return MultiTurnResult(
        moves=moves, fired_columns=fired, objective=float("nan"),
        status="greedy_fallback", n_vars=0, n_constraints=0,
    )


def solve_multi_turn(
    columns: list[Column],
    world,
    *,
    my_id: int,
    max_contesters_per_target: int = DEFAULT_MAX_CONTESTERS_PER_TARGET,
    max_wait_N: int = DEFAULT_MAX_WAIT_N,
    time_limit_seconds: float = 0.3,
) -> MultiTurnResult:
    """Multi-turn binary LP.

    Decision variables: x_i ∈ {0,1} for each column i (ours only — opp
    columns are excluded; if you want to include opp projections, fold
    them into outcome_tables as fixed arrivals in mpc.py).

    Constraints:
      - Per-source budget over time: for each (src, t) with t ∈ [0, max_wait_N]:
          Σ_{i: src(i)=src, wait_N(i) ≤ t} ships(i) · x_i
            ≤ initial_ships(src) + t · production(src)
      - Per-target gang-up cap: Σ_{i: tgt(i)=tgt} x_i ≤ max_contesters_per_target

    Objective: max Σ_i value(i) · x_i.

    Columns with value ≤ 0, or with owner != my_id, are pinned to x_i=0.
    Columns with wait_N > max_wait_N are also pinned to 0 (out of plan).

    Emits ONLY wait_N==0 columns from the solution (MPC commits only
    the current-turn launches; the rest are projected planning intent).

    Falls back to greedy when scipy.milp is unavailable or solver
    returns infeasible / times out.
    """
    if not columns:
        return MultiTurnResult(moves=[], fired_columns=[], objective=0.0,
                               status="empty", n_vars=0, n_constraints=0)

    if not _MILP_AVAILABLE:
        return _greedy_multi_turn_fallback(
            columns, world, my_id=int(my_id),
            max_contesters_per_target=int(max_contesters_per_target),
        )

    import numpy as np

    inv = _source_inventory(columns, world, my_id=int(my_id))

    # Filter columns: ours, with valid source, value > 0, wait_N ≤ max_wait_N.
    active: list[Column] = []
    for col in columns:
        if int(col.owner) != int(my_id):
            continue
        if int(col.src_id) not in inv:
            continue
        if float(col.value) <= 0.0:
            continue
        if int(col.wait_N) > int(max_wait_N):
            continue
        active.append(col)

    if not active:
        return MultiTurnResult(moves=[], fired_columns=[], objective=0.0,
                               status="no_positive_columns",
                               n_vars=0, n_constraints=0)

    n = len(active)
    # Objective: maximize Σ value · x → minimize -Σ value · x.
    c = -np.array([float(col.value) for col in active], dtype=float)

    A_rows: list[list[float]] = []
    b_ub: list[float] = []

    # Source budget rows per (src, t).
    src_ids = sorted({int(col.src_id) for col in active})
    for sid in src_ids:
        initial, prod = inv[sid]
        # Indices of active columns from this source.
        src_cols = [(j, col) for j, col in enumerate(active)
                    if int(col.src_id) == sid]
        for t in range(0, int(max_wait_N) + 1):
            # Σ_{wait_N ≤ t} ships · x_i ≤ initial + t · prod
            row = [0.0] * n
            any_in_row = False
            for j, col in src_cols:
                if int(col.wait_N) <= t:
                    row[j] = float(col.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_rows.append(row)
            b_ub.append(float(initial + t * prod))

    # Per-target gang-up cap.
    tgt_ids = sorted({int(col.tgt_id) for col in active})
    for tid in tgt_ids:
        row = [0.0] * n
        any_in_row = False
        for j, col in enumerate(active):
            if int(col.tgt_id) == tid:
                row[j] = 1.0
                any_in_row = True
        if not any_in_row:
            continue
        A_rows.append(row)
        b_ub.append(float(max_contesters_per_target))

    if not A_rows:
        # No constraints — pick everything positive.
        return _greedy_multi_turn_fallback(
            columns, world, my_id=int(my_id),
            max_contesters_per_target=int(max_contesters_per_target),
        )

    A = np.array(A_rows, dtype=float)
    b = np.array(b_ub, dtype=float)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n, dtype=int)
    constraints = LinearConstraint(A, ub=b)

    try:
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": time_limit_seconds})
    except Exception:
        return _greedy_multi_turn_fallback(
            columns, world, my_id=int(my_id),
            max_contesters_per_target=int(max_contesters_per_target),
        )

    if res.x is None:
        return _greedy_multi_turn_fallback(
            columns, world, my_id=int(my_id),
            max_contesters_per_target=int(max_contesters_per_target),
        )

    fired: list[Column] = []
    moves: list[list] = []
    for j, col in enumerate(active):
        if res.x[j] > 0.5:
            fired.append(col)
            if int(col.wait_N) == 0:
                moves.append([int(col.src_id), float(col.angle), int(col.ships)])

    return MultiTurnResult(
        moves=moves, fired_columns=fired, objective=float(-res.fun),
        status=str(getattr(res, "message", "milp_ok")),
        n_vars=n, n_constraints=len(A_rows),
    )
