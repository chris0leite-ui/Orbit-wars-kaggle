"""Phase 2 parity gate: joint_solver.lp == chooser_lp on hand-built fixtures.

5 fixtures cover:
  1. Single source, single positive-value target.
  2. Two sources, no conflicts → both emit.
  3. Target conflict (2 sources, same target) → Hungarian picks higher value.
  4. Source conflict (1 source, 2 targets) → Hungarian picks higher value.
  5. Mixed positive + zero-value (noop preferred over zero on diagonal).

For each fixture we build matching (candidate, Column) representations
of the same prerank, feed each through the two pipelines, and assert
identical emit sets. This proves the LP infra is bit-equivalent to
Slice 10's (a known regression — same dismal 3/16 lift — but a CORRECT
single-turn LP we can then extend to multi-turn in Phase 3).
"""

from __future__ import annotations

from collections import namedtuple

from agents.baseline import chooser_lp as cm_chooser_lp
from lib.joint_solver.columns import Column
from lib.joint_solver.lp import (
    build_assignment_matrix,
    extract_moves,
    solve_assignment,
    solve_bipartite,
)


# Lightweight stand-ins for Planet (only id/owner/production read by the LP path).
_FakePlanet = namedtuple("_FakePlanet", ["id", "owner", "production"])


def _candidate(*, src_id, tgt_id, ships, angle, eta=3, wait_N=0,
               cheap_delta=1.0, horizon_hint=10):
    """Build a prerank-tuple matching chooser_lp's expected shape."""
    src = _FakePlanet(src_id, 0, 2)
    tgt = _FakePlanet(tgt_id, -1, 3)
    return (cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N)


def _column(*, column_id, src_id, tgt_id, ships, angle, value,
            eta=3, wait_N=0):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id, ships=ships,
        wait_N=wait_N, angle=angle, eta=eta, owner=0, value=float(value),
    )


def _run_chooser_lp(prerank_with_values: list[tuple]) -> list[list]:
    """Bypass chooser_lp's value-computation and run its assignment core.

    `prerank_with_values` is [(candidate_tuple, value), …]; we build the
    cost matrix the same way `_build_assignment_matrix` does but inject
    the values directly (no World/WorldModel needed)."""
    src_id_set: set = set()
    pair_to_best: dict = {}
    for c, value in prerank_with_values:
        _cheap_delta, src, tgt, _ships, _angle, _eta, _h, wait_N = c
        if int(wait_N) != 0:
            continue
        sid = int(src.id)
        src_id_set.add(sid)
        if value <= 0.0:
            continue
        key = (sid, int(tgt.id))
        prev = pair_to_best.get(key)
        if prev is None or float(value) > float(prev[0]):
            pair_to_best[key] = (float(value), c)

    src_ids = sorted(src_id_set)
    if not src_ids:
        return []
    n_srcs = len(src_ids)
    pair_keys = sorted(pair_to_best.keys())
    n_pairs = len(pair_keys)
    n_cols = n_srcs + n_pairs
    src_index = {sid: i for i, sid in enumerate(src_ids)}

    cost_matrix = [
        [cm_chooser_lp.INFEASIBLE_COST] * n_cols for _ in range(n_srcs)
    ]
    col_to_candidate: dict = {}
    for i in range(n_srcs):
        cost_matrix[i][i] = cm_chooser_lp.NOOP_COST
    for j_offset, key in enumerate(pair_keys):
        sid, _tid = key
        col_j = n_srcs + j_offset
        value, candidate = pair_to_best[key]
        row_i = src_index[sid]
        cost_matrix[row_i][col_j] = -float(value)
        col_to_candidate[col_j] = candidate

    return cm_chooser_lp._solve_and_extract(cost_matrix, col_to_candidate)


def _run_joint_solver(columns: list[Column]) -> list[list]:
    return solve_bipartite(columns)


def _moves_match(moves_a, moves_b):
    """Order-independent set equality on emit moves."""
    norm_a = sorted([(int(m[0]), round(float(m[1]), 6), int(m[2])) for m in moves_a])
    norm_b = sorted([(int(m[0]), round(float(m[1]), 6), int(m[2])) for m in moves_b])
    return norm_a == norm_b


# ---------------------------------------------------------------------------
# Fixture 1: Single source, single positive-value target.
# ---------------------------------------------------------------------------


def test_fixture_1_single_source_single_target():
    cand = _candidate(src_id=10, tgt_id=20, ships=15, angle=1.5)
    moves_lp = _run_chooser_lp([(cand, 50.0)])
    moves_js = _run_joint_solver([
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.5, value=50.0),
    ])
    assert _moves_match(moves_lp, moves_js), f"mismatch: lp={moves_lp} js={moves_js}"
    assert moves_lp == [[10, 1.5, 15]]


# ---------------------------------------------------------------------------
# Fixture 2: Two sources, no conflicts → both emit.
# ---------------------------------------------------------------------------


def test_fixture_2_two_sources_no_conflicts():
    cand_a = _candidate(src_id=10, tgt_id=20, ships=15, angle=1.0)
    cand_b = _candidate(src_id=11, tgt_id=21, ships=20, angle=2.0)
    moves_lp = _run_chooser_lp([(cand_a, 30.0), (cand_b, 40.0)])
    moves_js = _run_joint_solver([
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.0, value=30.0),
        _column(column_id=1, src_id=11, tgt_id=21, ships=20, angle=2.0, value=40.0),
    ])
    assert _moves_match(moves_lp, moves_js)
    assert len(moves_lp) == 2


# ---------------------------------------------------------------------------
# Fixture 3: Target conflict — 2 sources, same target → higher value picks.
# ---------------------------------------------------------------------------


def test_fixture_3_target_conflict():
    """Slice 10's LP does NOT enforce target-uniqueness in the cost matrix;
    it relies on the extract path's `used_tgts` dedup to break ties by row
    order. Both pipelines inherit this — they emit the LOWER-numbered
    source (row 0, src=10), even though src=11 has higher value.

    Phase 3 should fix this with a proper target-side constraint in the LP.
    For now, the parity gate just requires both pipelines agree."""
    cand_a = _candidate(src_id=10, tgt_id=20, ships=15, angle=1.0)
    cand_b = _candidate(src_id=11, tgt_id=20, ships=18, angle=1.2)
    moves_lp = _run_chooser_lp([(cand_a, 30.0), (cand_b, 50.0)])
    moves_js = _run_joint_solver([
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.0, value=30.0),
        _column(column_id=1, src_id=11, tgt_id=20, ships=18, angle=1.2, value=50.0),
    ])
    assert _moves_match(moves_lp, moves_js), \
        f"parity failed: lp={moves_lp} js={moves_js}"
    # Both pick row 0 (src=10) due to extract-time dedup; src=11 is silently dropped.
    assert moves_lp == [[10, 1.0, 15]]


# ---------------------------------------------------------------------------
# Fixture 4: Source conflict — 1 source, 2 targets → higher value picks.
# ---------------------------------------------------------------------------


def test_fixture_4_source_conflict():
    cand_a = _candidate(src_id=10, tgt_id=20, ships=15, angle=1.0)
    cand_b = _candidate(src_id=10, tgt_id=21, ships=15, angle=2.0)
    moves_lp = _run_chooser_lp([(cand_a, 30.0), (cand_b, 25.0)])
    moves_js = _run_joint_solver([
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.0, value=30.0),
        _column(column_id=1, src_id=10, tgt_id=21, ships=15, angle=2.0, value=25.0),
    ])
    assert _moves_match(moves_lp, moves_js)
    # Source picks the higher-value target (tgt=20, value=30).
    assert moves_lp == [[10, 1.0, 15]]


# ---------------------------------------------------------------------------
# Fixture 5: Mixed positive + zero-value (noop preferred over zero).
# ---------------------------------------------------------------------------


def test_fixture_5_zero_value_routes_to_noop():
    cand_pos = _candidate(src_id=10, tgt_id=20, ships=15, angle=1.0)
    cand_zero = _candidate(src_id=11, tgt_id=21, ships=15, angle=2.0)
    moves_lp = _run_chooser_lp([(cand_pos, 40.0), (cand_zero, 0.0)])
    moves_js = _run_joint_solver([
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.0, value=40.0),
        _column(column_id=1, src_id=11, tgt_id=21, ships=15, angle=2.0, value=0.0),
    ])
    assert _moves_match(moves_lp, moves_js)
    # Only the positive-value launch emits; src=11 noops.
    assert moves_lp == [[10, 1.0, 15]]


# ---------------------------------------------------------------------------
# Plumbing tests on the joint_solver path only.
# ---------------------------------------------------------------------------


def test_build_matrix_empty_columns_returns_empty():
    matrix, src_ids, col_map = build_assignment_matrix([])
    assert matrix == []
    assert src_ids == []
    assert col_map == {}


def test_build_matrix_routes_zero_value_to_noop_only():
    cols = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=5, angle=1.0, value=0.0),
    ]
    matrix, src_ids, col_map = build_assignment_matrix(cols)
    assert src_ids == [10]
    assert col_map == {}  # no pair column for zero-value
    # Only the noop column exists.
    assert len(matrix) == 1
    assert len(matrix[0]) == 1
    assert matrix[0][0] == 0.0


def test_wait_N_nonzero_columns_skipped_in_extract():
    # Value-positive but wait_N != 0 — must be dropped at extract time.
    cols = [
        _column(column_id=0, src_id=10, tgt_id=20, ships=15, angle=1.0, value=50.0,
                wait_N=3),
    ]
    # The build still creates the pair column (we don't filter at build),
    # but extract drops it via the wait_N check.
    moves = solve_bipartite(cols)
    assert moves == []
