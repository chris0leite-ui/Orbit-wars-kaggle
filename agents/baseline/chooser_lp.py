"""LP chooser — joint bipartite-assignment over the whole turn's move-set.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §16.

The architectural shift: prior choosers scored candidates
independently and emitted greedily. This chooser solves the
WHOLE TURN'S allocation in one step via Hungarian assignment:
rows = our sources, cols = candidate targets (+ noop per source),
cell value = − closed-form value of "fire src_i at tgt_j."

scipy.optimize.linear_sum_assignment runs in microseconds for
typical ≤30 sources × ≤30 targets. Pure-Python greedy fallback
if scipy is unavailable in the bundle environment.

Why this differs from Slice 6 (which also used LP but failed):
- Slice 6 used LP as a "commit hint" ON TOP of the trajectory
  chooser. Two decision-makers; their disagreement created noise.
- Slice 10 makes the LP the ONLY decision-maker. No rollout to
  disagree with; no greedy emit to undermine the LP's choice.

Total unimodularity of the bipartite-matching LP guarantees
LP-relaxation optima are integer at the solver's polytope
extreme points. No rounding needed.
"""

from __future__ import annotations

import math

# Single-line imports below — bundler constraint (see proposer.py:71-76).
# scipy may or may not be present in Kaggle's bundle environment; the
# try-import + greedy fallback keeps the bundle self-contained.
try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _hungarian = None  # type: ignore[assignment]

from agents.baseline.predicates import _w1_value_bounds
from agents.baseline.predicates import w2_provably_held_reinforce
from agents.baseline.strategic_lp import _greedy_assignment as _greedy_assign
from lib.scoring import pv_horizon


EPISODE_END: int = 500
DEFAULT_GAMMA: float = 0.99
# Sentinel cost for infeasible (i, j) assignments. Hungarian treats
# these as "never pick" relative to real values.
INFEASIBLE_COST: float = 1e9
# Noop column cost — slightly negative so a source PREFERS not
# launching over a zero-value launch, but real positive-value
# launches will dominate. Float small relative to real values.
NOOP_COST: float = 0.0
# Reinforce value multiplier: a successfully defended planet keeps
# tgt.production × pv of production for us, and prevents opp from
# stealing it. Factor 2.0 mirrors the "favor delta" interpretation
# (my_prod up + opp_prod down vs the unsuccessful-defense baseline).
W2_VALUE_MULTIPLIER: float = 2.0


def _compute_candidate_value(c, world, model, me: int,
                             gamma: float = DEFAULT_GAMMA) -> float:
    """Return the closed-form value for a single candidate.

    Dispatches by candidate class:
      - wait_N > 0 → 0 (same rationale as Slice 8c — single-turn LP).
      - Migration (`tgt.owner == me`, no inbound threat) → use the
        solver-attached `cheap_delta` directly.
      - Defensive reinforce (`tgt.owner == me`, inbound threat) →
        W2 verdict; if commit, value = 2 × prod × pv_horizon.
      - Capture (`tgt.owner != me`) → `_w1_value_bounds` lower bound
        (the Wald-conservative value).
      - Otherwise → 0.

    Zero values get mapped to the noop column at LP-build time; the
    LP never emits them.
    """
    cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N = c

    # Wait-N>0 filter — same single-turn restriction as Slice 8c.
    if int(wait_N) != 0:
        return 0.0

    # Own→own classification: migration vs defensive reinforce.
    if int(tgt.owner) == int(me):
        try:
            threat_eta = model.time_to_enemy_threat(
                int(tgt.id), int(me), world,
            )
        except Exception:
            threat_eta = None

        if threat_eta is None:
            # Migration — solver's cheap_delta is the value.
            value = float(cheap_delta)
            return value if value > 0.0 else 0.0

        # Defensive reinforce — W2 verdict.
        try:
            verdict = w2_provably_held_reinforce(
                src, tgt, int(ships), int(wait_N), int(eta),
                world, model, int(me),
            )
        except Exception:
            return 0.0
        if verdict.kind != "commit":
            return 0.0
        # W2 returns lower_bound=0 by design; compute value separately.
        step = int(getattr(world, "step", 0) or 0)
        arrival = int(wait_N) + int(eta)
        pv = pv_horizon(int(step), int(arrival), gamma=float(gamma),
                        t_total=EPISODE_END)
        return W2_VALUE_MULTIPLIER * float(int(tgt.production)) * float(pv)

    # Capture — use the differential's bounded-interval lower bound.
    try:
        lo, hi = _w1_value_bounds(
            src, tgt, int(ships), int(wait_N), int(eta),
            world, model, int(me), gamma=float(gamma),
        )
    except Exception:
        return 0.0
    return float(lo) if lo > 0.0 else 0.0


def _build_assignment_matrix(prerank, world, model, me: int,
                             gamma: float = DEFAULT_GAMMA):
    """Construct the (sources × columns) cost matrix for assignment.

    Returns `(cost_matrix, src_ids, col_to_candidate)`:
      - `cost_matrix`: list of lists, shape (N_srcs, N_cols).
      - `src_ids`: ordered list of source planet IDs (row labels).
      - `col_to_candidate`: dict[col_index → candidate_tuple] for
        non-noop columns. Noop columns are absent from this dict.

    Layout:
      - First N_srcs columns are "noop per source" (cost 0, source i
        gets its own noop in column i).
      - Remaining columns are unique (src, tgt) pairs from the
        prerank, one per distinct (src_id, tgt_id) combination.
        Cell cost = -max value over candidates with that (src, tgt).

    Infeasible cells (no candidate value > 0) are filled with
    INFEASIBLE_COST so the solver never picks them over a real
    value or a noop.

    Why noop-per-source: each source is allowed to NOT launch (one
    of its row's cells must be picked under Hungarian; the noop
    cell at row i is preferable to a +INFEASIBLE_COST cell).
    """
    # Collect unique sources and (src, tgt) pairs.
    src_id_set: set = set()
    pair_to_best: dict = {}  # (src_id, tgt_id) → (best_value, candidate)
    for c in prerank:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        sid = int(src.id)
        src_id_set.add(sid)
        value = _compute_candidate_value(c, world, model, int(me), gamma)
        if value <= 0.0:
            continue
        key = (sid, int(tgt.id))
        prev = pair_to_best.get(key)
        if prev is None or float(value) > float(prev[0]):
            pair_to_best[key] = (value, c)

    src_ids = sorted(src_id_set)
    if not src_ids:
        return [], [], {}

    n_srcs = len(src_ids)
    # Columns: N_srcs noop columns + one per (src, tgt) pair.
    pair_keys = sorted(pair_to_best.keys())
    n_pairs = len(pair_keys)
    n_cols = n_srcs + n_pairs

    src_index = {sid: i for i, sid in enumerate(src_ids)}

    # Initialize cost matrix.
    cost_matrix: list = [
        [INFEASIBLE_COST] * n_cols for _ in range(n_srcs)
    ]
    col_to_candidate: dict = {}

    # Noop columns: cost 0 only on the diagonal (source i can noop in
    # column i). Off-diagonal noop cells are infeasible (a source
    # can't pick another source's noop).
    for i in range(n_srcs):
        cost_matrix[i][i] = NOOP_COST

    # Pair columns.
    for j_offset, key in enumerate(pair_keys):
        sid, tid = key
        col_j = n_srcs + j_offset
        value, candidate = pair_to_best[key]
        row_i = src_index[sid]
        cost_matrix[row_i][col_j] = -float(value)
        col_to_candidate[col_j] = candidate

    return cost_matrix, src_ids, col_to_candidate


def _solve_and_extract(cost_matrix, col_to_candidate) -> list:
    """Solve the assignment LP and return emit-shape moves.

    Calls scipy's `linear_sum_assignment` when available; falls back
    to the pure-Python greedy from `strategic_lp` otherwise.

    Returns a list of `[src_id, angle, ships]` triples ready for the
    env. Only fires `wait_N == 0` candidates (the filter is already
    applied at value-compute time).
    """
    if not cost_matrix or not cost_matrix[0]:
        return []

    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])

    if _SCIPY_AVAILABLE:
        import numpy as np
        cost_arr = np.array(cost_matrix, dtype=float)
        row_ind, col_ind = _hungarian(cost_arr)
    else:
        row_ind, col_ind = _greedy_assign(cost_matrix)

    moves: list = []
    used_srcs: set = set()
    used_tgts: set = set()
    for i, j in zip(row_ind, col_ind):
        if int(j) not in col_to_candidate:
            continue  # noop column — source picked "don't launch"
        c = col_to_candidate[int(j)]
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        if int(wait_N) != 0:
            continue  # belt-and-suspenders (value-compute already filters)
        sid = int(src.id)
        tid = int(tgt.id)
        # Belt-and-suspenders: LP shouldn't allow conflicts, but guard.
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        moves.append([sid, float(angle), int(ships)])
    return moves


def choose_lp(snap_base, prerank, baseline_favors,
              me: int, num_seats: int, wallclock_ms: float,
              min_horizon: int, max_horizon: int, gamma: float,
              world, model) -> list:
    """Joint-LP chooser. Signature-compatible with `choose_trajectory`.

    Unused kwargs (`snap_base`, `baseline_favors`, `min_horizon`,
    `max_horizon`, `wallclock_ms`) are accepted for parity but the
    LP doesn't use them — assignment runs in microseconds, no
    wallclock budgeting needed.

    Pipeline:
      1. For each candidate, compute its analytical value.
      2. Build the bipartite cost matrix.
      3. Hungarian assignment (or greedy fallback).
      4. Extract moves from the solution.
    """
    if not prerank:
        return []

    cost_matrix, src_ids, col_to_candidate = _build_assignment_matrix(
        prerank, world, model, int(me), float(gamma),
    )
    if not cost_matrix:
        return []

    return _solve_and_extract(cost_matrix, col_to_candidate)
