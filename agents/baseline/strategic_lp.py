"""Strategic LP — per-turn assignment-problem solve for long-horizon planning.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §12 Slice 6.

The chooser's per-turn decision is local: which candidate to emit
NOW. The LP provides a STRATEGIC ANCHOR: given the current
(source × target) capture-time matrix, which source-to-target
assignment maximizes long-horizon value? The LP runs once per
turn; matches between candidates and the LP assignment get a
"LP commit" verdict that backstops the inner chooser alongside
W1/W2.

Value model: each target captured at time `t` is worth
`tgt.production × (EPISODE_END - t)` — the production stream we
gain from holding it through the game's end. The LP finds the
assignment that maximizes the sum.

Algorithm: scipy's `linear_sum_assignment` (Hungarian-equivalent
in O(N³)) when scipy is importable; a pure-Python greedy
O(N² × log N) fallback otherwise. Both produce the same answer
on most boards (≤10 sources × ≤10 targets); greedy can be
suboptimal in pathological many-to-many cases but is always
correct for "assign each source to at most one target."

Cost: O(sources × targets) for matrix build + O(N³) for
Hungarian on ≤10×10 boards. Total <1 ms per turn.
"""

from __future__ import annotations

import math

# Single-line imports — bundler constraint (see proposer.py:71-76).
# scipy may or may not be present in Kaggle's bundle environment; the
# try-import + fallback keeps the bundle self-contained.
try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _hungarian = None  # type: ignore[assignment]

from lib.fleet import speed as fleet_speed


EPISODE_END: int = 500
# Sentinel cost for infeasible (i, j) pairs. Hungarian treats it as
# essentially "never pick"; greedy explicitly skips.
INFEASIBLE_COST: float = 1e9


def build_capture_matrix(world, model, me: int):
    """Build the (sources × targets) capture-time matrix.

    Returns:
        sources: list of (src_id, src_planet) for my planets with ≥1 ship.
        targets: list of (tgt_id, tgt_planet) for non-mine planets.
        matrix: 2D list `matrix[i][j]` = earliest tick source i can
                capture target j with its CURRENT ship count. Infeasible
                (under-sized capture or zero-speed fleet) → +inf.

    Capture-time math:
      - eta = ceil(flight_distance / fleet_speed(src.ships))
      - feasibility: src.ships > predicted garrison at arrival
      - Time-of-capture = eta (arrival tick; combat resolves there).

    No wait_N modelling — the LP captures "what can we DO this turn,"
    not "what can we plan over multiple turns." Wait-then-fire is
    handled separately by the proposer.
    """
    sources: list = []
    targets: list = []
    for p in world.planets_by_id.values():
        if int(p.owner) == int(me) and int(p.ships) > 0:
            sources.append((int(p.id), p))
        elif int(p.owner) != int(me):
            targets.append((int(p.id), p))

    matrix: list = []
    for sid, src in sources:
        row: list = []
        spd = fleet_speed(int(src.ships))
        for tid, tgt in targets:
            if spd <= 0:
                row.append(float("inf"))
                continue
            dist = math.hypot(
                float(src.x) - float(tgt.x),
                float(src.y) - float(tgt.y),
            )
            flight = max(0.0, dist - float(src.radius) - float(tgt.radius) - 0.1)
            eta = int(math.ceil(flight / spd))

            pred_garrison = model.ships_at(int(tgt.id), eta)
            if pred_garrison is None:
                row.append(float("inf"))
                continue
            required = int(math.ceil(float(pred_garrison))) + 1
            if required > int(src.ships):
                # Source can't fire now; LP doesn't model accumulation.
                row.append(float("inf"))
                continue
            row.append(float(eta))
        matrix.append(row)

    return sources, targets, matrix


def _build_cost_matrix(sources, targets, capture_times):
    """Convert capture-time matrix to (negated) value matrix for
    minimization. Lower cost = higher value. Infeasible pairs get
    `INFEASIBLE_COST` so the solver never picks them over a real pair.
    """
    n_s = len(sources)
    n_t = len(targets)
    cost: list = [[INFEASIBLE_COST] * n_t for _ in range(n_s)]
    for i, (sid, src) in enumerate(sources):
        for j, (tid, tgt) in enumerate(targets):
            t = capture_times[i][j]
            if t == float("inf"):
                continue
            value = float(int(tgt.production)) * float(EPISODE_END - t)
            if value > 0:
                cost[i][j] = -value
    return cost


def _greedy_assignment(cost):
    """Pure-Python greedy fallback. Picks the lowest-cost unblocked
    (i, j) pair iteratively, banning rows and columns as it goes.

    Strictly correct for one-source-to-one-target assignment (no
    hidden constraint to violate); may be suboptimal vs Hungarian
    for tightly coupled cases (e.g., when an early greedy pick
    forces a worse second pick). On typical Orbit Wars boards
    (≤10 sources/targets) the gap to optimal is negligible.

    Returns `(row_ind, col_ind)` as parallel lists mirroring scipy's
    `linear_sum_assignment` output shape (but as lists, not arrays).
    """
    n_s = len(cost)
    n_t = len(cost[0]) if n_s > 0 else 0
    used_rows: set = set()
    used_cols: set = set()
    row_ind: list = []
    col_ind: list = []
    while True:
        best_i = -1
        best_j = -1
        best_c = INFEASIBLE_COST
        for i in range(n_s):
            if i in used_rows:
                continue
            for j in range(n_t):
                if j in used_cols:
                    continue
                if cost[i][j] < best_c:
                    best_c = cost[i][j]
                    best_i = i
                    best_j = j
        if best_i < 0 or best_c >= INFEASIBLE_COST:
            break
        row_ind.append(best_i)
        col_ind.append(best_j)
        used_rows.add(best_i)
        used_cols.add(best_j)
    return row_ind, col_ind


def solve_assignment(sources, targets, capture_times) -> dict:
    """Return `{src_id: tgt_id}` of the optimal one-to-one assignment
    maximizing `Σ tgt.production × (EPISODE_END - capture_time)`.

    Sources or targets with no feasible pair are absent from the
    output (no spurious assignments).

    Uses scipy.optimize.linear_sum_assignment when available;
    falls back to a pure-Python greedy otherwise.
    """
    if not sources or not targets:
        return {}

    cost = _build_cost_matrix(sources, targets, capture_times)

    if _SCIPY_AVAILABLE:
        # scipy expects a numpy array. Local import to keep scipy as
        # the optional path.
        import numpy as np
        cost_arr = np.array(cost, dtype=float)
        row_ind, col_ind = _hungarian(cost_arr)
    else:
        row_ind, col_ind = _greedy_assignment(cost)

    assignment: dict = {}
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= INFEASIBLE_COST:
            continue
        sid = sources[i][0]
        tid = targets[j][0]
        assignment[int(sid)] = int(tid)
    return assignment


def compute_lp_assignment(world, model, me: int) -> dict:
    """Convenience wrapper: build matrix and solve assignment in one call.

    Returns `{src_id: tgt_id}` per `solve_assignment`. Used by
    `chooser_layered` to inform per-candidate LP-bias decisions.
    """
    sources, targets, matrix = build_capture_matrix(world, model, int(me))
    return solve_assignment(sources, targets, matrix)
