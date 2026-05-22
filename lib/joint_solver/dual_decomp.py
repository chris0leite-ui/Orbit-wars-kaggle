"""Lagrangian dual decomposition of the joint LP.

ITEM 5 of `composed-noodling-riddle.md`. Replaces `solve_outcome_aware`'s
MILP inner with closed-form per-source water-fill + per-target subset
enumeration + 3-iter subgradient on the source-budget multipliers.

Why: MILP inner at ~300ms/turn is the wallclock bottleneck. Phase ε.2
(full-K maximin × Stackelberg outer) needs ≤50ms inner to fit. Dual
decomposition of constraint (C3) — `Σ ships·x ≤ R + P·u` per
(source, time) — gives:

  L(x, y, λ) = Σ_p Σ_S [V_p(S) − Σ_{c∈S}(κ + λ̄_c)·n_c] · y_{p,S}
             + Σ_{s,u} λ_{s,u}·(R_s + P_s·u)

where λ̄_c = Σ_{u≥w_c} λ_{s(c),u} is the accumulated per-column rent.

Inner loop, per iteration:
  - Per-target argmax over subsets: choose S*_p maximizing
    V_p(S) − Σ_{c∈S}(κ + λ̄_c)·n_c. Closed-form over the existing
    enumerate_outcomes table.
  - Compute used_{s,u} = Σ_{c ∈ ∪ S*_p, w_c ≤ u} n_c. Where any (s, u)
    over-spends, increase λ_{s,u} via subgradient step. Under-spent
    cells decay toward 0.
  - 3 iterations max. Warm-start λ from previous turn (game state
    changes slowly).

Final feasibility fix-up: if last iteration still has violations,
drop the column from the over-spent source whose removal causes the
smallest objective loss. Repeat until feasible.

Gate: opt-in via `LP_SOLVER=dual` env var. Default `milp` keeps
existing path. Both coexist; the dispatcher in `lp_outcome.py`
selects between them at the top of `solve_outcome_aware`.

Parity gate (tests/test_dual_decomp_parity.py):
  - Median objective gap ≤ 3% on 8 seeds.
  - p95 gap ≤ 5%.
  - max single-turn objective Δ ≤ 0.5 × ALPHA_OPP_PENALTY.
  - Wallclock p95 ≤ 50 ms on a 30-planet board.
  - Determinism: same seed → byte-identical moves.
"""
from __future__ import annotations

import os
import time
from typing import Any

from lib.joint_solver.columns import Column
from lib.joint_solver.outcome_table import (
    MAX_ENUMERATION_BITS,
    Arrival,
    OutcomeRow,
    enumerate_outcomes,
)


DEFAULT_MAX_ITER = 3
DEFAULT_STEP_SIZE = 1.0      # base step for subgradient update
DEFAULT_WARM_DECAY = 0.7     # multiplicative decay on cross-turn warm-start


# Module-level warm-start cache, keyed by my_id. Decayed by
# DEFAULT_WARM_DECAY each turn (next-turn budgets are similar but not
# identical to this-turn's).
_LAMBDA_WARM_START: dict[int, dict[tuple[int, int], float]] = {}


def _dual_solver_enabled() -> bool:
    """Read env at call time so tests / A/B harnesses toggle cleanly."""
    return os.environ.get("LP_SOLVER", "milp").strip().lower() == "dual"


def _accumulated_rent(col: Column, lam: dict[tuple[int, int], float],
                      max_wait: int) -> float:
    """λ̄_c = Σ_{u ≥ w_c} λ_{s(c), u}.

    Each column's rent is the sum of all time-cell multipliers it
    participates in (its budget constraint binds at u ≥ w_c).
    """
    sid = int(col.src_id)
    w = int(col.wait_N)
    total = 0.0
    for u in range(w, int(max_wait) + 1):
        v = lam.get((sid, u))
        if v is not None and v > 0.0:
            total += v
    return total


def _max_wait_n(active_columns: list[Column]) -> int:
    if not active_columns:
        return 0
    return max(int(c.wait_N) for c in active_columns)


def _per_planet_argmax(
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    col_by_id: dict[int, Column],
    lam: dict[tuple[int, int], float],
    max_wait: int,
    ship_cost: float,
    *,
    base_value_fn,
) -> tuple[dict[int, tuple[int, ...]], dict[int, float]]:
    """For each target, pick the subset maximizing (V_p(S) − rent).

    `base_value_fn(pid, row)` returns the rent-FREE per-(planet, subset)
    value (V_p + endgame + topology). Subtract per-column rent here.
    Returns (chosen, per_planet_value) where per_planet_value is the
    rent-free value at the chosen subset (used for diagnostics).
    """
    chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    for pid, table in per_planet_tables.items():
        best_subset: tuple[int, ...] = ()
        best_val = float("-inf")
        best_base = 0.0
        for subset, row in table.items():
            base = float(base_value_fn(pid, row))
            # Rent contribution: Σ_{c ∈ subset}(κ + λ̄_c)·n_c.
            rent = 0.0
            for col_id in subset:
                col = col_by_id.get(int(col_id))
                if col is None:
                    continue
                lambda_bar = _accumulated_rent(col, lam, max_wait)
                rent += (float(ship_cost) + lambda_bar) * float(col.ships)
            v = base - rent
            if v > best_val:
                best_val = v
                best_subset = subset
                best_base = base
        chosen[pid] = best_subset
        per_planet_value[pid] = best_base
    return chosen, per_planet_value


def _check_feasibility(
    chosen: dict[int, tuple[int, ...]],
    col_by_id: dict[int, Column],
    inv: dict[int, tuple[int, int]],
    max_wait: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Compute used_{s,u} (cumulative ships from each source through
    fire-time u) and budget_{s,u} = R_s + P_s · u. Returns both dicts;
    caller computes violations as `used − budget`."""
    used: dict[tuple[int, int], int] = {}
    for subset in chosen.values():
        for col_id in subset:
            col = col_by_id.get(int(col_id))
            if col is None:
                continue
            sid = int(col.src_id)
            w = int(col.wait_N)
            for u in range(w, int(max_wait) + 1):
                used[(sid, u)] = used.get((sid, u), 0) + int(col.ships)
    budget: dict[tuple[int, int], int] = {}
    for sid, (initial, prod) in inv.items():
        for u in range(int(max_wait) + 1):
            budget[(sid, u)] = int(initial) + int(prod) * int(u)
    return used, budget


def _subgradient_update(
    lam: dict[tuple[int, int], float],
    used: dict[tuple[int, int], int],
    budget: dict[tuple[int, int], int],
    inv: dict[int, tuple[int, int]],
    iteration: int,
    step_base: float = DEFAULT_STEP_SIZE,
) -> tuple[dict[tuple[int, int], float], int]:
    """In-place subgradient update on λ. Returns (new_lam, max_violation).

    For each (s, u): violation = used − budget.
      - If violation > 0: λ += step · violation / source_pool. Diminishing
        step `step_base / (iteration + 2)`.
      - Else: λ decays toward 0 (no need to penalize feasible cells).

    Normalizing the step by source pool (sum of column ships at this
    source) keeps λ magnitudes bounded across very-different-scale
    sources (e.g., a 100-ship source vs a 5-ship source).
    """
    new_lam = dict(lam)
    step = float(step_base) / float(int(iteration) + 2)
    max_violation = 0
    # Source pool size (ship count cap) for normalization.
    src_pool = {sid: max(1, initial + prod * 5) for sid, (initial, prod) in inv.items()}
    keys = set(used.keys()) | set(budget.keys()) | set(lam.keys())
    for sid, u in keys:
        u_amt = int(used.get((sid, u), 0))
        b = int(budget.get((sid, u), 0))
        violation = u_amt - b
        cur = float(new_lam.get((sid, u), 0.0))
        if violation > 0:
            delta = step * (float(violation) / float(src_pool.get(sid, 1)))
            new_lam[(sid, u)] = max(0.0, cur + delta)
            if violation > max_violation:
                max_violation = violation
        else:
            # Decay under-utilized cells.
            new_lam[(sid, u)] = max(0.0, cur * 0.5)
    # Prune zeros to keep dict small.
    new_lam = {k: v for k, v in new_lam.items() if v > 1e-9}
    return new_lam, max_violation


def _feasibility_fixup(
    chosen: dict[int, tuple[int, ...]],
    col_by_id: dict[int, Column],
    inv: dict[int, tuple[int, int]],
    max_wait: int,
    base_value_fn,
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    ship_cost: float,
) -> dict[int, tuple[int, ...]]:
    """If `chosen` is infeasible, drop the column whose removal causes
    the smallest objective loss until feasible.

    For each infeasible (s, u), find the column from source s with
    `w_c ≤ u` in `chosen[p(c)]` whose removal reduces objective the
    least. Remove it: replace chosen[p(c)] with the same subset MINUS
    that column. Recheck feasibility. Repeat.
    """
    iters = 0
    max_fix_iters = 50
    while iters < max_fix_iters:
        used, budget = _check_feasibility(chosen, col_by_id, inv, max_wait)
        worst_violation = 0
        worst_cell: tuple[int, int] | None = None
        for k, u_amt in used.items():
            v = int(u_amt) - int(budget.get(k, 0))
            if v > worst_violation:
                worst_violation = v
                worst_cell = k
        if worst_cell is None or worst_violation <= 0:
            return chosen
        sid, u_lim = worst_cell
        # Candidates: columns currently in chosen, from source sid, with
        # w_c ≤ u_lim. Each is removable.
        candidates: list[tuple[int, int, int, int]] = []
        # (col_id, planet_id, ships, marginal_loss)
        for pid, subset in chosen.items():
            for col_id in subset:
                col = col_by_id.get(int(col_id))
                if col is None:
                    continue
                if int(col.src_id) != int(sid):
                    continue
                if int(col.wait_N) > int(u_lim):
                    continue
                # Marginal loss = base value of current subset − base value
                # of subset minus this column.
                table = per_planet_tables.get(int(pid))
                if table is None:
                    continue
                cur_row = table.get(tuple(subset))
                new_subset = tuple(x for x in subset if x != int(col_id))
                new_row = table.get(new_subset)
                if cur_row is None or new_row is None:
                    continue
                loss = float(base_value_fn(pid, cur_row)) - float(base_value_fn(pid, new_row))
                candidates.append((int(col_id), int(pid), int(col.ships), loss))
        if not candidates:
            # No removable column at the worst cell; should not happen.
            return chosen
        # Drop the one with smallest loss (least valuable to keep).
        candidates.sort(key=lambda t: t[3])
        drop_col_id, drop_pid, _drop_ships, _drop_loss = candidates[0]
        chosen[drop_pid] = tuple(x for x in chosen[drop_pid] if x != drop_col_id)
        iters += 1
    return chosen


def solve_dual_decomp_inner(
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    active_columns: list[Column],
    inv: dict[int, tuple[int, int]],
    *,
    my_id: int,
    base_value_fn,
    ship_cost: float = 1.0,
    max_iter: int = DEFAULT_MAX_ITER,
    time_limit_seconds: float = 0.05,
) -> tuple[dict[int, tuple[int, ...]], dict[int, float], str]:
    """Inner Lagrangian solver — replaces the MILP-via-scipy step in
    `solve_outcome_aware`.

    Returns (per_planet_chosen, per_planet_value, status).

    Reuses outputs from `solve_outcome_aware`'s setup:
      - per_planet_tables: per-target {subset → OutcomeRow}.
      - active_columns: filtered column list.
      - inv: per-source (initial_ships, production).
      - base_value_fn(pid, row): closure capturing my_id, world,
        topology_scores, opp_id, currently_winning — returns the
        rent-FREE per-(planet, subset) value (V_p + endgame + topology).

    Algorithm:
      1. Warm-start λ from previous turn (decayed) or zero.
      2. For up to max_iter iterations:
         - Per-target argmax of (V_p − rent).
         - Compute used vs budget per (source, time).
         - If feasible, break.
         - Else subgradient update.
      3. Feasibility fix-up: greedy column drops until feasible.

    `status` string encodes iteration count + max violation for diagnostics.
    """
    t0 = time.perf_counter()
    col_by_id = {int(c.column_id): c for c in active_columns}
    max_wait = _max_wait_n(active_columns)

    # Warm-start λ from previous turn (decayed). Keyed by my_id.
    prev = _LAMBDA_WARM_START.get(int(my_id), {})
    lam: dict[tuple[int, int], float] = {
        k: v * DEFAULT_WARM_DECAY for k, v in prev.items() if v > 1e-9
    }

    chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    n_iters = 0
    final_violation = 0
    for it in range(int(max_iter)):
        n_iters += 1
        chosen, per_planet_value = _per_planet_argmax(
            per_planet_tables, col_by_id, lam, max_wait,
            ship_cost=ship_cost, base_value_fn=base_value_fn,
        )
        used, budget = _check_feasibility(chosen, col_by_id, inv, max_wait)
        max_v = 0
        for k, u_amt in used.items():
            v = int(u_amt) - int(budget.get(k, 0))
            if v > max_v:
                max_v = v
        final_violation = max_v
        if max_v <= 0:
            break
        # Subgradient update.
        lam, _ = _subgradient_update(lam, used, budget, inv, it)
        if time.perf_counter() - t0 > float(time_limit_seconds):
            break

    # Save warm-start for next turn.
    _LAMBDA_WARM_START[int(my_id)] = lam

    # Final feasibility fix-up if needed.
    if final_violation > 0:
        chosen = _feasibility_fixup(
            chosen, col_by_id, inv, max_wait, base_value_fn,
            per_planet_tables, ship_cost,
        )
        # Re-derive per_planet_value at the new subsets.
        for pid, subset in chosen.items():
            row = per_planet_tables[pid].get(subset)
            if row is not None:
                per_planet_value[pid] = float(base_value_fn(pid, row))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    status = (
        f"dual_decomp:iter={n_iters},max_v={final_violation},"
        f"elapsed_ms={elapsed_ms:.1f}"
    )
    return chosen, per_planet_value, status


def clear_warm_start() -> None:
    """Reset the per-my_id λ warm-start cache. Tests + per-game boundary
    fingerprint reset call this."""
    _LAMBDA_WARM_START.clear()
