"""Phase 5C: outcome-table-aware LP for post-opening turns.

The Phase 4 LP scored each candidate column with a scalar value (W1/W2
lower bounds in isolation). This makes the LP unable to express the
GLOBAL property "if I commit ships to offense here, am I leaving home
undefended?" — the root cause of the steps 70-100 collapse documented
in audit/2026-05-20-phase5b-root-cause-analysis.md.

This module replaces per-candidate scoring with per-planet-subset
scoring. For each planet `p` with candidate arrivals targeting it,
we enumerate all 2^k subsets via Phase 1's outcome_table, compute the
production stream per owner for each subset, and let the LP pick
EXACTLY ONE subset per planet via auxiliary binary variables y_{p,S}.

The objective rewards `prod_stream_me(p, S) - α · prod_stream_opp(p, S)`,
so DEFENSE EMERGES FROM THE MATH: a planet under heavy opp threat has
large `prod_stream_opp(empty)` (opp captures it and produces); firing
defenders shifts the subset choice to one where `prod_stream_opp` is
smaller, raising objective value. No separate defensive-value
multiplier or W2 mid-bound hack.

Per-source ship budget over time is preserved from Phase 4 (still a
linear constraint on x variables).

MILP via scipy.optimize.milp (HiGHS); greedy fallback if unavailable
or infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _MILP_AVAILABLE = True
except ImportError:
    _MILP_AVAILABLE = False
    milp = None  # type: ignore[assignment]
    LinearConstraint = None  # type: ignore[assignment]
    Bounds = None  # type: ignore[assignment]

from lib.joint_solver.columns import Column
from lib.joint_solver.outcome_table import MAX_ENUMERATION_BITS, Arrival, OutcomeRow, enumerate_outcomes


# ---------------------------------------------------------------------------
# Constants (initial values; tunable via introspect)
# ---------------------------------------------------------------------------

# Game horizon. PRINCIPLED VALUE: 500. The Orbit Wars episode ends at step 500
# (per the env: `lib/fast_sim.DEFAULT_CONFIG['episodeSteps']`); production
# accrued through that tick is real game-end ship count. Earlier values
# (T_END=200) were arbitrary clipping to keep "forecasts speculative" —
# the right answer is to model the actual game.
T_END = 500
# Weight on opp production in the objective.
# PRINCIPLED VALUE: 1.0. The game's win condition is `my_ships - opp_ships`
# at T_END. Opp's accumulated production directly subtracts from our margin.
# Anything less than 1.0 is arbitrary.
ALPHA_OPP_PENALTY = 1.0
# Ship cost coefficient in the objective. Phase 5G (2026-05-20): bumped
# from 0.01 to 1.0 after the critique diagnosis: enumerate_ship_counts
# in agents/baseline/proposer.py emits 3 ship-count variants per (src,
# tgt) — [capture_size, 2×capture_size, full_budget] — all with
# IDENTICAL cheap_delta (per-candidate value doesn't depend on ship
# count for a successful capture) AND identical outcome_table value
# (same prod_stream regardless of ships). At SHIP_COST=0.01, the
# penalty (42 ships × 0.01 = 0.42) is dwarfed by value (~180), and
# the LP picks the LARGEST variant by tie-break. Result: each source
# drains in ONE fire then is idle for many turns. SHIP_COST=1.0
# makes the per-ship penalty meaningful: 42-ship variant costs 42 vs
# 9-ship variant costs 9, breaking the tie toward efficient launches.
SHIP_COST = 1.0
MAX_CONTESTERS_PER_PLANET = MAX_ENUMERATION_BITS  # 2^6 = 64 subsets per planet
TIME_LIMIT_SECONDS = 0.3          # MILP wallclock cap
# PRINCIPLED VALUE: 0. The LP's objective already penalizes losing a planet
# (opp's prod_stream gets credit when ownership flips); over-draining a
# source is naturally bad math, no need for an arbitrary reservation.
DEFENDER_GUARD = 0


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OutcomeAwareResult:
    """Output of solve_outcome_aware."""
    moves: list                          # [src_id, angle, ships] for wait_N==0 fires
    fired_columns: list[Column]          # all selected columns (any wait_N)
    objective: float                     # achieved objective value
    status: str                          # solver status string
    n_x_vars: int                        # number of candidate (x) variables
    n_y_vars: int                        # number of subset (y) variables
    n_constraints: int                   # number of constraint rows
    per_planet_chosen: dict[int, tuple[int, ...]] = field(default_factory=dict)
    per_planet_value: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_inventory(columns: list[Column], world, *, my_id: int
                      ) -> dict[int, tuple[int, int]]:
    """For each our-source-id present in columns, return (initial_ships, production)."""
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


def _build_per_planet_arrivals(
    active_columns: list[Column],
    world,
    model,
    *,
    my_id: int,
    step_now: int,
) -> dict[int, tuple[list[Arrival], list[Arrival]]]:
    """For each planet that has ≥1 candidate column targeting it, build the
    (fixed_arrivals, candidate_arrivals) pair for outcome_table.

    Phase 5D: fixed arrivals come ONLY from `model.ledger[p.id]`. The
    upstream `_model_with_opp_projection` already merges opp's projected
    counter-launches into the ledger; reading them a second time from a
    separate `opp_arrivals` parameter (as Phase 5C did) double-counted
    every opp arrival.

    Candidate arrivals = active columns with tgt_id == p.id.

    If a planet has more than MAX_CONTESTERS_PER_PLANET candidates, keep
    the top ones by per-candidate value (in Column.value) as a pre-filter.
    Excess candidates are dropped from the LP entirely.
    """
    # Group columns by target.
    by_tgt: dict[int, list[Column]] = {}
    for col in active_columns:
        by_tgt.setdefault(int(col.tgt_id), []).append(col)

    # Bug #6: build the keep-set of column ids referenced as parents
    # by ANY column. The per-planet pre-filter would otherwise drop a
    # low-value parent and the linkage constraint at L495-518 would
    # force-zero every compound that pointed at it — silent action-space
    # shrinkage. Compute across all active columns (parents may live in
    # a different by_tgt group than their compounds).
    parent_keepset: set[int] = {
        int(getattr(c, "parent_column_id"))
        for c in active_columns
        if getattr(c, "parent_column_id", None) is not None
    }

    out: dict[int, tuple[list[Arrival], list[Arrival]]] = {}
    for tgt_pid, cols in by_tgt.items():
        # Cap at MAX_CONTESTERS_PER_PLANET via per-candidate value. The
        # keep-set is force-kept regardless of value rank.
        if len(cols) > MAX_CONTESTERS_PER_PLANET:
            forced = [c for c in cols if int(c.column_id) in parent_keepset]
            optional = [c for c in cols if int(c.column_id) not in parent_keepset]
            # Bug #7: secondary sort keys so prerank_passthrough's
            # uniform value=1.0 doesn't yield non-deterministic survivors.
            # Prefer higher ships, earlier launches (smaller wait_N),
            # smaller column_id (id-ascending for stable identity tie-break).
            optional.sort(
                key=lambda c: (
                    float(c.value),
                    int(c.ships),
                    -int(c.wait_N),
                    -int(c.column_id),
                ),
                reverse=True,
            )
            budget = max(0, MAX_CONTESTERS_PER_PLANET - len(forced))
            cols = forced + optional[:budget]

        # Fixed arrivals — ONLY from model.ledger (opp projections live there).
        fixed: list[Arrival] = []
        for eta_arr, owner, ships in model.ledger.get(int(tgt_pid), []):
            if int(ships) <= 0:
                continue
            fixed.append(Arrival(
                eta=int(eta_arr), owner=int(owner), ships=int(ships),
                column_id=None,
            ))

        # Candidate arrivals from our columns. Total arrival tick from the
        # planner's NOW perspective = wait_N + eta (flight time).
        cands: list[Arrival] = []
        for col in cols:
            total_eta = int(col.wait_N) + int(col.eta)
            cands.append(Arrival(
                eta=total_eta, owner=int(my_id), ships=int(col.ships),
                column_id=int(col.column_id),
            ))
        out[int(tgt_pid)] = (fixed, cands)
    return out


def _value_for_outcome(row: OutcomeRow, my_id: int,
                       alpha_opp_penalty: float,
                       discounted: bool = False) -> float:
    """Subset value: prod_stream_me − α · prod_stream_opp.

    Sums opp production across ALL non-me, non-neutral owners (4P-aware).

    `discounted`: when True, reads `row.prod_stream_discounted` (the
    γ-weighted per-tick production accrual) instead of the integer
    `prod_stream`. Caller must have constructed the row via
    `enumerate_outcomes(..., discount_gamma=γ)` so the discounted dict
    is populated.
    """
    if discounted:
        me_prod = float(row.prod_stream_discounted.get(int(my_id), 0.0))
        opp_prod = float(sum(
            v for owner, v in row.prod_stream_discounted.items()
            if int(owner) >= 0 and int(owner) != int(my_id)
        ))
        return me_prod - float(alpha_opp_penalty) * opp_prod
    me_prod = float(row.prod_stream.get(int(my_id), 0))
    opp_prod = float(sum(
        v for owner, v in row.prod_stream.items()
        if int(owner) != int(my_id) and int(owner) >= 0
    ))
    return me_prod - float(alpha_opp_penalty) * opp_prod


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------


def _greedy_fallback(
    active_columns: list[Column],
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    world,
    *,
    my_id: int,
    alpha_opp_penalty: float,
    discounted: bool = False,
) -> OutcomeAwareResult:
    """Pure-Python greedy fallback when MILP is unavailable / infeasible.

    For each planet, pick the subset with the highest
    (prod_stream_me − α · prod_stream_opp) value. Then check global
    source budget feasibility; if a launch would over-spend, drop it
    (greedy, lowest-value-marginal first).
    """
    step_now = int(getattr(world, "step", 0) or 0)
    inv = _source_inventory(active_columns, world, my_id=int(my_id))

    # Per planet: choose best subset.
    chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    for pid, table in per_planet_tables.items():
        best_subset = ()
        best_value = _value_for_outcome(table[()], my_id, alpha_opp_penalty, discounted)
        for subset, row in table.items():
            v = _value_for_outcome(row, my_id, alpha_opp_penalty, discounted)
            if v > best_value:
                best_value = v
                best_subset = subset
        chosen[pid] = best_subset
        per_planet_value[pid] = best_value

    # Collect fired column_ids from chosen subsets.
    fired_ids = {cid for s in chosen.values() for cid in s}
    by_col_id = {int(c.column_id): c for c in active_columns}

    # Check source budget; greedy drop launches if over-spent.
    emitted_per_src_fire: dict[tuple[int, int], int] = {}
    fired: list[Column] = []
    drop_order = sorted(
        (by_col_id[cid] for cid in fired_ids),
        key=lambda c: float(c.value),
    )
    for col in drop_order:
        sid = int(col.src_id)
        initial, prod = inv.get(sid, (0, 0))
        wait_N = int(col.wait_N)
        used = sum(v for (s, w), v in emitted_per_src_fire.items()
                   if s == sid and w <= wait_N)
        if used + int(col.ships) > initial + prod * max(0, wait_N) - DEFENDER_GUARD:
            # Drop this column from the chosen subset (replace with empty).
            for pid, s in chosen.items():
                if int(col.column_id) in s:
                    chosen[pid] = tuple(c for c in s if c != int(col.column_id))
            continue
        emitted_per_src_fire[(sid, wait_N)] = (
            emitted_per_src_fire.get((sid, wait_N), 0) + int(col.ships)
        )
        fired.append(col)

    # Compute final objective.
    obj = sum(
        _value_for_outcome(per_planet_tables[pid][s], my_id, alpha_opp_penalty,
                           discounted)
        for pid, s in chosen.items()
    )

    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired if int(c.wait_N) == 0
    ]
    return OutcomeAwareResult(
        moves=moves, fired_columns=fired, objective=float(obj),
        status="greedy_fallback",
        n_x_vars=len(active_columns),
        n_y_vars=sum(len(t) for t in per_planet_tables.values()),
        n_constraints=0,
        per_planet_chosen=chosen,
        per_planet_value=per_planet_value,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve_outcome_aware(
    columns: list[Column],
    world,
    model,
    *,
    my_id: int,
    t_end: int = T_END,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
    discount_gamma: float | None = None,
) -> OutcomeAwareResult:
    """Solve the outcome-aware LP for post-opening turns.

    Inputs:
      columns: list of Column with pre-computed per-candidate value (used
        only as a pre-filter when a planet has > MAX_CONTESTERS_PER_PLANET
        candidates).
      world, model: World and WorldModel snapshots. `model.ledger` is
        EXPECTED to already include opp projections (the upstream
        `_model_with_opp_projection` merges them in); the Phase 5C
        separate `opp_arrivals` parameter was removed in Phase 5D after
        the double-count audit.
      my_id: our seat.

    Returns OutcomeAwareResult with the chosen moves (wait_N==0 fires)
    plus diagnostics.
    """
    if not columns:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="empty_columns",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    step_now = int(getattr(world, "step", 0) or 0)

    # Filter to our positive-value columns with a valid source.
    # EXCEPTION: compound columns (parent_column_id != None) are
    # Phase F2a production-feedback fires from planets we'd capture
    # mid-horizon. Their src isn't yet in inv (it's opp-owned now).
    # They're feasible only if their parent capture fires; that's
    # enforced via a linkage constraint added below, not by inv.
    inv = _source_inventory(columns, world, my_id=int(my_id))
    active: list[Column] = []
    for col in columns:
        if int(col.owner) != int(my_id):
            continue
        is_compound = getattr(col, "parent_column_id", None) is not None
        if not is_compound and int(col.src_id) not in inv:
            continue
        if float(col.value) <= 0.0:
            continue
        active.append(col)
    if not active:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="no_positive_columns",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    # Build per-planet arrival sets and outcome tables.
    per_planet_arrivals = _build_per_planet_arrivals(
        active, world, model,
        my_id=int(my_id), step_now=step_now,
    )
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]] = {}
    for tgt_pid, (fixed, cands) in per_planet_arrivals.items():
        planet = world.planets_by_id.get(int(tgt_pid))
        if planet is None:
            continue
        try:
            table = enumerate_outcomes(
                initial_owner=int(planet.owner),
                initial_ships=float(int(planet.ships)),
                production=int(planet.production),
                horizon=int(t_end),
                fixed_arrivals=fixed,
                candidate_arrivals=cands,
                discount_gamma=discount_gamma,
            )
        except ValueError:
            continue  # too many candidates after pre-filter; shouldn't happen
        per_planet_tables[int(tgt_pid)] = table

    # The set of column_ids that actually made it into per-planet enumeration
    # (the pre-filter may have dropped some when k > MAX_CONTESTERS_PER_PLANET).
    enumerated_col_ids: set[int] = set()
    for table in per_planet_tables.values():
        for subset in table:
            enumerated_col_ids.update(subset)
    # Drop active columns whose column_id wasn't enumerated.
    active = [c for c in active if int(c.column_id) in enumerated_col_ids
              or int(c.tgt_id) not in per_planet_tables]
    # Actually, only columns whose target was enumerated count. If column.tgt
    # isn't in per_planet_tables, it was pruned. Drop them.
    active = [c for c in active if int(c.column_id) in enumerated_col_ids]
    if not active:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="no_active_after_prefilter",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    if not _MILP_AVAILABLE:
        return _greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
        )

    import numpy as np

    # ---- Build MILP -------------------------------------------------------
    n_x = len(active)
    col_id_to_x_idx: dict[int, int] = {int(c.column_id): j for j, c in enumerate(active)}

    # y variables: list per planet.
    y_index: dict[tuple[int, tuple[int, ...]], int] = {}
    y_planet_subsets: dict[int, list[tuple[int, ...]]] = {}
    for pid, table in per_planet_tables.items():
        subsets = list(table.keys())
        y_planet_subsets[pid] = subsets
        for s in subsets:
            y_index[(pid, s)] = n_x + len(y_index)
    n_y = len(y_index)
    n_total = n_x + n_y

    # Cost vector. milp minimizes c^T·x, so negate values.
    c_vec = np.zeros(n_total, dtype=float)
    for j, col in enumerate(active):
        c_vec[j] = float(ship_cost) * float(col.ships)  # ship-cost penalty
    use_discounted_value = (
        discount_gamma is not None and 0.0 < float(discount_gamma) < 1.0
    )
    for (pid, s), y_idx in y_index.items():
        row = per_planet_tables[pid][s]
        value = _value_for_outcome(row, my_id, alpha_opp_penalty,
                                   use_discounted_value)
        c_vec[y_idx] = -float(value)  # negate so milp picks high-value subsets

    A_eq_rows: list[list[float]] = []
    b_eq: list[float] = []
    A_ub_rows: list[list[float]] = []
    b_ub: list[float] = []

    # (1) Subset uniqueness per planet: Σ_S y_{p,S} = 1.
    for pid, subsets in y_planet_subsets.items():
        row = [0.0] * n_total
        for s in subsets:
            row[y_index[(pid, s)]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    # (2) Candidate↔subset linkage: x_c - Σ_{S∋c} y_{p(c),S} = 0.
    for j, col in enumerate(active):
        pid = int(col.tgt_id)
        row = [0.0] * n_total
        row[j] = 1.0
        for s in y_planet_subsets.get(pid, []):
            if int(col.column_id) in s:
                row[y_index[(pid, s)]] = -1.0
        A_eq_rows.append(row)
        b_eq.append(0.0)

    # (3) Source budget over time. Phase F2a: skip compound columns —
    # their ships come from a planet we'd capture mid-horizon, not from
    # any source in `inv`. Compound columns are gated via the linkage
    # constraint (4) below; the captured planet's post-capture
    # production is implicit in the column's ship-count construction.
    src_ids = sorted({int(c.src_id) for c in active
                      if getattr(c, "parent_column_id", None) is None})
    fire_times = sorted({int(c.wait_N) for c in active})
    for sid in src_ids:
        if sid not in inv:
            continue
        initial, prod = inv[sid]
        for u in fire_times:
            row = [0.0] * n_total
            any_in_row = False
            for j, col in enumerate(active):
                if getattr(col, "parent_column_id", None) is not None:
                    continue  # compound col not in src-budget
                if int(col.src_id) == sid and int(col.wait_N) <= u:
                    row[j] = float(col.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_ub_rows.append(row)
            b_ub.append(float(initial + prod * max(0, u) - DEFENDER_GUARD))

    # (4) Phase F2a linkage: x_compound <= x_parent_capture.
    # Encoded as A_ub row `+1 * x_compound − 1 * x_parent <= 0`.
    # parent_column_id may reference a column that got dropped at the
    # per-planet-MILP-prefilter step (lp_outcome.py:381). In that case
    # we instead pin x_compound = 0 (force the row to b_ub=0 with only
    # the +1 term — equivalent to x_compound <= 0).
    col_id_to_idx: dict[int, int] = {
        int(col.column_id): j for j, col in enumerate(active)
    }
    for j, col in enumerate(active):
        pid_parent = getattr(col, "parent_column_id", None)
        if pid_parent is None:
            continue
        row = [0.0] * n_total
        row[j] = 1.0
        parent_idx = col_id_to_idx.get(int(pid_parent))
        if parent_idx is None:
            # Parent dropped; force this compound col to 0.
            A_ub_rows.append(row)
            b_ub.append(0.0)
        else:
            row[parent_idx] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

    # Compose constraints.
    constraints_list = []
    if A_eq_rows:
        A_eq = np.array(A_eq_rows, dtype=float)
        b_eq_arr = np.array(b_eq, dtype=float)
        constraints_list.append(LinearConstraint(A_eq, lb=b_eq_arr, ub=b_eq_arr))
    if A_ub_rows:
        A_ub = np.array(A_ub_rows, dtype=float)
        b_ub_arr = np.array(b_ub, dtype=float)
        constraints_list.append(LinearConstraint(A_ub, ub=b_ub_arr))
    if not constraints_list:
        return _greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
        )

    bounds = Bounds(lb=np.zeros(n_total), ub=np.ones(n_total))
    integrality = np.ones(n_total, dtype=int)
    n_constraints = len(A_eq_rows) + len(A_ub_rows)

    try:
        res = milp(c=c_vec, constraints=constraints_list,
                   integrality=integrality, bounds=bounds,
                   options={"time_limit": float(time_limit_seconds)})
    except Exception:
        return _greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
        )

    if res.x is None:
        return _greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
        )

    # Extract.
    fired: list[Column] = []
    moves: list = []
    for j, col in enumerate(active):
        if res.x[j] > 0.5:
            fired.append(col)
            if int(col.wait_N) == 0:
                moves.append([int(col.src_id), float(col.angle), int(col.ships)])
    per_planet_chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    for (pid, s), y_idx in y_index.items():
        if res.x[y_idx] > 0.5:
            per_planet_chosen[pid] = s
            row = per_planet_tables[pid][s]
            per_planet_value[pid] = _value_for_outcome(
                row, my_id, alpha_opp_penalty, use_discounted_value,
            )

    return OutcomeAwareResult(
        moves=moves, fired_columns=fired, objective=float(-res.fun),
        status=str(getattr(res, "message", "milp_ok")),
        n_x_vars=int(n_x), n_y_vars=int(n_y), n_constraints=int(n_constraints),
        per_planet_chosen=per_planet_chosen,
        per_planet_value=per_planet_value,
    )
