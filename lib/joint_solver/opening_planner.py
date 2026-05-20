"""Opening planner — one-shot multi-turn MILP for the first ~30 ticks.

The Phase 3/4 per-turn LP plans wait_N>0 launches that MPC re-solves
away each turn (Phase 4 n=8 A/B: 0/16 vs trajectory baseline; emit/fire
ratio 0.23 in early game). The PI's reframe: the opening is a small,
deterministic, closed-form subgame — solve it once as a multi-turn
commit-and-execute schedule, not re-derive per turn.

Architecture (substrate replacement for steps 0 .. OPENING_HORIZON):
- For each (source, launch_tick, target) triple, pre-compute trajectory
  feasibility, capture-size at arrival, defensive feasibility.
- Binary MILP: x_{s,t,p} ∈ {0,1} per surviving triple. Objective rewards
  early capture of high-prod targets (`prod_p · (T_END − arrival_tick)`).
- Constraints: per-source ship budget over time; per-target gang-up cap.
- Solve via scipy.optimize.milp (HiGHS); pure-Python greedy fallback.

Stateless re-derivation each turn: at every solve_turn call, re-run the
planner on the current world. Deterministic state propagation +
lexicographic tie-break = stable schedule that emits only entries
with fire_step == step_now.

Once step >= OPENING_HORIZON, mpc.py falls through to the Phase 4 LP.
"""

from __future__ import annotations

import math
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

from agents.baseline.proposer import aim_and_eta
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


# ---------------------------------------------------------------------------
# Constants (tunable; initial values per the plan)
# ---------------------------------------------------------------------------

OPENING_HORIZON = 30        # planner active for steps 0..(OPENING_HORIZON-1)
T_END = 200                 # value horizon: prod·(T_END - arrival)
OPP_BONUS = 1.10            # multiplier for capturing opp planets (strips their prod)
HOLD_WINDOW = 12            # ticks of post-capture defense we require feasibility for
DEFENDER_GUARD = 2          # reserve at least this many ships on each source (subtracted ONCE from budget)
MIN_SOURCE_SHIPS = 3        # skip sources with fewer ships (newly captured planets fire sooner)
MAX_CONTESTERS_PER_TARGET = 1  # opening: each target captured at most once (avoid wasteful gang-ups)
TOP_PAIRS_PER_SOURCE = 12
TOP_TARGETS_PER_SOURCE = 8  # K in "top-K targets by prod/(dist+1)"
STRIDE = 1                  # launch-tick stride (must-include t=step_now)
ROI_THRESHOLD = 0.5         # accept launches with value ≥ ROI_THRESHOLD × ships invested


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled launch. `fire_step` is ABSOLUTE (not relative to step_now)."""
    fire_step: int
    src_id: int
    tgt_id: int
    ships: int
    angle: float
    eta: int
    value: float


@dataclass
class OpeningPlan:
    """Output of `plan()`."""
    schedule: list[ScheduleEntry]
    objective: float
    n_vars: int
    n_constraints: int
    status: str
    pruning_waterfall: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal candidate (one pre-pruned MILP variable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    column_id: int
    src_id: int
    tgt_id: int
    fire_step: int     # absolute step in the env
    eta: int           # ticks of flight from fire_step
    arrival: int       # fire_step + eta (absolute step)
    ships: int         # capture size required at arrival
    angle: float
    value: float       # objective coefficient
    src_idx: int       # row index in source-budget constraints
    tgt_idx: int       # row index in target-cap constraints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _read_obs_fleets(world):
    raw = world.obs_raw
    if isinstance(raw, dict):
        return raw.get("fleets", []) or []
    return getattr(raw, "fleets", []) or []


def _nearest_enemy(world, tgt, me):
    """Return the enemy planet (not me, not neutral) closest to `tgt`, or None."""
    best = None
    best_d = float("inf")
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        if owner == int(me) or owner < 0:
            continue
        d = _dist(p, tgt)
        if d < best_d:
            best_d = d
            best = p
    return best


def _ships_to_capture(tgt, owner_at_arrival: int, garrison_at_arrival: float,
                      my_id: int) -> int:
    """Closed-form ship count needed to capture `tgt` at arrival.
    If `tgt` is already mine at arrival, return 0 (reinforce = no-op for opening).
    """
    if owner_at_arrival == int(my_id):
        return 0
    return max(1, int(math.ceil(garrison_at_arrival)) + 1)


def _expected_hold_duration(tgt, arrival: int, capture_residual: int,
                            world, model, my_id: int) -> int:
    """Closed-form expected hold duration in ticks AFTER capture.

    Decision logic, empirically tuned:
      - If opp arrives BEFORE us (opp_threat_eta < arrival): hold = 0
        (we won't take this planet at all; opp gets there first).
      - If opp's earliest arrival ≥ arrival + HOLD_WINDOW: we got there
        with margin; assume opp focuses elsewhere → hold = T_END - arrival
        (full game-end credit). This is the typical opening case for
        targets closer to us than to opp.
      - Else (tight race, opp could arrive soon): scale the hold by 3×
        the minimum-arrival delta to reflect that opp won't typically
        spend their first action attacking our outpost.

    The 3× safety factor matches the production-vs-opp-source compare:
    even if opp arrives at opp_threat, our garrison has had time to
    grow, and our reinforcements have time to land.
    """
    try:
        opp_threat_eta = model.time_to_enemy_threat(int(tgt.id), int(my_id), world)
    except Exception:
        opp_threat_eta = None
    if opp_threat_eta is None:
        return max(0, T_END - arrival)
    delta = int(opp_threat_eta) - arrival
    if delta <= 0:
        return 0
    if delta >= HOLD_WINDOW:
        return max(0, T_END - arrival)
    return min(max(0, T_END - arrival), 3 * delta)


def _is_minimally_holdable(tgt, arrival: int, capture_residual: int,
                           world, model, my_id: int) -> bool:
    """Lower-bound feasibility: did we get to the planet first AND survive
    immediate counter? If opp arrives before us, the capture is futile.

    Anything beyond this is handled by the value-weighting in
    `_expected_hold_duration`, so the MILP picks captures with the best
    production-over-hold-window."""
    try:
        opp_threat_eta = model.time_to_enemy_threat(int(tgt.id), int(my_id), world)
    except Exception:
        opp_threat_eta = None
    if opp_threat_eta is None:
        return True
    return int(opp_threat_eta) > arrival


# ---------------------------------------------------------------------------
# Candidate generation (the prune chain)
# ---------------------------------------------------------------------------


def _build_candidates(world, model, my_id: int, num_seats: int,
                      ) -> tuple[list[_Candidate], dict[str, int]]:
    """Apply the 6-step prune chain. Return (candidates, waterfall_stats)."""
    waterfall = {"naive_upper_bound": 0, "after_source_pool": 0,
                 "after_top_targets": 0, "after_reachability": 0,
                 "after_trajectory": 0, "after_feasibility": 0,
                 "after_top_pairs": 0}

    step_now = int(world.step)
    omega = float(world.omega)

    # 1. Source pool — my planets with at least MIN_SOURCE_SHIPS ships.
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(my_id)
                  and int(p.ships) >= MIN_SOURCE_SHIPS]
    waterfall["after_source_pool"] = len(my_planets)
    if not my_planets:
        return [], waterfall

    # All non-mine, non-comet planets are potential targets.
    comet_ids = set(world.comet_ids) if world.comet_ids else set()
    all_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != int(my_id) and int(p.id) not in comet_ids]

    waterfall["naive_upper_bound"] = (
        len(my_planets) * len(all_targets) * ((OPENING_HORIZON // STRIDE) + 1)
    )

    src_ids_in_use: set[int] = set()
    tgt_ids_in_use: set[int] = set()
    all_candidates: list[_Candidate] = []
    next_id = 0

    fire_offsets = [0] + list(range(STRIDE, OPENING_HORIZON, STRIDE))

    for src in my_planets:
        # 2. Per-source top-K targets by prod / (dist + 1).
        scored_targets = sorted(
            ((float(t.production) / (_dist(src, t) + 1.0), t) for t in all_targets),
            key=lambda x: x[0], reverse=True,
        )
        top_targets = [t for _s, t in scored_targets[:TOP_TARGETS_PER_SOURCE]]
        waterfall["after_top_targets"] += len(top_targets)

        # 3+4+5: reachability × trajectory × stride-2 fire ticks.
        per_src_pruned: list[_Candidate] = []
        for tgt in top_targets:
            for offset in fire_offsets:
                fire_step = step_now + offset
                # Initial ship estimate for aim_and_eta — refine via fixed point.
                ships_est = max(DEFENDER_GUARD, int(tgt.ships) + 1)
                # Two-step refinement is enough (eta converges fast).
                for _ in range(2):
                    res = aim_and_eta(src, tgt, ships_est, omega, wait_N=offset)
                    if res is None:
                        break
                    angle, eta_flight = res
                    if eta_flight is None or eta_flight <= 0 or eta_flight > OPENING_HORIZON + 10:
                        res = None
                        break
                    arrival_total = offset + int(eta_flight)
                    # Predict garrison at arrival via WorldModel (closed-form).
                    # Use the ledger as-is; opp counter-projection lives in (C3).
                    base_arrivals = list(model.ledger.get(int(tgt.id), []))
                    try:
                        owner_at_arr, gar_at_arr = predict_garrison_at(
                            tgt, arrival_total, base_arrivals,
                        )
                    except Exception:
                        res = None
                        break
                    needed = _ships_to_capture(tgt, int(owner_at_arr), float(gar_at_arr), my_id)
                    if needed <= 0:
                        res = None
                        break
                    if needed == ships_est:
                        break
                    ships_est = needed
                if res is None:
                    continue
                angle, eta_flight = res
                arrival_total = offset + int(eta_flight)
                # Recompute garrison with final ships estimate (no opp projection here).
                base_arrivals = list(model.ledger.get(int(tgt.id), []))
                owner_at_arr, gar_at_arr = predict_garrison_at(
                    tgt, arrival_total, base_arrivals,
                )
                needed = _ships_to_capture(tgt, int(owner_at_arr), float(gar_at_arr), my_id)
                if needed <= 0:
                    continue
                # Source budget at fire tick (post-production, pre-launch).
                src_ships_at_fire = int(src.ships) + int(src.production) * offset
                if needed + DEFENDER_GUARD > src_ships_at_fire:
                    continue  # can't afford while keeping defender

                # 4. Trajectory feasibility (only meaningful at offset=0 since
                # predict_fleet_fate uses current positions; conservative for
                # offset>0 — we accept the approximation in the opening window).
                if offset == 0:
                    try:
                        fate = predict_fleet_fate(src, tgt, angle, needed, world)
                    except Exception:
                        fate = None
                    if fate is not None and getattr(fate, "outcome", "") != "target":
                        waterfall.setdefault("dropped_trajectory", 0)
                        waterfall["dropped_trajectory"] += 1
                        continue

                capture_residual = needed - int(math.ceil(gar_at_arr))
                if capture_residual < 1:
                    continue

                # Value = production × hold_window × opp_bonus, where
                # hold_window is the expected ticks we hold post-capture
                # before opp could plausibly recapture. Indefensible
                # captures get hold_window=0 → value=0 → naturally rejected
                # by the ROI gate below. Contested captures with positive
                # hold get a fair-but-not-inflated value.
                hold_dur = _expected_hold_duration(
                    tgt, arrival_total, capture_residual, world, model, my_id,
                )
                if hold_dur <= 0:
                    waterfall.setdefault("dropped_defense", 0)
                    waterfall["dropped_defense"] += 1
                    continue
                opp_bonus = OPP_BONUS if int(tgt.owner) != -1 else 1.0
                value = float(int(tgt.production)) * float(hold_dur) * float(opp_bonus)
                # Per-launch ROI filter — gentler than 1:1 to match the
                # baseline's aggressive opening throughput. Even half-ROI
                # captures contribute to the production base once held.
                if value < ROI_THRESHOLD * float(needed):
                    waterfall.setdefault("dropped_low_roi", 0)
                    waterfall["dropped_low_roi"] += 1
                    continue

                src_ids_in_use.add(int(src.id))
                tgt_ids_in_use.add(int(tgt.id))
                per_src_pruned.append(_Candidate(
                    column_id=next_id, src_id=int(src.id), tgt_id=int(tgt.id),
                    fire_step=fire_step, eta=int(eta_flight), arrival=step_now + arrival_total,
                    ships=int(needed), angle=float(angle), value=float(value),
                    src_idx=-1, tgt_idx=-1,  # filled in below
                ))
                next_id += 1

        # 6. Per-source cap: ensure target diversity FIRST, then trim.
        # Group by (src, tgt); within each group keep the top 3 fire_steps
        # by raw value. This prevents a single high-ROI tiny capture from
        # crowding out larger valuable captures at other targets.
        by_tgt: dict[int, list[_Candidate]] = {}
        for c in per_src_pruned:
            by_tgt.setdefault(int(c.tgt_id), []).append(c)
        diverse: list[_Candidate] = []
        for tid, group in by_tgt.items():
            group.sort(key=lambda c: c.value, reverse=True)
            diverse.extend(group[:3])
        # Now take top TOP_PAIRS_PER_SOURCE by raw value across this source's
        # diverse candidates.
        diverse.sort(key=lambda c: c.value, reverse=True)
        keep = diverse[:TOP_PAIRS_PER_SOURCE]
        all_candidates.extend(keep)

    waterfall["after_top_pairs"] = len(all_candidates)
    waterfall["after_reachability"] = len(all_candidates)  # rolled together
    waterfall["after_trajectory"] = len(all_candidates)
    waterfall["after_feasibility"] = len(all_candidates)

    # Renumber column_ids contiguously and fill in src/tgt indexes.
    src_idx_map = {sid: i for i, sid in enumerate(sorted(src_ids_in_use))}
    tgt_idx_map = {tid: i for i, tid in enumerate(sorted(tgt_ids_in_use))}
    out: list[_Candidate] = []
    for new_id, c in enumerate(all_candidates):
        out.append(_Candidate(
            column_id=new_id, src_id=c.src_id, tgt_id=c.tgt_id,
            fire_step=c.fire_step, eta=c.eta, arrival=c.arrival,
            ships=c.ships, angle=c.angle, value=c.value,
            src_idx=src_idx_map[c.src_id], tgt_idx=tgt_idx_map[c.tgt_id],
        ))
    return out, waterfall


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------


def _greedy_fallback(candidates: list[_Candidate], world, my_id: int,
                     ) -> tuple[list[_Candidate], float]:
    """Pure-Python descending-value greedy with budget + gang-up tracking."""
    step_now = int(world.step)
    # Per-source remaining ship pool, indexed by (src_id, fire_offset).
    src_inv: dict[int, tuple[int, int]] = {}  # src_id -> (initial_ships, production)
    for c in candidates:
        if c.src_id in src_inv:
            continue
        src_p = world.planets_by_id.get(c.src_id)
        if src_p is not None:
            src_inv[c.src_id] = (int(src_p.ships), int(src_p.production))

    emitted_by_src_fire: dict[tuple[int, int], int] = {}  # (src, fire_step) -> ships used
    tgt_count: dict[int, int] = {}
    chosen: list[_Candidate] = []
    obj = 0.0

    for c in sorted(candidates, key=lambda x: x.value, reverse=True):
        if tgt_count.get(c.tgt_id, 0) >= MAX_CONTESTERS_PER_TARGET:
            continue
        # Source budget check: cumulative emissions up to c.fire_step ≤ available.
        initial, prod = src_inv.get(c.src_id, (0, 0))
        offset = c.fire_step - step_now
        used = sum(v for (s, fs), v in emitted_by_src_fire.items()
                   if s == c.src_id and fs <= c.fire_step)
        if used + c.ships > initial + prod * max(0, offset) - DEFENDER_GUARD:
            continue
        chosen.append(c)
        emitted_by_src_fire[(c.src_id, c.fire_step)] = (
            emitted_by_src_fire.get((c.src_id, c.fire_step), 0) + c.ships
        )
        tgt_count[c.tgt_id] = tgt_count.get(c.tgt_id, 0) + 1
        obj += c.value
    return chosen, obj


# ---------------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------------


def _solve_milp(candidates: list[_Candidate], world, my_id: int,
                time_limit_seconds: float):
    """Run the MILP. Return (chosen_candidates, objective, status, n_constraints)."""
    if not candidates:
        return [], 0.0, "empty", 0
    if not _MILP_AVAILABLE:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "greedy_fallback", 0

    import numpy as np

    n = len(candidates)
    step_now = int(world.step)

    # Inventories.
    src_ids = sorted({c.src_id for c in candidates})
    src_inv: dict[int, tuple[int, int]] = {}
    for sid in src_ids:
        p = world.planets_by_id.get(sid)
        if p is None:
            src_inv[sid] = (0, 0)
        else:
            src_inv[sid] = (int(p.ships), int(p.production))

    tgt_ids = sorted({c.tgt_id for c in candidates})

    # Objective: minimize -value (with lex tie-breaker for stability).
    c_vec = np.array(
        [-(c.value - 1e-6 * c.column_id) for c in candidates], dtype=float,
    )

    A_rows: list[list[float]] = []
    b_ub: list[float] = []

    # (C1) Per-source budget over time. DEFENDER_GUARD is subtracted ONCE
    # from the right-hand side per (src, u) row — not per launch — so a
    # source can do many launches as long as its CUMULATIVE outflow leaves
    # DEFENDER_GUARD ships at the home.
    #   Σ_{c: src(c)=src, c.fire_step ≤ u} ships(c) · x_c
    #     ≤ initial(src) + prod(src) · (u - step_now) - DEFENDER_GUARD
    fire_ticks_for_budget = sorted({c.fire_step for c in candidates})
    for sid in src_ids:
        initial, prod = src_inv[sid]
        for u in fire_ticks_for_budget:
            row = [0.0] * n
            any_in_row = False
            for j, c in enumerate(candidates):
                if c.src_id == sid and c.fire_step <= u:
                    row[j] = float(c.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_rows.append(row)
            b_ub.append(float(initial + prod * max(0, u - step_now) - DEFENDER_GUARD))

    # (C2) Per-target gang-up cap.
    for tid in tgt_ids:
        row = [0.0] * n
        any_in_row = False
        for j, c in enumerate(candidates):
            if c.tgt_id == tid:
                row[j] = 1.0
                any_in_row = True
        if not any_in_row:
            continue
        A_rows.append(row)
        b_ub.append(float(MAX_CONTESTERS_PER_TARGET))

    if not A_rows:
        # No constraints — just pick all positive.
        chosen = [c for c in candidates if c.value > 0]
        obj = sum(c.value for c in chosen)
        return chosen, obj, "no_constraints", 0

    A = np.array(A_rows, dtype=float)
    b = np.array(b_ub, dtype=float)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n, dtype=int)
    constraints = LinearConstraint(A, ub=b)

    try:
        res = milp(c=c_vec, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": time_limit_seconds})
    except Exception:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "milp_exception_greedy", len(A_rows)

    if res.x is None:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "milp_no_solution_greedy", len(A_rows)

    chosen = [c for j, c in enumerate(candidates) if res.x[j] > 0.5]
    obj = sum(c.value for c in chosen)
    return chosen, obj, "milp_ok", len(A_rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def opening_plan(world, model, my_id: int, num_seats: int,
         *, time_limit_seconds: float = 0.15) -> OpeningPlan:
    """Build the opening schedule for the current world."""
    candidates, waterfall = _build_candidates(world, model, my_id, num_seats)
    if not candidates:
        return OpeningPlan(schedule=[], objective=0.0,
                           n_vars=0, n_constraints=0,
                           status="no_candidates",
                           pruning_waterfall=waterfall)

    chosen, obj, status, n_constraints = _solve_milp(
        candidates, world, my_id, time_limit_seconds,
    )

    schedule = [
        ScheduleEntry(
            fire_step=c.fire_step, src_id=c.src_id, tgt_id=c.tgt_id,
            ships=c.ships, angle=c.angle, eta=c.eta, value=c.value,
        )
        for c in sorted(chosen, key=lambda c: (c.fire_step, c.column_id))
    ]

    return OpeningPlan(
        schedule=schedule, objective=float(obj),
        n_vars=len(candidates), n_constraints=int(n_constraints),
        status=str(status), pruning_waterfall=waterfall,
    )
