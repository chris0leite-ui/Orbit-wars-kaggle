"""Opening search — widened multi-turn MILP reading from the trajectory matrix.

Phase η.2 of /root/.knowledge-base/plans/do-it-thoroughly-consider-tingly-fox.md.

Drop-in replacement for `opening_planner.opening_plan` with three
structural changes the seed-42 introspect mandated:

1. **No K=8 prune.** All non-mine non-comet planets are considered as
   targets. The matrix has the viability data; no need to pre-filter
   by `prod/(dist+1)` (which dropped planet 16 on seed 42).

2. **Closed-form leaf value.** Replaces the heuristic `production ×
   hold_duration × γ^t` with `leaf_value_for_portfolios` over the
   committed schedule. The LP's own leaf model — same math the
   LP at step 30+ would use — so the opening's choices and the LP's
   choices stay coherent across the OPENING_HORIZON boundary.

3. **Capture-chain support.** For each parent capture candidate
   (my_src → neutral_tgt) the captured planet becomes a SOURCE for
   chain candidates at `launch_tick >= parent.arrival_tick + 1`. The
   chain candidate carries `parent_column_id` and the MILP adds a
   linkage row `x_chain ≤ x_parent` (mirroring Phase F2a's compound
   candidate linkage in `lp_outcome.py`).

The MILP shape (binary x per candidate, source-budget over time,
target gang-up) is preserved. Greedy fallback when `scipy.optimize.milp`
is unavailable.

Default OFF; opt-in via `LP_OPENING_SEARCH=1`. Flip to default ON only
after the n=8 A/Bs (vs FND base, vs moKOR-orbitfix) clear.
"""

from __future__ import annotations

import math
import os
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

from typing import Any

from lib.joint_solver.opening_planner import (
    OPENING_HORIZON,
    OPENING_DEFENDER_GUARD,
    OPENING_T_END,
    MAX_CONTESTERS_PER_TARGET,
    MIN_SOURCE_SHIPS,
    OpeningPlan,
    ScheduleEntry,
)
from lib.joint_solver.trajectory_matrix import (
    TrajectoryEntry,
    get_default as get_default_matrix,
)
from lib.world_model import simulate_planet_timeline


# ---------------------------------------------------------------------------
# Schedule cache — solve once per game, decant per turn
# ---------------------------------------------------------------------------
#
# Bug seen on the n=8 A/B vs FND base (0W/8L): stateless re-derive each
# turn picks "best fire at fire_step=step_now" — the matrix is anchored
# on initial planets, so the SAME early fire keeps being optimal at
# every turn. Net effect: we re-fire planet 0 → planet 8 every other
# tick, draining home and never executing the wait_N>0 captures the
# MILP nominally planned.
#
# Fix: solve the full schedule ONCE at first call within a game,
# cache it, and emit only the fires with `fire_step == step_now` on
# subsequent turns. Cache invalidates on game-fingerprint change
# (matches trajectory_matrix's fingerprint via obs_d["initial_planets"]).


class _ScheduleCache:
    """Per-game schedule cache. Same singleton pattern as
    trajectory_matrix / pending_schedule."""

    def __init__(self) -> None:
        self._fingerprint: Any = None
        self._schedule: list[ScheduleEntry] = []
        self._n_vars: int = 0
        self._n_constraints: int = 0
        self._status: str = ""
        self._objective: float = 0.0
        self._waterfall: dict = {}

    def reset(self) -> None:
        self._fingerprint = None
        self._schedule = []
        self._n_vars = 0
        self._n_constraints = 0
        self._status = ""
        self._objective = 0.0
        self._waterfall = {}

    def get_or_solve(self, ctx, *, time_limit_seconds: float = 0.4
                     ) -> OpeningPlan:
        fp = self._fingerprint_from_ctx(ctx)
        if fp != self._fingerprint:
            op = _solve_full_schedule(ctx, time_limit_seconds=time_limit_seconds)
            self._fingerprint = fp
            self._schedule = list(op.schedule)
            self._n_vars = int(op.n_vars)
            self._n_constraints = int(op.n_constraints)
            self._status = str(op.status)
            self._objective = float(op.objective)
            self._waterfall = dict(op.pruning_waterfall or {})
        return OpeningPlan(
            schedule=list(self._schedule),
            objective=float(self._objective),
            n_vars=int(self._n_vars),
            n_constraints=int(self._n_constraints),
            status=str(self._status),
            pruning_waterfall=dict(self._waterfall),
        )

    @staticmethod
    def _fingerprint_from_ctx(ctx) -> tuple:
        """Same anchor as trajectory_matrix — initial planet state."""
        obs_d = getattr(ctx, "obs_d", None) or {}
        init = obs_d.get("initial_planets") or []
        return ("opening_schedule", tuple(tuple(p) for p in init),
                round(float(getattr(ctx, "omega", 0.0)), 6))


_DEFAULT_CACHE = _ScheduleCache()


def clear_schedule_cache() -> None:
    _DEFAULT_CACHE.reset()


def get_schedule_cache() -> _ScheduleCache:
    return _DEFAULT_CACHE


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------


def opening_search_enabled() -> bool:
    """Re-read env var per call so tests + A/B harnesses can toggle without
    importlib.reload."""
    return os.environ.get("LP_OPENING_SEARCH", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


# ---------------------------------------------------------------------------
# Internal candidate (one pre-pruned MILP variable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchCandidate:
    column_id: int
    src_id: int
    tgt_id: int
    fire_step: int          # absolute env step at which the fire launches
    eta: int                # ticks of flight from launch
    arrival: int            # absolute arrival step
    ships: int              # capture size at arrival
    angle: float
    value: float            # closed-form leaf-stream value
    parent_column_id: Optional[int] = None   # chain linkage; None for direct fires
    # Source's available ships at fire_step (initial + prod*offset for
    # direct fires; post-capture garrison + prod*delta for chains).
    src_ships_at_fire: int = 0
    src_prod: int = 0


# ---------------------------------------------------------------------------
# Closed-form leaf value
# ---------------------------------------------------------------------------


def _leaf_value_for_capture(entry: TrajectoryEntry, ships_fired: int,
                            world, my_id: int,
                            *, alpha_opp_penalty: float = 1.0,
                            ship_cost: float = 1.0,
                            t_end: int = OPENING_T_END) -> float:
    """Closed-form per-candidate leaf value.

    Mirrors `leaf_outcome_table.leaf_value_for_portfolios` for a SINGLE
    capture's contribution: production stream of (tgt) accruing to me
    from `arrival_tick` through `t_end`, minus α × opp's foregone
    production stream on the same planet (if it was opp's), minus
    ship_cost × ships_fired.

    Per-candidate decomposition holds because the MILP solves over the
    sum of per-candidate values — the additive structure matches
    `_value_for_outcome`'s per-planet accumulation in lp_outcome.py.
    """
    tgt = world.planets_by_id.get(int(entry.tgt_id))
    if tgt is None:
        return 0.0

    step_now = int(getattr(world, "step", 0) or 0)
    ticks_held = max(0, int(t_end) - int(entry.arrival_tick))
    if ticks_held <= 0:
        return 0.0
    prod = int(tgt.production)

    # Stream me — what we GAIN from the capture: prod × ticks_held.
    me_stream = float(prod) * float(ticks_held)

    # Opp stream — what we STRIP from opp: if tgt was previously opp's,
    # they would have accrued prod × ticks_held from now (step_now) until
    # capture (arrival), then through (t_end - arrival).
    opp_owner_at_arr = int(entry.arrival_owner)
    if opp_owner_at_arr != int(my_id) and opp_owner_at_arr >= 0:
        # Opp owned at arrival; without our capture they'd own through t_end.
        opp_stream = float(alpha_opp_penalty) * float(prod) * float(ticks_held)
    else:
        opp_stream = 0.0

    # Ship cost — uniform per-ship.
    cost = float(ship_cost) * float(ships_fired)

    return me_stream + opp_stream - cost


# ---------------------------------------------------------------------------
# Candidate generation — wider source/target set, chain support
# ---------------------------------------------------------------------------


def _build_candidates(ctx, *, alpha_opp_penalty: float = 1.0,
                      ship_cost: float = 1.0,
                      max_chain_depth: int = 1) -> list[_SearchCandidate]:
    """Build the candidate list reading from the trajectory matrix.

    For each (my-planet src, non-mine non-comet tgt, launch_tick), look
    up the matrix; if viable, emit a direct candidate. For each direct
    candidate (parent), additionally emit chain candidates that fire
    from the to-be-captured tgt at `launch_tick >= parent.arrival + 1`,
    using the post-capture garrison estimate as the chain-source budget.
    """
    world = ctx.world
    my_id = int(ctx.me)
    step_now = int(getattr(world, "step", 0) or 0)
    matrix = get_default_matrix()

    candidates: list[_SearchCandidate] = []
    next_id = 0

    my_planets = [
        p for p in world.planets_by_id.values()
        if int(p.owner) == my_id and int(p.ships) >= MIN_SOURCE_SHIPS
    ]
    if not my_planets:
        return []

    comet_ids = set(world.comet_ids) if world.comet_ids else set()

    # --------- Direct candidates (parent = my-planet → non-mine tgt) ---------
    # Map (my_src_id, parent's tgt_id) → list of parent candidate column_ids,
    # used to anchor chains to the EARLIEST parent for that (src, tgt) pair.
    earliest_parent_by_capture: dict[tuple[int, int], int] = {}
    parent_candidates_by_capture: dict[tuple[int, int], list[_SearchCandidate]] = {}

    for src in my_planets:
        src_initial_ships = int(src.ships)
        src_prod = int(src.production)
        for tgt in world.planets_by_id.values():
            if int(tgt.id) == int(src.id):
                continue
            if int(tgt.owner) == my_id:
                continue
            if int(tgt.id) in comet_ids:
                continue

            for launch_tick in range(0, OPENING_HORIZON):
                entry = matrix.get(int(src.id), int(tgt.id), int(launch_tick))
                if entry is None:
                    continue
                # Affordability re-check at this launch_tick (matrix
                # stored a worst-case budget; same math). The MILP
                # re-checks via the C1 constraint, but the per-candidate
                # check prunes obviously infeasible entries early.
                src_budget = src_initial_ships + src_prod * int(launch_tick)
                if entry.ships_needed + OPENING_DEFENDER_GUARD > src_budget:
                    continue
                # Closed-form leaf value (uses min capture size as ships
                # fired — opening_search v1 doesn't variant-enumerate
                # ship counts; that's v2).
                value = _leaf_value_for_capture(
                    entry, int(entry.ships_needed), world, my_id,
                    alpha_opp_penalty=alpha_opp_penalty,
                    ship_cost=ship_cost,
                )
                if value <= 0.0:
                    continue
                cand = _SearchCandidate(
                    column_id=next_id,
                    src_id=int(src.id),
                    tgt_id=int(tgt.id),
                    fire_step=int(step_now + launch_tick),
                    eta=int(entry.eta_flight),
                    arrival=int(step_now + entry.launch_tick + entry.eta_flight),
                    ships=int(entry.ships_needed),
                    angle=float(entry.angle),
                    value=float(value),
                    parent_column_id=None,
                    src_ships_at_fire=int(src_budget),
                    src_prod=int(src_prod),
                )
                candidates.append(cand)
                next_id += 1
                key = (int(src.id), int(tgt.id))
                parent_candidates_by_capture.setdefault(key, []).append(cand)
                # Track earliest parent per capture for chain anchoring.
                existing = earliest_parent_by_capture.get(key)
                if existing is None:
                    earliest_parent_by_capture[key] = cand.column_id
                else:
                    earliest_cand = next(
                        c for c in candidates if c.column_id == existing
                    )
                    if cand.fire_step < earliest_cand.fire_step:
                        earliest_parent_by_capture[key] = cand.column_id

    # --------- Chain candidates (depth-1 only in v1) ---------
    # For each EARLIEST parent capture (my_src → neutral_tgt), emit
    # chain candidates that fire from neutral_tgt at launch_tick >=
    # parent.arrival_tick + 1. Skip chains for opp captures (post-
    # capture garrison includes ships lost to opp counter-fire — too
    # uncertain for v1).
    if max_chain_depth >= 1:
        for (src_my, captured_tgt), parent_col_id in earliest_parent_by_capture.items():
            parent = next(c for c in candidates if c.column_id == parent_col_id)
            # Only chain off NEUTRAL captures (opp captures have unstable
            # post-capture garrison in v1).
            captured_planet = world.planets_by_id.get(int(captured_tgt))
            if captured_planet is None:
                continue
            if int(captured_planet.owner) != -1:
                continue
            arrival_abs = int(parent.arrival)
            arrival_rel = arrival_abs - step_now
            if arrival_rel >= OPENING_HORIZON - 1:
                continue  # no time to chain-fire before horizon
            # Post-capture garrison = parent.ships - target_garrison_at_arrival.
            # We don't store target_garrison_at_arrival on the SearchCandidate
            # (matrix's TrajectoryEntry has it via arrival_garrison). Look it
            # up from the matrix again.
            parent_entry = matrix.get(
                parent.src_id, parent.tgt_id, parent.fire_step - step_now,
            )
            if parent_entry is None:
                continue
            post_capture_garrison = max(
                0,
                int(parent.ships) - int(math.ceil(parent_entry.arrival_garrison)),
            )
            chain_src_prod = int(captured_planet.production)
            # Chain target enumeration: every non-mine non-comet that
            # isn't the captured target itself (no self-loops).
            for chain_tgt in world.planets_by_id.values():
                if int(chain_tgt.id) == int(captured_tgt):
                    continue
                if int(chain_tgt.id) == int(src_my):
                    continue  # don't fire back at our own source via chain
                if int(chain_tgt.owner) == my_id:
                    continue
                if int(chain_tgt.id) in comet_ids:
                    continue
                for chain_launch_tick in range(arrival_rel + 1, OPENING_HORIZON):
                    chain_entry = matrix.get(
                        int(captured_tgt), int(chain_tgt.id),
                        int(chain_launch_tick),
                    )
                    if chain_entry is None:
                        continue
                    # Chain-source budget at chain_launch_tick.
                    chain_src_budget = (
                        post_capture_garrison
                        + chain_src_prod
                        * max(0, int(chain_launch_tick) - int(arrival_rel))
                    )
                    if chain_entry.ships_needed + OPENING_DEFENDER_GUARD > chain_src_budget:
                        continue
                    value = _leaf_value_for_capture(
                        chain_entry, int(chain_entry.ships_needed),
                        world, my_id,
                        alpha_opp_penalty=alpha_opp_penalty,
                        ship_cost=ship_cost,
                    )
                    if value <= 0.0:
                        continue
                    chain_cand = _SearchCandidate(
                        column_id=next_id,
                        src_id=int(captured_tgt),
                        tgt_id=int(chain_tgt.id),
                        fire_step=int(step_now + chain_launch_tick),
                        eta=int(chain_entry.eta_flight),
                        arrival=int(step_now + chain_entry.launch_tick + chain_entry.eta_flight),
                        ships=int(chain_entry.ships_needed),
                        angle=float(chain_entry.angle),
                        value=float(value),
                        parent_column_id=int(parent_col_id),
                        src_ships_at_fire=int(chain_src_budget),
                        src_prod=int(chain_src_prod),
                    )
                    candidates.append(chain_cand)
                    next_id += 1

    return candidates


# ---------------------------------------------------------------------------
# MILP solver — mirrors opening_planner._solve_milp shape + chain linkage
# ---------------------------------------------------------------------------


def _solve_milp(candidates: list[_SearchCandidate], world,
                time_limit_seconds: float):
    """Run the MILP. Return (chosen, objective, status, n_constraints)."""
    if not candidates:
        return [], 0.0, "empty", 0
    if not _MILP_AVAILABLE:
        chosen, obj = _greedy_fallback(candidates, world)
        return chosen, obj, "greedy_fallback", 0

    import numpy as np

    n = len(candidates)
    step_now = int(getattr(world, "step", 0) or 0)

    # Direct (parent-less) candidates contribute to source-budget
    # constraints from world.planets_by_id ship counts. Chain candidates
    # have their source-budget enforced via the linkage to parent (the
    # parent's capture brings the source online); we don't double-bind
    # them in the C1 constraint because their src_id isn't in
    # world.planets_by_id with my-ownership at step_now.
    direct_candidates = [c for c in candidates if c.parent_column_id is None]
    chain_candidates = [c for c in candidates if c.parent_column_id is not None]

    src_ids = sorted({c.src_id for c in direct_candidates})
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

    # (C1) Per-source budget over time — direct candidates only.
    fire_ticks_for_budget = sorted({c.fire_step for c in direct_candidates})
    for sid in src_ids:
        initial, prod = src_inv[sid]
        for u in fire_ticks_for_budget:
            row = [0.0] * n
            any_in_row = False
            for j, c in enumerate(candidates):
                if c.parent_column_id is not None:
                    continue
                if c.src_id == sid and c.fire_step <= u:
                    row[j] = float(c.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_rows.append(row)
            b_ub.append(float(
                initial + prod * max(0, u - step_now) - OPENING_DEFENDER_GUARD
            ))

    # (C2) Per-target gang-up cap (applies to ALL candidates: direct +
    # chain. We don't want to capture the same target via multiple paths).
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

    # (C3) Chain linkage: x_chain <= x_parent. Row form:
    # +1 * x_chain - 1 * x_parent <= 0.
    col_id_to_idx = {c.column_id: j for j, c in enumerate(candidates)}
    for chain in chain_candidates:
        parent_id = chain.parent_column_id
        if parent_id is None:
            continue
        parent_idx = col_id_to_idx.get(int(parent_id))
        chain_idx = col_id_to_idx[chain.column_id]
        row = [0.0] * n
        row[chain_idx] = 1.0
        if parent_idx is not None:
            row[parent_idx] = -1.0
        # else: parent wasn't generated (shouldn't happen since we built
        # chain from this parent's column_id). Defensive: pin x_chain=0.
        A_rows.append(row)
        b_ub.append(0.0)

    # (C4) Chain-source budget over time — for each captured planet
    # acting as source, ensure total chain emissions <= post-capture
    # available budget. The chain candidate's src_ships_at_fire is
    # already the per-tick budget at that launch_tick assuming the
    # capture; the constraint is then cumulative-by-tick like C1.
    chain_src_ids = sorted({c.src_id for c in chain_candidates})
    chain_fire_ticks = sorted({c.fire_step for c in chain_candidates})
    for sid in chain_src_ids:
        # Find the parent that captures sid — there should be exactly one
        # parent candidate per (my_src, sid) pair anchored via
        # earliest_parent_by_capture, and chain candidates from sid all
        # reference the same parent. Look up post_capture_garrison via
        # any one chain candidate (they all share the same parent).
        sample_chain = next(c for c in chain_candidates if c.src_id == sid)
        parent_idx = col_id_to_idx.get(int(sample_chain.parent_column_id))
        if parent_idx is None:
            continue
        parent_cand = candidates[parent_idx]
        # Post-capture garrison at parent's arrival + 0 ticks.
        # We embedded src_ships_at_fire in each chain candidate already,
        # so build a constraint at each chain_fire_step that limits
        # cumulative ship spend by the source's budget at that fire_step.
        for u in chain_fire_ticks:
            row = [0.0] * n
            any_in_row = False
            for j, c in enumerate(candidates):
                if c.parent_column_id is None:
                    continue
                if c.src_id == sid and c.fire_step <= u:
                    row[j] = float(c.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_rows.append(row)
            # Budget at u = post_capture_garrison + prod*(u - parent.arrival).
            # Find post_capture_garrison from sample_chain's src_ships_at_fire
            # backed out: src_ships_at_fire = post + prod*(fire_step - arrival)
            # → post = src_ships_at_fire - prod*(fire_step - arrival).
            post_capture = (
                int(sample_chain.src_ships_at_fire)
                - int(sample_chain.src_prod)
                * max(0, int(sample_chain.fire_step) - int(parent_cand.arrival))
            )
            available_at_u = (
                post_capture
                + int(sample_chain.src_prod)
                * max(0, int(u) - int(parent_cand.arrival))
                - OPENING_DEFENDER_GUARD
            )
            b_ub.append(float(max(0, available_at_u)))

    if not A_rows:
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
        chosen, obj = _greedy_fallback(candidates, world)
        return chosen, obj, "milp_exception_greedy", len(A_rows)

    if res.x is None:
        chosen, obj = _greedy_fallback(candidates, world)
        return chosen, obj, "milp_no_solution_greedy", len(A_rows)

    chosen = [c for j, c in enumerate(candidates) if res.x[j] > 0.5]
    obj = sum(c.value for c in chosen)
    return chosen, obj, "milp_ok", len(A_rows)


def _greedy_fallback(candidates: list[_SearchCandidate], world
                     ) -> tuple[list[_SearchCandidate], float]:
    """Pure-Python descending-value greedy with budget + gang-up + chain
    linkage tracking."""
    step_now = int(getattr(world, "step", 0) or 0)
    src_inv: dict[int, tuple[int, int]] = {}
    for c in candidates:
        if c.parent_column_id is not None:
            continue
        if c.src_id in src_inv:
            continue
        p = world.planets_by_id.get(c.src_id)
        if p is not None:
            src_inv[c.src_id] = (int(p.ships), int(p.production))

    emitted_by_src_fire: dict[tuple[int, int], int] = {}
    tgt_count: dict[int, int] = {}
    chosen_ids: set[int] = set()
    chosen: list[_SearchCandidate] = []
    obj = 0.0

    # Process direct candidates first (parents must commit before chains).
    direct_sorted = sorted(
        (c for c in candidates if c.parent_column_id is None),
        key=lambda x: x.value, reverse=True,
    )
    for c in direct_sorted:
        if tgt_count.get(c.tgt_id, 0) >= MAX_CONTESTERS_PER_TARGET:
            continue
        initial, prod = src_inv.get(c.src_id, (0, 0))
        used = sum(v for (s, fs), v in emitted_by_src_fire.items()
                   if s == c.src_id and fs <= c.fire_step)
        if used + c.ships > initial + prod * max(0, c.fire_step - step_now) - OPENING_DEFENDER_GUARD:
            continue
        chosen.append(c)
        chosen_ids.add(c.column_id)
        emitted_by_src_fire[(c.src_id, c.fire_step)] = (
            emitted_by_src_fire.get((c.src_id, c.fire_step), 0) + c.ships
        )
        tgt_count[c.tgt_id] = tgt_count.get(c.tgt_id, 0) + 1
        obj += c.value

    # Then chain candidates (must have parent chosen + chain-src budget).
    chain_emitted: dict[tuple[int, int], int] = {}
    chain_sorted = sorted(
        (c for c in candidates if c.parent_column_id is not None),
        key=lambda x: x.value, reverse=True,
    )
    for c in chain_sorted:
        if c.parent_column_id not in chosen_ids:
            continue  # parent not fired
        if tgt_count.get(c.tgt_id, 0) >= MAX_CONTESTERS_PER_TARGET:
            continue
        chain_used = sum(
            v for (s, fs), v in chain_emitted.items()
            if s == c.src_id and fs <= c.fire_step
        )
        if chain_used + c.ships > c.src_ships_at_fire - OPENING_DEFENDER_GUARD:
            continue
        chosen.append(c)
        chosen_ids.add(c.column_id)
        chain_emitted[(c.src_id, c.fire_step)] = (
            chain_emitted.get((c.src_id, c.fire_step), 0) + c.ships
        )
        tgt_count[c.tgt_id] = tgt_count.get(c.tgt_id, 0) + 1
        obj += c.value

    return chosen, obj


# ---------------------------------------------------------------------------
# Public entry point — drop-in replacement for opening_planner.opening_plan
# ---------------------------------------------------------------------------


def _solve_full_schedule(ctx, *, time_limit_seconds: float = 0.4
                         ) -> OpeningPlan:
    """One-shot full-schedule solve. Internal — opening_plan_search wraps
    this in the per-game cache so we don't re-derive every turn."""
    candidates = _build_candidates(ctx)
    if not candidates:
        return OpeningPlan(
            schedule=[], objective=0.0, n_vars=0, n_constraints=0,
            status="search_no_candidates",
            pruning_waterfall={"n_candidates": 0},
        )

    chosen, obj, status, n_constraints = _solve_milp(
        candidates, ctx.world, time_limit_seconds,
    )

    schedule = [
        ScheduleEntry(
            fire_step=c.fire_step, src_id=c.src_id, tgt_id=c.tgt_id,
            ships=c.ships, angle=c.angle, eta=c.eta, value=c.value,
        )
        for c in sorted(chosen, key=lambda c: (c.fire_step, c.column_id))
    ]

    n_direct = sum(1 for c in candidates if c.parent_column_id is None)
    n_chain = sum(1 for c in candidates if c.parent_column_id is not None)

    return OpeningPlan(
        schedule=schedule, objective=float(obj),
        n_vars=len(candidates), n_constraints=int(n_constraints),
        status=f"search:{status}",
        pruning_waterfall={
            "n_candidates": len(candidates),
            "n_direct": n_direct,
            "n_chain": n_chain,
            "n_chosen": len(chosen),
        },
    )


def opening_plan_search(ctx, *, time_limit_seconds: float = 0.4
                        ) -> OpeningPlan:
    """Build the opening schedule using the trajectory-matrix-backed
    widened search. SOLVES ONCE PER GAME (cached by fingerprint),
    decanting per turn — fixes the n=8 0W/8L regression where
    stateless re-derive picked the same early fire at every turn.

    Returns an `OpeningPlan` with the same shape as
    `opening_planner.opening_plan`, so `lib/pipeline/opening.py` can
    dispatch between the two via the env-var gate.

    `time_limit_seconds` default 0.4s — only the first turn of each
    game pays this cost; subsequent turns are O(1) cache lookup.
    """
    return _DEFAULT_CACHE.get_or_solve(ctx, time_limit_seconds=time_limit_seconds)
