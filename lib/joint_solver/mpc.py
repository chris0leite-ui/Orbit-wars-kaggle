"""Per-turn orchestration for the analytical agent.

Each turn:
  1. Build World + WorldModel from obs.
  2. Run the proposer + migration solver → prerank tuples.
  3. Convert prerank to Columns with closed-form value (via value.py).
  4. Solve the multi-turn LP via lp.solve_multi_turn.
  5. Return emit moves (only wait_N == 0 columns from the LP solution).

Fallback chain on solver issues:
  - MILP infeasible / scipy missing → greedy multi-turn (lp.py internal).
  - Any unexpected exception → empty move list (safer than corrupting
    a turn; the env keeps the previous state).

Caching: outcome tables and timeline scaffolding could be cached across
turns (the in-flight fleet ledger evolves slowly). Deferred to Phase 4
once we measure wallclock per turn from the smoke runs.

Stackelberg opp-counter is intentionally NOT implemented in Phase 3 MVP
— the STOP gate is "multi-turn-coordinated move per 5 turns", which is
demonstrable with single-iteration LP solves. The opp-projection
extension is a Phase 3.5 / Phase 4 add.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.chooser_trajectory import (
    merge_ledgers,
    predict_opp_responses,
)
from agents.baseline.migration_solver import propose_migrations
from agents.baseline.proposer import MAX_HORIZON, propose
from lib.intent import World
from lib.joint_solver.columns import column_from_candidate
from lib.joint_solver.lp import (
    DEFAULT_MAX_CONTESTERS_PER_TARGET,
    DEFAULT_MAX_WAIT_N,
    MultiTurnResult,
    solve_multi_turn,
)
from lib.joint_solver.lp_outcome import (
    OutcomeAwareResult,
    solve_outcome_aware,
)
from lib.joint_solver.opening_planner import OPENING_HORIZON, plan as opening_plan
from lib.joint_solver.predicate import is_winning_state
from lib.joint_solver.portfolio import smallest_winning_portfolio
from lib.joint_solver.value import DEFAULT_GAMMA, value_for_candidate
from lib.world_model import WorldModel, simulate_planet_timeline


@dataclass
class MpcDiagnostics:
    """Per-turn telemetry for the introspect script and tests."""
    step: int
    n_prerank: int
    n_columns: int
    n_positive_columns: int
    n_fired_columns: int
    n_emitted_moves: int
    objective: float
    solver_status: str
    n_vars: int
    n_constraints: int
    n_opp_projections: int = 0
    is_winning_state: bool = False
    portfolio_size: int = 0
    portfolio_filtered: bool = False
    n_columns_before_filter: int = 0
    fired_wait_distribution: dict[int, int] = field(default_factory=dict)


def _model_with_opp_projection(world, model, *, my_id: int, num_seats: int):
    """Return a (possibly-augmented) WorldModel that incorporates the
    1-turn opp lookahead from predict_opp_responses. This is the
    "single-shot Stackelberg" opp model: assume each enemy source fires
    its nearest-best target now; merge those projected arrivals into the
    ledger so our value function sees a pessimistic garrison evolution.

    The PI's directive emphasised joint modeling — this is the minimal
    opp-projection step. Phase 4 may extend to a 2-iter Stackelberg
    (re-project opp given our LP solution).

    Returns `(model_with_opp, opp_arrivals)`. If projection fails or
    produces no arrivals, returns `(model, [])`.
    """
    try:
        opp_arrivals = predict_opp_responses(world, int(my_id), int(num_seats))
    except Exception:
        return model, []
    if not opp_arrivals:
        return model, []
    new_ledger = merge_ledgers(model.ledger, opp_arrivals)
    new_timelines = dict(model.timelines)
    for pid in {pid for (pid, _e, _o, _s) in opp_arrivals}:
        planet = world.planets_by_id.get(int(pid))
        if planet is None:
            continue
        new_timelines[int(pid)] = simulate_planet_timeline(
            planet, new_ledger.get(int(pid), []), model.horizon,
        )
    augmented = WorldModel(
        ledger=new_ledger, timelines=new_timelines, horizon=model.horizon,
    )
    return augmented, opp_arrivals


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}


def _num_seats(planets, fleets) -> int:
    """Infer num_seats from owner ids present in obs."""
    owners = {int(p.owner) for p in planets if int(p.owner) >= 0}
    owners.update(int(f.owner) for f in fleets if int(f.owner) >= 0)
    if not owners:
        return 2
    return max(2, max(owners) + 1)


def _build_columns(prerank, world, model, *, my_id: int,
                   gamma: float = DEFAULT_GAMMA):
    """Convert a prerank list to Columns with values via value_for_candidate."""
    columns = []
    for idx, c in enumerate(prerank):
        value = float(value_for_candidate(c, world, model, my_id=int(my_id),
                                          gamma=float(gamma)))
        columns.append(column_from_candidate(
            c, column_id=idx, owner=int(my_id), value=value,
        ))
    return columns


def solve_turn(obs, configuration=None, *,
               gamma: float = DEFAULT_GAMMA,
               max_contesters_per_target: int = DEFAULT_MAX_CONTESTERS_PER_TARGET,
               max_wait_N: int = DEFAULT_MAX_WAIT_N,
               time_limit_seconds: float = 0.3,
               return_diagnostics: bool = False):
    """Compute a turn's moves.

    Returns either `moves` (default) or `(moves, MpcDiagnostics)` when
    `return_diagnostics=True`.
    """
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return ([] if not return_diagnostics else
                ([], MpcDiagnostics(step=int(obs_d.get("step", 0) or 0),
                                    n_prerank=0, n_columns=0,
                                    n_positive_columns=0, n_fired_columns=0,
                                    n_emitted_moves=0, objective=0.0,
                                    solver_status="empty_obs",
                                    n_vars=0, n_constraints=0)))

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return ([] if not return_diagnostics else
                ([], MpcDiagnostics(step=int(obs_d.get("step", 0) or 0),
                                    n_prerank=0, n_columns=0,
                                    n_positive_columns=0, n_fired_columns=0,
                                    n_emitted_moves=0, objective=0.0,
                                    solver_status="no_targets_or_sources",
                                    n_vars=0, n_constraints=0)))

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    step_now = int(obs_d.get("step", 0) or 0)

    # Phase 4 endgame predicate gate — applies BEFORE both the opening
    # planner (Phase 5A) and the LP (Phase 4). If we're already winning by
    # closed-form math, idle — no need to risk ships, whether in opening
    # or post-opening. 2P only; 4P bypasses.
    if num_seats == 2:
        opp_id_pre = 1 - me
        try:
            if is_winning_state(world, me, opp_id_pre):
                if not return_diagnostics:
                    return []
                return [], MpcDiagnostics(
                    step=step_now, n_prerank=0, n_columns=0,
                    n_positive_columns=0, n_fired_columns=0,
                    n_emitted_moves=0, objective=0.0,
                    solver_status="endgame_winning_idle",
                    n_vars=0, n_constraints=0,
                    n_opp_projections=0, is_winning_state=True,
                    portfolio_size=0, portfolio_filtered=False,
                    n_columns_before_filter=0,
                )
        except Exception:
            pass  # if predicate evaluation fails, fall through

    # Phase 5A: opening planner dispatch.
    # For step < OPENING_HORIZON, solve the opening as a one-shot multi-turn
    # MILP and commit to its schedule. We emit only entries with
    # fire_step == step_now (the rest of the schedule is planning intent the
    # next re-derivation will refresh). Critically, we do NOT fall through
    # to Phase 4 LP just because the schedule has no fire_step==step_now
    # entry — the planner's INTENT is to wait this tick and fire on a
    # scheduled later tick; falling through would override that with
    # arbitrary Phase 4 emissions and break the commit-and-execute contract.
    #
    # Only true fallback: planner returned no candidates at all (e.g.,
    # source pool empty because every my-planet has < MIN_SOURCE_SHIPS).
    # In that case, let the existing Phase 4 LP handle the turn.
    if step_now < OPENING_HORIZON:
        op = opening_plan(world, model, me, num_seats)
        if op.n_vars > 0 or op.schedule:
            opening_moves = [
                [int(e.src_id), float(e.angle), int(e.ships)]
                for e in op.schedule if int(e.fire_step) == step_now
            ]
            if not return_diagnostics:
                return opening_moves
            wait_dist: dict[int, int] = {}
            for e in op.schedule:
                offset = int(e.fire_step) - step_now
                wait_dist[offset] = wait_dist.get(offset, 0) + 1
            diag = MpcDiagnostics(
                step=step_now,
                n_prerank=0,
                n_columns=op.n_vars,
                n_positive_columns=op.n_vars,
                n_fired_columns=len(op.schedule),
                n_emitted_moves=len(opening_moves),
                objective=float(op.objective),
                solver_status=f"opening:{op.status}",
                n_vars=int(op.n_vars),
                n_constraints=int(op.n_constraints),
                n_opp_projections=0,
                is_winning_state=False,
                portfolio_size=0,
                portfolio_filtered=False,
                n_columns_before_filter=op.n_vars,
                fired_wait_distribution=wait_dist,
            )
            return opening_moves, diag
        # else: no candidates at all — fall through to Phase 4 LP.

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    # Proposer + migration solver (mirror agents/baseline/main.py:267-283).
    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=MAX_HORIZON + 1,
    )
    migrations = propose_migrations(world, model, me)
    prerank = list(prerank) + list(migrations)

    # Single-shot Stackelberg: project opp's likely best counter-launches
    # via predict_opp_responses and bake them into the model that the
    # value function sees. With this, our W1/W2 lower bounds reflect a
    # pessimistic-but-realistic garrison evolution (opp produces AND
    # counter-attacks within the planning horizon).
    model_with_opp, opp_arrivals = _model_with_opp_projection(
        world, model, my_id=me, num_seats=num_seats,
    )

    columns = _build_columns(prerank, world, model_with_opp, my_id=me, gamma=gamma)
    n_columns_before_filter = len(columns)

    # Endgame predicate gate (2P only — 4P bypasses).
    #
    # If the closed-form winning-state predicate already holds for me,
    # ANY launch is a risk: I gain nothing from capturing more (production
    # margin × turns_left already > opp's recovery capacity), but every
    # ship I send is one less defender if opp coordinates a surprise.
    # → return [] (preserve ownership).
    #
    # If the predicate is False but a smallest_winning_portfolio exists,
    # restrict the LP to columns targeting those planets (plus own-planet
    # reinforces / migrations). Focuses ship spend on targets that
    # actually flip the predicate to True.
    #
    # If neither (4P, or 2P with no winnable portfolio), fall through to
    # the LP unfiltered.
    winning_now = False
    portfolio_filtered = False
    portfolio: list[int] = []
    if num_seats == 2:
        opp_id = 1 - me  # 2P: the unique non-me seat.
        try:
            winning_now = bool(is_winning_state(world, me, opp_id))
        except Exception:
            winning_now = False
        if winning_now:
            if not return_diagnostics:
                return []
            diag = MpcDiagnostics(
                step=int(obs_d.get("step", 0) or 0),
                n_prerank=len(prerank),
                n_columns=len(columns),
                n_positive_columns=sum(1 for c in columns if c.value > 0.0),
                n_fired_columns=0,
                n_emitted_moves=0,
                objective=0.0,
                solver_status="endgame_winning_idle",
                n_vars=0,
                n_constraints=0,
                n_opp_projections=len(opp_arrivals),
                is_winning_state=True,
                portfolio_size=0,
                portfolio_filtered=False,
                n_columns_before_filter=n_columns_before_filter,
            )
            return [], diag
        try:
            portfolio = smallest_winning_portfolio(world, me, opp_id)
        except Exception:
            portfolio = []
        if portfolio:
            portfolio_set = set(int(pid) for pid in portfolio)
            # Keep columns targeting portfolio planets OR own-planet
            # reinforces / migrations (target.owner == me). Own-planet
            # filter is by tgt_id ∈ {ids of my planets}.
            my_planet_ids = {int(p.id) for p in my_planets}
            filtered = [
                c for c in columns
                if int(c.tgt_id) in portfolio_set or int(c.tgt_id) in my_planet_ids
            ]
            # Only apply the filter if it leaves a positive-value column;
            # otherwise the filter would zero out the LP. Defensive.
            if any(c.value > 0.0 for c in filtered):
                columns = filtered
                portfolio_filtered = True

    # Phase 5C: outcome-table-aware LP replaces the per-candidate Phase 4 LP.
    # The new objective uses production-stream-per-owner from outcome_table
    # subsets so defense and offense are valued on the same scale and the
    # LP makes a GLOBAL offense-vs-defense tradeoff (not per-candidate).
    res_oc: OutcomeAwareResult = solve_outcome_aware(
        columns, world, model_with_opp, opp_arrivals,
        my_id=me,
        time_limit_seconds=float(time_limit_seconds),
    )

    if not return_diagnostics:
        return res_oc.moves

    wait_dist: dict[int, int] = {}
    for col in res_oc.fired_columns:
        w = int(col.wait_N)
        wait_dist[w] = wait_dist.get(w, 0) + 1

    diag = MpcDiagnostics(
        step=int(obs_d.get("step", 0) or 0),
        n_prerank=len(prerank),
        n_columns=len(columns),
        n_positive_columns=sum(1 for c in columns if c.value > 0.0),
        n_fired_columns=len(res_oc.fired_columns),
        n_emitted_moves=len(res_oc.moves),
        objective=float(res_oc.objective) if res_oc.objective == res_oc.objective else 0.0,
        solver_status=str(res_oc.status),
        n_vars=int(res_oc.n_x_vars + res_oc.n_y_vars),
        n_constraints=int(res_oc.n_constraints),
        n_opp_projections=len(opp_arrivals),
        is_winning_state=winning_now,
        portfolio_size=len(portfolio),
        portfolio_filtered=portfolio_filtered,
        n_columns_before_filter=n_columns_before_filter,
        fired_wait_distribution=wait_dist,
    )
    return res_oc.moves, diag
