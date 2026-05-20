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

    res: MultiTurnResult = solve_multi_turn(
        columns, world,
        my_id=me,
        max_contesters_per_target=int(max_contesters_per_target),
        max_wait_N=int(max_wait_N),
        time_limit_seconds=float(time_limit_seconds),
    )

    if not return_diagnostics:
        return res.moves

    wait_dist: dict[int, int] = {}
    for col in res.fired_columns:
        w = int(col.wait_N)
        wait_dist[w] = wait_dist.get(w, 0) + 1

    diag = MpcDiagnostics(
        step=int(obs_d.get("step", 0) or 0),
        n_prerank=len(prerank),
        n_columns=len(columns),
        n_positive_columns=sum(1 for c in columns if c.value > 0.0),
        n_fired_columns=len(res.fired_columns),
        n_emitted_moves=len(res.moves),
        objective=float(res.objective) if res.objective == res.objective else 0.0,
        solver_status=str(res.status),
        n_vars=int(res.n_vars),
        n_constraints=int(res.n_constraints),
        n_opp_projections=len(opp_arrivals),
        fired_wait_distribution=wait_dist,
    )
    return res.moves, diag
