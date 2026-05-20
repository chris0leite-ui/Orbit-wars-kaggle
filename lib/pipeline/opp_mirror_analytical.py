"""Mirror-analytical opp model — action-reactive Stage-4 alternative.

Given my candidate portfolio (a list of Columns I'd commit this turn),
predict opp's best-response by running the analytical pipeline's inner
LP from opp's POV.

Internal flow:
  1. Build an augmented WorldModel with my_portfolio's arrivals merged
     into the ledger (treating my future fleets as committed).
  2. From opp's POV:
       - opp_planets    = planets owned by opp
       - target_pool    = (all planets - opp) + opp's threatened
       - propose() with my_id=opp_id over (opp_planets, target_pool)
       - value_for_candidate() with my_id=opp_id
       - solve_outcome_aware() with my_id=opp_id and the augmented model
  3. Return opp's fired_columns translated to the 4-tuple shape:
       (target_pid, eta_relative_from_step_now, opp_owner, ships)

This is *recursive in form, not depth* — opp's own opp model is baked
into the world augmentation (my arrivals), not a nested call. No
recursion, bounded cost.

Cost: one solve_outcome_aware call per my portfolio (~15 ms per Phase B
funnel). For K_MY=8: 120 ms total. Inside Phase D's 200 ms substrate
budget.
"""

from __future__ import annotations

from agents.baseline.chooser_trajectory import merge_ledgers
from agents.baseline.migration_solver import propose_migrations
from agents.baseline.proposer import MAX_HORIZON, propose
from lib.joint_solver.columns import column_from_candidate
from lib.joint_solver.lp_outcome import solve_outcome_aware
from lib.joint_solver.value import DEFAULT_GAMMA, value_for_candidate
from lib.pipeline.types import TurnContext
from lib.world_model import WorldModel, simulate_planet_timeline


def _columns_to_arrivals(
    my_portfolio: list, step_now: int, my_id: int,
) -> list[tuple[int, int, int, int]]:
    """Convert my portfolio's Columns into (pid, eta_abs, owner, ships).

    eta_abs = step_now + wait_N + eta (matches lp_outcome.py:173 +
    leaf_outcome_table.column_to_arrival).
    """
    arrivals = []
    for col in my_portfolio:
        if int(col.ships) <= 0:
            continue
        eta_abs = int(step_now) + int(col.wait_N) + int(col.eta)
        arrivals.append((int(col.tgt_id), eta_abs, int(my_id), int(col.ships)))
    return arrivals


def _augment_model_with_my_arrivals(
    ctx: TurnContext, my_arrivals: list[tuple[int, int, int, int]],
) -> WorldModel:
    """Build a new WorldModel with my_arrivals merged into the ledger.

    Same pattern as opp_greedy_roi but for my arrivals (opp's POV sees
    them as committed enemy fleets).
    """
    if not my_arrivals:
        return ctx.model
    new_ledger = merge_ledgers(ctx.model.ledger, my_arrivals)
    new_timelines = dict(ctx.model.timelines)
    for pid in {pid for (pid, _e, _o, _s) in my_arrivals}:
        planet = ctx.world.planets_by_id.get(int(pid))
        if planet is None:
            continue
        new_timelines[int(pid)] = simulate_planet_timeline(
            planet, new_ledger.get(int(pid), []), ctx.model.horizon,
        )
    return WorldModel(
        ledger=new_ledger,
        timelines=new_timelines,
        horizon=ctx.model.horizon,
    )


def predict_opp_response_to_my_portfolio(
    ctx: TurnContext,
    my_portfolio: list,
    *,
    opp_id: int | None = None,
    time_limit_seconds: float = 0.04,
    gamma: float = DEFAULT_GAMMA,
) -> list[tuple[int, int, int, int]]:
    """Mirror-analytical opp response to my candidate portfolio.

    Returns a list of `(target_pid, eta_relative_from_step_now,
    opp_owner, ships)` — same shape as `predict_opp_multi_launch` /
    `leaf_outcome_table` consumers.

    Empty list on any failure (defensive — falls back to "opp does
    nothing" in the caller).
    """
    # Determine opp id. 2P only for now (4P uses single strongest opp;
    # gate elsewhere).
    if opp_id is None:
        if ctx.num_seats == 2:
            opp_id = 1 - int(ctx.me)
        else:
            # Pick the strongest non-me opp by current ship total.
            ship_totals: dict[int, int] = {}
            for p in ctx.planets:
                if int(p.owner) < 0 or int(p.owner) == ctx.me:
                    continue
                ship_totals[int(p.owner)] = (
                    ship_totals.get(int(p.owner), 0) + int(p.ships)
                )
            if not ship_totals:
                return []
            opp_id = max(ship_totals.items(), key=lambda kv: kv[1])[0]

    opp_id = int(opp_id)

    # Build opp's view of planets.
    opp_planets = [p for p in ctx.planets if int(p.owner) == opp_id]
    other_from_opp_view = [p for p in ctx.planets if int(p.owner) != opp_id]
    if not opp_planets or not other_from_opp_view:
        return []

    # Augment the model with my candidate arrivals (opp's POV: my future
    # fleets are committed).
    step_now = int(ctx.step_now)
    my_arrivals_abs = _columns_to_arrivals(
        my_portfolio, step_now=step_now, my_id=int(ctx.me),
    )
    augmented_model = _augment_model_with_my_arrivals(ctx, my_arrivals_abs)

    # Opp's threatened-own pool (planets opp owns and I'm threatening).
    try:
        threatened_opp = [
            p for p in opp_planets
            if augmented_model.time_to_enemy_threat(int(p.id), opp_id, ctx.world) is not None
        ]
    except Exception:
        threatened_opp = []
    opp_target_pool = other_from_opp_view + threatened_opp

    # Generate opp's prerank candidates.
    try:
        opp_prerank = propose(
            opp_planets, opp_target_pool, ctx.world, augmented_model,
            opp_id, ctx.omega,
            baseline_len=MAX_HORIZON + 1,
        )
    except Exception:
        return []
    try:
        opp_migrations = propose_migrations(ctx.world, augmented_model, opp_id)
    except Exception:
        opp_migrations = []
    opp_prerank = list(opp_prerank) + list(opp_migrations)

    if not opp_prerank:
        return []

    # Build opp's columns with value_for_candidate from opp's POV.
    opp_columns = []
    for idx, c in enumerate(opp_prerank):
        try:
            v = float(value_for_candidate(
                c, ctx.world, augmented_model,
                my_id=opp_id, gamma=float(gamma),
            ))
        except Exception:
            v = 0.0
        opp_columns.append(column_from_candidate(
            c, column_id=idx, owner=opp_id, value=v,
        ))

    if not opp_columns:
        return []

    # Run the inner LP from opp's POV.
    try:
        res = solve_outcome_aware(
            opp_columns, ctx.world, augmented_model,
            my_id=opp_id,
            time_limit_seconds=float(time_limit_seconds),
        )
    except Exception:
        return []

    # Translate fired_columns to (pid, eta_rel_from_step_now, owner, ships).
    opp_arrivals: list[tuple[int, int, int, int]] = []
    for col in (res.fired_columns or []):
        if int(col.ships) <= 0:
            continue
        eta_rel = int(col.wait_N) + int(col.eta)
        opp_arrivals.append((
            int(col.tgt_id), int(eta_rel), int(opp_id), int(col.ships),
        ))
    return opp_arrivals
