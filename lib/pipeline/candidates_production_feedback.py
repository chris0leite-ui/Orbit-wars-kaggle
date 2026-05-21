"""Phase F2a — production-feedback compound candidates.

For each base candidate that captures an opp planet `tgt` at arrival
`step_now + base.wait_N + base.eta`, generate "fire-from-captured-tgt"
candidates targeting other opp planets at fire-steps in
[arrival+1, arrival+max_compound_delay].

Each compound candidate is a Column with:
  - src_id = captured planet's id
  - tgt_id = some other opp planet
  - parent_column_id = the base candidate's column_id (LP enforces
    `x_compound <= x_parent` via linkage constraints)
  - ships = production(tgt) * compound_delay (post-capture
    accumulation; conservative estimate ignoring base-capture residue)
  - wait_N = compound_fire_step - step_now (LP's source-budget loop
    skips compound columns because their `src` isn't ours yet)

`predict_fleet_fate(tgt, opp_tgt, angle, ships, world, wait_N=N)`
already advances tgt's orbital position by N ticks before ray-casting,
so the closed-form trajectory check is correct even though tgt is
opp-owned at step_now.

Closes the action-space blind spot identified after Phase F1's 1/4
parity: the LP couldn't compound multi-tier captures within its
planning horizon, while the trajectory rollout did this implicitly.
"""

from __future__ import annotations

import math

from lib.aim import aim_orbiting
from lib.orbit import is_orbiting, predict_relative
from lib.trajectory import predict_fleet_fate
from lib.world_model import simulate_planet_timeline
from lib.joint_solver.columns import Column
from lib.pipeline.types import CandidateSet, TurnContext


DEFAULT_MAX_BASE_ARRIVAL_REL = 8       # only compound from captures arriving within 8 ticks
DEFAULT_MAX_COMPOUND_DELAY = 3         # fire from captured planet up to 3 ticks post-capture
DEFAULT_MAX_COMPOUND_TARGETS_PER_SRC = 3  # top-3 nearest opp planets per compound src
DEFAULT_MIN_COMPOUND_SHIPS = 3         # don't bother with launches below this size


def generate_compound_candidates(
    base_columns: list,
    ctx: TurnContext,
    *,
    next_col_id_start: int,
    max_base_arrival_rel: int = DEFAULT_MAX_BASE_ARRIVAL_REL,
    max_compound_delay: int = DEFAULT_MAX_COMPOUND_DELAY,
    max_compound_targets_per_src: int = DEFAULT_MAX_COMPOUND_TARGETS_PER_SRC,
    min_compound_ships: int = DEFAULT_MIN_COMPOUND_SHIPS,
) -> list:
    """Return new compound Columns for the LP to consider on top of base."""
    out: list = []
    next_id = int(next_col_id_start)

    me = int(ctx.me)
    omega = float(ctx.omega)
    step_now = int(ctx.step_now)
    world = ctx.world

    # Pre-build a lookup of opp planets (other than self-as-target) for
    # compound-target enumeration. We iterate ALL non-me planets — the
    # planet captured at compound_fire might itself be a re-capture
    # candidate, but that's a higher-order effect we skip.
    other_planets = list(ctx.other_planets)

    for base in base_columns:
        # Skip compound-on-compound (Phase F2a is depth-1 only).
        if getattr(base, "parent_column_id", None) is not None:
            continue
        # Only compound on captures (target currently opp-owned).
        tgt = world.planets_by_id.get(int(base.tgt_id))
        if tgt is None:
            continue
        if int(tgt.owner) == me:
            continue
        # Arrival relative to step_now (column.eta is from launch; for
        # wait_N > 0 launches, arrival = wait_N + eta).
        arrival_rel = int(base.wait_N) + int(base.eta)
        if arrival_rel > int(max_base_arrival_rel):
            continue

        tgt_tuple_now = (
            int(tgt.id), int(tgt.owner),
            float(tgt.x), float(tgt.y),
            int(tgt.ships), int(tgt.production), float(tgt.radius),
        )
        tgt_is_orbiting = is_orbiting(list(tgt_tuple_now))

        # Bugs #3/#4 fix: simulate planet tgt's timeline given (existing
        # ledger + base capture arrival). owner_at and ships_at give the
        # exact post-capture state, so we don't have to approximate with
        # `production * delay` and we can reject compounds when opp has
        # already re-captured before compound_fire_rel.
        existing_arrivals = list(ctx.model.ledger.get(int(tgt.id), []))
        augmented_arrivals = existing_arrivals + [
            (int(arrival_rel), int(me), int(base.ships)),
        ]
        try:
            tgt_timeline = simulate_planet_timeline(
                tgt, augmented_arrivals, ctx.model.horizon,
            )
        except Exception:
            continue
        owner_at = tgt_timeline.get("owner_at", {})
        ships_at = tgt_timeline.get("ships_at", {})

        # Enumerate compound fires at compound_fire_step = arrival + delay.
        for delay in range(1, int(max_compound_delay) + 1):
            compound_fire_rel = arrival_rel + delay  # relative to step_now
            # Bug #4: must still own the planet at compound_fire_rel.
            owner_at_fire = owner_at.get(int(compound_fire_rel))
            if owner_at_fire is None or int(owner_at_fire) != int(me):
                continue
            # Bug #3: use exact garrison from the simulated timeline
            # instead of `production * delay`. Reserve 1 ship as a
            # post-fire stub so the planet isn't left at zero garrison
            # (matches DEFENDER_GUARD's intent in opening_planner without
            # importing it cross-module).
            ships_at_fire = ships_at.get(int(compound_fire_rel))
            if ships_at_fire is None:
                continue
            ships_avail = max(0, int(float(ships_at_fire)) - 1)
            if ships_avail < int(min_compound_ships):
                continue

            # Captured planet's expected position at compound_fire_rel.
            if tgt_is_orbiting:
                src_x_c, src_y_c = predict_relative(
                    list(tgt_tuple_now), omega, compound_fire_rel,
                )
            else:
                src_x_c, src_y_c = float(tgt.x), float(tgt.y)

            # Rank candidate compound targets by distance-at-compound-fire-time
            # (cheap heuristic; aim_orbiting + predict_fleet_fate validate
            # feasibility per actual target).
            scored = []
            for o_tgt in other_planets:
                if int(o_tgt.id) == int(tgt.id):
                    continue
                # We DO want to allow re-attacking the originally-captured
                # planet's neighbours including planets I currently own
                # (defensive reinforce from new source) — but for simplicity,
                # restrict to non-me targets first. (Reinforce-from-captured
                # could be a follow-up if needed.)
                if int(o_tgt.owner) == me:
                    continue
                o_tuple = [
                    int(o_tgt.id), int(o_tgt.owner),
                    float(o_tgt.x), float(o_tgt.y),
                    int(o_tgt.ships), int(o_tgt.production), float(o_tgt.radius),
                ]
                if is_orbiting(o_tuple):
                    o_x_c, o_y_c = predict_relative(
                        o_tuple, omega, compound_fire_rel,
                    )
                else:
                    o_x_c, o_y_c = float(o_tgt.x), float(o_tgt.y)
                d2 = (o_x_c - src_x_c) ** 2 + (o_y_c - src_y_c) ** 2
                scored.append((d2, o_tgt, o_tuple, o_x_c, o_y_c))
            scored.sort(key=lambda t: t[0])
            scored = scored[: int(max_compound_targets_per_src)]

            for _d2, o_tgt, o_tuple_now, o_x_c, o_y_c in scored:
                # Aim from captured src@compound_fire to o_tgt.
                # aim_orbiting projects target forward by eta internally,
                # so pass o_tgt's STARTING position at compound_fire (we
                # already rotated it above for distance ranking; reuse).
                o_tuple_at_compound = list(o_tuple_now)
                o_tuple_at_compound[2] = o_x_c
                o_tuple_at_compound[3] = o_y_c
                try:
                    if is_orbiting(o_tuple_at_compound):
                        res = aim_orbiting(
                            (src_x_c, src_y_c), float(tgt.radius),
                            o_tuple_at_compound, float(o_tgt.radius),
                            ships_avail, omega,
                        )
                        if res is None:
                            continue
                        angle, _arrival_xy, eta_flight = res
                        eta_flight = max(1, int(math.ceil(float(eta_flight))))
                    else:
                        angle = math.atan2(o_y_c - src_y_c, o_x_c - src_x_c)
                        from lib.fleet import speed as fleet_speed
                        spd = fleet_speed(ships_avail)
                        if spd <= 0:
                            continue
                        flight = max(
                            0.0,
                            math.hypot(src_x_c - o_x_c, src_y_c - o_y_c)
                            - float(tgt.radius) - float(o_tgt.radius) - 0.1,
                        )
                        eta_flight = max(1, int(math.ceil(flight / spd)))
                except Exception:
                    continue

                # Trajectory feasibility: closed-form ray-cast from
                # captured planet at compound_fire_rel.
                try:
                    fate = predict_fleet_fate(
                        tgt, o_tgt, float(angle), int(ships_avail),
                        world, wait_N=int(compound_fire_rel),
                    )
                except Exception:
                    continue
                if fate is None or fate.outcome != "target":
                    continue
                if int(fate.hit_planet_id) != int(o_tgt.id):
                    continue

                # Build compound column.
                out.append(Column(
                    column_id=int(next_id),
                    src_id=int(tgt.id),
                    tgt_id=int(o_tgt.id),
                    ships=int(ships_avail),
                    wait_N=int(compound_fire_rel),
                    angle=float(angle),
                    eta=int(eta_flight),
                    owner=int(me),
                    value=1.0,  # placeholder; pre-rank applies its own value
                    horizon_hint=0,
                    cheap_delta=0.0,
                    is_opp=False,
                    parent_column_id=int(base.column_id),
                ))
                next_id += 1

    return out


def candidates_with_production_feedback(ctx: TurnContext) -> CandidateSet:
    """Stage-2 alternative: base candidates + compound candidates.

    Returns a CandidateSet whose prerank is the base set (compound
    candidates are appended downstream by `prerank_with_production_feedback`
    because they need column_ids assigned post-prerank).

    Phase F2a integration: this module pairs with a custom prerank stage
    that builds base columns first, THEN appends compound columns
    referencing base column_ids. See `prerank_passthrough_with_feedback`.
    """
    from lib.pipeline.candidates import candidates_default
    return candidates_default(ctx)
