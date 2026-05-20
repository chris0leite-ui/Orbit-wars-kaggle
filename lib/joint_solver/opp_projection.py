"""Multi-launch opp projection (Phase 5D).

Replaces `predict_opp_responses` (1 launch per opp source) with a
SEQUENCE of opp launches per source over a 15-tick horizon, mirroring
`lib/opp_model.lite_greedy_policy`'s ROI-greedy logic.

Why: Phase 5C's outcome-table-aware LP inherited the 1-shot opp model's
under-projection bias. Real opp (the trajectory baseline) fires 5-8
launches per turn from multiple sources; we projected ~4-8 total. The
outcome_table's `prod_stream` values were computed against an
under-projected opp → defense values too low → mid-game collapse
documented in audit/2026-05-20-phase5b-root-cause-analysis.md.

Algorithm (Approach A — pure Python, ~30 ms per turn):
For each opp source s, simulate K=HORIZON ticks of opp behavior:
  - Accumulate production each tick: current_ships += s.production
  - Each tick, if current_ships ≥ OPP_MIN_SHIPS:
    - Pick best non-self, non-already-targeted target by production / (dist + 1)
    - Compute capture cost given target's garrison + production during flight
    - If unaffordable: skip
    - Else: project (target_id, absolute_eta, opp_id, ships) and subtract from budget

Returns `list[(target_pid, eta_absolute, opp_owner, ships)]` — same shape
as `predict_opp_responses` for drop-in replacement, but typically 5-10×
more entries.
"""

from __future__ import annotations

import math

from lib.fleet import speed
fleet_speed = speed
from lib.trajectory import predict_fleet_fate


# Horizon for the multi-launch projection. Empirical seed-42 trace
# (step 50): HORIZON=15 over-projected opp ships by 67% (360sh vs
# 216sh actual in-flight); the LP saw opp as 2× threatening and
# under-fired (28 launches vs baseline's 63). HORIZON=8 dials back
# to match observed opp activity more closely.
HORIZON = 8

# Per-source cap on projected launches. Even within HORIZON, real opps
# don't fire every single tick — they sometimes hold, defend, or wait
# for production. Cap at 3 prevents runaway projection from any one
# source.
MAX_LAUNCHES_PER_SOURCE = 3

# Per-launch minimums (mirror lite_greedy_policy:179 / opp_model.py:200-204).
OPP_MIN_SHIPS = 5          # don't project launches below this size
OPP_SHIP_FRACTION = 0.7    # aggressive launch fraction
OPP_MIN_LAUNCH_SHIPS = 5   # absolute minimum per launch
MAX_TARGETS_SCANNED = 6    # top-K targets considered per opp source per tick


def predict_opp_multi_launch(world, my_id: int, num_seats: int,
                             *, horizon: int = HORIZON) -> list[tuple[int, int, int, int]]:
    """Project opp launches over `horizon` ticks per opp source.

    For each opp seat o ≠ my_id, for each opp planet s owned by o:
        Simulate s's production + launch behavior over [0, horizon).
        Each launch records (target_pid, absolute_eta, opp_owner, ships).

    Returns `list[(target_pid, eta_absolute, opp_owner, ships)]`.

    `eta_absolute` is measured from the CURRENT tick (i.e., launches
    projected at tick_offset=t with flight_time=f produce eta=t+f).
    """
    arrivals: list[tuple[int, int, int, int]] = []

    all_planets = list(world.planets_by_id.values())
    if not all_planets:
        return arrivals

    for opp_id in range(num_seats):
        if opp_id == int(my_id):
            continue
        opp_planets = [p for p in all_planets if int(p.owner) == opp_id]
        for src in opp_planets:
            # Project this source's launches over the horizon.
            _project_source(
                src, opp_id, all_planets, world, arrivals,
                horizon=int(horizon),
            )
    return arrivals


def _project_source(src, opp_id: int, all_planets: list, world,
                    arrivals: list, *, horizon: int) -> None:
    """Simulate per-tick launch decisions for ONE opp source."""
    current_ships = float(int(src.ships))
    prod = int(src.production)
    # Avoid spamming the same target across multiple ticks: once we project
    # a launch to a target, don't re-target it in the same projection.
    already_targeted: set[int] = set()
    launches_made = 0

    for tick_offset in range(horizon):
        if launches_made >= MAX_LAUNCHES_PER_SOURCE:
            break
        # Accrue production at the START of each tick except the very first.
        if tick_offset > 0:
            current_ships += prod
        if current_ships < OPP_MIN_SHIPS:
            continue

        # Pick best target by production/distance, excluding already-targeted
        # and own planets.
        best = _pick_target(
            src, opp_id, all_planets, already_targeted=already_targeted,
        )
        if best is None:
            continue

        # Compute aggressive launch size, capped at budget.
        budget = int(current_ships)
        agg_ships = max(OPP_MIN_LAUNCH_SHIPS, int(budget * OPP_SHIP_FRACTION))
        if agg_ships > budget:
            agg_ships = budget
        if agg_ships < OPP_MIN_LAUNCH_SHIPS:
            continue

        # Compute flight ETA via straight-line + fleet speed (mirror
        # lite_greedy_policy:205-211; no aim_orbiting precision — that's
        # fine for a projection).
        spd = fleet_speed(agg_ships)
        if spd <= 0:
            continue
        dx = float(best.x) - float(src.x)
        dy = float(best.y) - float(src.y)
        d = math.sqrt(dx * dx + dy * dy)
        flight = max(0.0, d - float(src.radius) - float(best.radius) - 0.1)
        eta = max(1, int(math.ceil(flight / spd)))

        # Capture-size feasibility. Defenders accrue production only if owned.
        if int(best.owner) == -1:
            defenders_at_eta = float(int(best.ships))
        else:
            # Defenders grow by best.production per tick from the projection's
            # NOW (tick_offset=0) to the arrival (tick_offset + eta).
            defenders_at_eta = (
                float(int(best.ships)) + float(int(best.production))
                * float(tick_offset + eta)
            )
        needed = int(math.ceil(defenders_at_eta)) + 1
        if needed > budget:
            already_targeted.add(int(best.id))
            continue  # can't afford this target this tick

        ships_launch = max(agg_ships, needed)
        if ships_launch > budget:
            ships_launch = budget

        # Trajectory feasibility (sun/oob/wrong-planet).
        angle = math.atan2(dy, dx)
        try:
            fate = predict_fleet_fate(src, best, angle, ships_launch, world)
        except Exception:
            fate = None
        if fate is not None and getattr(fate, "outcome", "") != "target":
            # Not feasible — opp wouldn't actually try this one.
            already_targeted.add(int(best.id))
            continue

        # Record projection.
        arrivals.append((
            int(best.id),
            int(tick_offset + eta),
            int(opp_id),
            int(ships_launch),
        ))
        current_ships -= float(ships_launch)
        already_targeted.add(int(best.id))
        launches_made += 1


def _pick_target(src, opp_id: int, all_planets: list,
                 *, already_targeted: set[int]):
    """Return best (production / (dist + 1)) target for opp source,
    excluding opp's own planets and already-targeted ones."""
    best = None
    best_score = -1.0
    sx = float(src.x)
    sy = float(src.y)
    for t in all_planets:
        if int(t.owner) == int(opp_id):
            continue
        if int(t.id) == int(src.id):
            continue
        if int(t.id) in already_targeted:
            continue
        dx = float(t.x) - sx
        dy = float(t.y) - sy
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            continue
        score = float(int(t.production)) / (d + 1.0)
        if score > best_score:
            best_score = score
            best = t
    return best
