"""Orbital phase-lead targeting — keystone library for Options C, D, E, G.

For each (source, target) planet pair, this module answers two questions:

1. **closest_approach(source, target, omega, horizon)** — what is the
   minimum-distance moment over the next `horizon` turns, and when does
   it occur? This is the time-dependent version of `geometry.dist` that
   takes orbital motion into account. The empirical battlefield report
   (`audit/2026-05-12-battlefield-geometry-report.md` §8b) shows that,
   averaged across all orbiting targets, the closest-approach distance
   is 35% shorter than the instantaneous distance at t=0; the best
   target per seed shows a 75% median savings.

2. **best_launch_plan(source, target, omega, ships, horizon)** — what
   launch turn (relative to NOW) yields the minimum fleet-travel
   distance to the target, accounting for orbital motion of both
   source and target? Returns the lead-aim angle, arrival turn, ETA,
   and travel distance.

Pure-library: no agent-loop integration yet. Consumers (Options D, E, G)
register later once score-scale calibration is sorted.

Verification against `kaggle_environments` orbit math is in
`tests/test_orbit_lead.py::test_closest_approach_matches_env_seed42`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fleet import speed as fleet_speed
from .geometry import CENTER, Point
from .orbit import is_orbiting, predict_relative

DEFAULT_HORIZON: int = 100  # turns


# ---------------------------------------------------------------------------
# Position prediction (handles static + orbiting uniformly)
# ---------------------------------------------------------------------------


def position_at(planet, omega: float, lead_turns: float) -> Point:
    """Predicted (x, y) of `planet` `lead_turns` after NOW.

    Static planets return their current (x, y) unchanged. Orbiting
    planets rotate by `omega * lead_turns` radians around (50, 50).
    """
    if is_orbiting(planet):
        return predict_relative(planet, omega, lead_turns)
    return (planet[2], planet[3])


def distance_at(source, target, omega: float, lead_turns: float) -> float:
    """Centre-to-centre distance between `source` and `target` at the
    same future turn `lead_turns` after NOW."""
    sx, sy = position_at(source, omega, lead_turns)
    tx, ty = position_at(target, omega, lead_turns)
    return math.hypot(sx - tx, sy - ty)


# ---------------------------------------------------------------------------
# Closest-approach: when over the next `horizon` turns is the pair nearest?
# ---------------------------------------------------------------------------


def closest_approach(
    source,
    target,
    omega: float,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[int, float]:
    """Scan turns [0, horizon] and return (best_lead_turns, min_distance).

    Both planets are advanced to the same future turn for each candidate.
    Static planets contribute a constant; orbiting planets sweep along
    their rings. Resolution is one turn (integer); orbital speeds in
    Orbit Wars are at most ~0.05 rad/turn so sub-turn resolution would
    add < 0.1 distance unit on a 10-unit ring — not worth the cost.
    """
    best_t = 0
    best_d = distance_at(source, target, omega, 0)
    for t in range(1, horizon + 1):
        d = distance_at(source, target, omega, t)
        if d < best_d:
            best_d = d
            best_t = t
    return best_t, best_d


# ---------------------------------------------------------------------------
# Best-launch-plan: when to fire so a fleet arrives at minimum travel cost?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchPlan:
    """Outcome of `best_launch_plan`."""

    launch_offset: int        # turns from NOW to wait before launching
    arrival_turn: int         # turns from NOW that the fleet hits the target
    eta: int                  # turns of flight (arrival_turn - launch_offset)
    distance: float           # board units the fleet must traverse
    source_pos: Point         # source (x, y) at launch
    target_pos: Point         # target (x, y) at arrival (lead-aim point)
    aim_angle_rad: float      # math.atan2(ty - sy, tx - sx), for the env's action format


def best_launch_plan(
    source,
    target,
    omega: float,
    ships: int | float,
    horizon: int = DEFAULT_HORIZON,
    inner_iters: int = 2,
) -> LaunchPlan | None:
    """Find the launch offset (relative to NOW) that minimises the
    fleet's travel distance to `target`, accounting for orbital motion
    of BOTH source and target.

    The fleet flies in a straight line: it leaves `source` at the moment
    of launch and aims at `target`'s predicted position at arrival. We
    scan candidate arrival turns t_arr in [1, horizon] and solve, for
    each, the (small) fixed point in launch turn:

        D = | source(t_arr - ETA) - target(t_arr) |
        ETA = ceil(D / fleet_speed)
        launch = t_arr - ETA

    For static sources the fixed point is trivial. For orbiting sources
    two iterations are plenty (orbital speeds << fleet speeds).

    Returns the `LaunchPlan` with the smallest travel distance, or None
    if no `t_arr` in [1, horizon] has a non-negative launch offset
    (target is unreachable within the horizon — too far for the given
    fleet size).
    """
    v = fleet_speed(ships)
    if v <= 0:
        return None
    best: LaunchPlan | None = None
    for t_arr in range(1, horizon + 1):
        tx, ty = position_at(target, omega, t_arr)
        # Initial guess: source position == source NOW (works exactly for
        # static sources). Iterate to refine for orbiting sources.
        launch_guess = t_arr - 1  # any non-negative; refined below
        sx, sy = position_at(source, omega, 0)
        d = math.hypot(sx - tx, sy - ty)
        eta = max(1, int(math.ceil(d / v)))
        launch = t_arr - eta
        if launch < 0:
            continue
        for _ in range(inner_iters):
            sx, sy = position_at(source, omega, launch)
            d = math.hypot(sx - tx, sy - ty)
            eta_new = max(1, int(math.ceil(d / v)))
            if eta_new == eta:
                break
            eta = eta_new
            launch = t_arr - eta
            if launch < 0:
                break
        if launch < 0:
            continue
        if best is None or d < best.distance:
            best = LaunchPlan(
                launch_offset=launch,
                arrival_turn=t_arr,
                eta=eta,
                distance=d,
                source_pos=(sx, sy),
                target_pos=(tx, ty),
                aim_angle_rad=math.atan2(ty - sy, tx - sx),
            )
    return best


# ---------------------------------------------------------------------------
# Convenience: compare against the "fire NOW at instantaneous target" baseline
# ---------------------------------------------------------------------------


def naive_launch_plan(
    source,
    target,
    omega: float,
    ships: int | float,
) -> LaunchPlan:
    """The straight-up `launch NOW aimed at the target's predicted
    position at ETA' alternative. Useful as the baseline that
    `best_launch_plan` must beat for the phase-lead primitive to earn
    its keep at integration time.

    Iterates the same fixed point but pins launch_offset = 0.
    """
    v = fleet_speed(ships)
    sx, sy = position_at(source, omega, 0)
    # Initial guess: aim at target's current position.
    tx, ty = position_at(target, omega, 0)
    d = math.hypot(sx - tx, sy - ty)
    eta = max(1, int(math.ceil(d / v)))
    for _ in range(3):  # converges very fast
        tx, ty = position_at(target, omega, eta)
        d = math.hypot(sx - tx, sy - ty)
        eta_new = max(1, int(math.ceil(d / v)))
        if eta_new == eta:
            break
        eta = eta_new
    return LaunchPlan(
        launch_offset=0,
        arrival_turn=eta,
        eta=eta,
        distance=d,
        source_pos=(sx, sy),
        target_pos=(tx, ty),
        aim_angle_rad=math.atan2(ty - sy, tx - sx),
    )
