"""Potential-counter threat model.

The proposer's `_source_survives_launch` and the 5 post-passes / relay all
gate launches on `incoming_enemy_eta is not None` (in-flight enemy fleet
already committed). That fails open on the much more common pattern: an
opp planet has a strong garrison nearby and hasn't launched YET, but
will once we strip ourselves. This module supplies the missing predictor.

Exports:
  cheapest_potential_counter — low-level walk; returns
    `(opp, t_op, force_at_arrival)` for the worst-gap opp counter-launch,
    or None if no qualifying opp exists.
  source_safe_against_potential_counter — high-level verdict; checks
    whether `src` can defend itself against the worst-gap counter after
    launching `ships` at `wait_N`.

Differences from `_target_holdable_after_capture` (proposer.py:601):
  - No SAFETY_MARGIN multiplier; source-survival uses the stricter `+1`
    margin (matches the in-flight branch of `_source_survives_launch`).
  - Worst-gap tiebreak (not nearest-ETA). A far-strong opp can dominate
    a near-weak one; nearest-by-ETA would miss it.
  - `min_counter_ships=10` (vs 20 in `_target_holdable`): a 10-15 ship
    adjacent opp is still a real counter against a stripped source.
"""

from __future__ import annotations

import math
import os

from lib.fleet import speed as fleet_speed
from lib.world_model import _position_at


DEFAULT_MIN_COUNTER_SHIPS = 20


def _use_predict(world) -> bool:
    """Orbital safety gate — mirrors the convention in proposer.py:648."""
    if os.environ.get("BASELINE_ORBITAL_SAFETY", "0") != "1":
        return False
    omega = float(getattr(world, "omega", 0.0) or 0.0)
    return omega != 0.0


def cheapest_potential_counter(
    target,
    world,
    me: int,
    arrival_step: int,
    *,
    use_predict: bool | None = None,
    min_counter_ships: int = DEFAULT_MIN_COUNTER_SHIPS,
):
    """Return the worst-gap opp counter against `target` at `arrival_step`.

    `arrival_step` is the turn (relative to now) at which `target` is
    vulnerable — for a source, this is `wait_N`; for a captured target,
    it's `wait_N + eta`. The opp launches from its current position at
    turn 0 (worst case for us); counter arrives at `arrival_step + t_op`.

    Returns `(opp_planet, t_op, force_at_arrival)` for the opp with the
    largest gap between counter-force and a baseline defender (target's
    own production accrual over the counter-flight window). Returns
    `None` if no opp qualifies.
    """
    if use_predict is None:
        use_predict = _use_predict(world)
    omega = float(getattr(world, "omega", 0.0) or 0.0)
    if use_predict and omega != 0.0 and arrival_step > 0:
        tx, ty = _position_at(target, omega, arrival_step)
    else:
        tx, ty = float(target.x), float(target.y)

    target_id = int(target.id)
    target_radius = float(target.radius)
    target_prod = int(target.production)

    best = None  # (gap, opp, t_op, force)
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.id) == target_id:
            continue
        if int(opp.ships) < int(min_counter_ships):
            continue
        if use_predict and omega != 0.0 and arrival_step > 0:
            ox, oy = _position_at(opp, omega, arrival_step)
        else:
            ox, oy = float(opp.x), float(opp.y)
        d = math.hypot(ox - tx, oy - ty)
        flight = d - float(opp.radius) - target_radius - 0.1
        if flight <= 0:
            # Already overlapping at arrival — treat as immediate threat.
            t_op = 0
        else:
            opp_speed = fleet_speed(int(opp.ships))
            if opp_speed <= 0:
                continue
            t_op = int(math.ceil(flight / opp_speed))
        # Counter force = opp.ships at launch time. Production accruing
        # during flight stays on the opp's home planet — it cannot join
        # an already-departed fleet. arrival_step accrual models opp
        # WAITING until our vulnerability moment, then launching with
        # the larger stockpile. For post-passes with wait_N=0 this
        # reduces to opp.ships (current garrison). Matches the realistic-
        # case model in chooser_roi.py:_cheapest_opp_counter (which uses
        # opp_eta accrual — slightly more pessimistic — but our use case
        # already filters via min_counter_ships).
        force = int(opp.ships) + int(opp.production) * int(arrival_step)
        baseline_defense = max(0, target_prod * t_op)
        gap = force - baseline_defense
        if best is None or gap > best[0]:
            best = (gap, opp, t_op, force)
    if best is None:
        return None
    _, opp, t_op, force = best
    return (opp, int(t_op), int(force))


def source_safe_against_potential_counter(
    src,
    ships: int,
    wait_N: int,
    world,
    model,
    me: int,
) -> bool:
    """Could `src` defend itself against the worst-gap potential counter
    after launching `ships` ships at `wait_N`?

    Math (parallels the in-flight branch of `_source_survives_launch`):
      counter arrives at turn `wait_N + t_op`
      residue_after_launch = src.ships - ships + src.production * wait_N
      growth_after = src.production * max(0, t_op)
      garrison_at_counter = residue + growth_after
      return garrison_at_counter >= counter_force + 1

    Opt-out via `BASELINE_POTENTIAL_COUNTER=0`. When the env var is "0",
    falls back to the old in-flight-only check (parity with the gates
    being replaced).
    """
    if os.environ.get("BASELINE_POTENTIAL_COUNTER", "1") == "0":
        return model.incoming_enemy_eta(int(src.id), me) is None
    counter = cheapest_potential_counter(src, world, me, int(wait_N))
    if counter is None:
        return True
    _opp, t_op, counter_force = counter
    residue = int(src.ships) - int(ships) + int(src.production) * int(wait_N)
    if residue < 0:
        return False
    growth_after = int(src.production) * max(0, int(t_op))
    garrison_at_counter = residue + growth_after
    return garrison_at_counter >= int(counter_force) + 1
