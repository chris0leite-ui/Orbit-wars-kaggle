"""Compound-aware augmentations for v21_compound's candidate scoring.

Three responsibilities, all consumed inside the cheap-pre-rank loop:

1. `fleet_path_safe(src, angle, ships, target_xy, world)` — drops
   sun-bound and out-of-bounds candidates at proposer time, BEFORE they
   enter the validate stage. Currently sun_avoid only runs at realize
   time (mechanism layer), so the chooser wastes validate-budget slots
   on candidates that will be dropped at emit. Pre-filtering also
   addresses PI's observation that recent strategies have been flying
   into the sun — by ensuring such candidates never reach the chooser.

   Cheap: single point-to-segment distance test (microseconds), not the
   ~1-2 ms ray-cast `predict_fleet_fate` would cost.

2. `compound_bonus(src, tgt, ships, eta, world, model, me, anchor_xy, mission_book)`
   — additive score component combining:
     * `chain_bonus`     : if capturing this target unlocks a cheap
                           follow-on capture from the target within 15
                           turns, add a fraction of that follow-on's ECV.
     * `rotation_bonus`  : positive if the planet drifts toward our
                           cluster centroid over 30 turns.
     * `carry_bonus`     : if (src, tgt) matches a live commit in
                           mission_book, add a small stability bonus.

Sized so the SUM of bonuses is at most ~half the cheap_marginal_value
they augment — they tilt the ranking, never override it.

Design principle (Rule 40): all filters are model-derived. There is no
hard MAX_ETA cap; if `pv_horizon` says the target is too late, ECV
drops naturally. There is no hard MIN_FLEET_SIZE bump; if the fleet
would lose combat, model.ships_at returns a higher predicted defender
than our ships. There is no DANGER3 multiplier; rotation/chain capture
the same intuition via geometric prediction.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.geometry import path_clears_sun, BOARD_SIZE, SUN_RADIUS
from lib.geo.rotation import rotation_alignment
from lib.scoring import pv_horizon


# Episode horizon for PV-discount inside chain estimate.
EPISODE_STEPS = 500

# Bonus weights. Tunable via env vars for ablation. Keep small so they
# tilt the cheap-marginal-value ranking without overriding it.
CHAIN_BONUS_WEIGHT = 0.30           # fraction of follow-on ECV added
CHAIN_LOOKAHEAD_TURNS = 15          # how far to look after the capture
ROTATION_BONUS_WEIGHT = 0.02        # × production × alignment in [-1, +1]
ROTATION_HORIZON = 30
PATH_SUN_SAFETY = 0.5               # match lib/trajectory.py's SUN_SAFETY


# --- 1. Path safety -------------------------------------------------------


def fleet_path_safe(src, angle: float, ships: int, eta: int) -> bool:
    """Cheap pre-filter: drop candidates whose straight-line trajectory
    crosses the sun or leaves the board.

    The fleet flies a STRAIGHT line at `fleet_speed(ships)` per step
    regardless of where the target IS at eta — combat resolves when
    target's position coincides with the fleet's position. So the path
    we care about is spawn -> spawn + speed*eta*direction.

    Returns False for:
      - segment crossing within SUN_RADIUS + PATH_SUN_SAFETY of the sun
      - endpoint outside the [0, BOARD_SIZE] box
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    spawn = (
        float(src.x) + cos_a * (float(src.radius) + 0.1),
        float(src.y) + sin_a * (float(src.radius) + 0.1),
    )
    speed_val = fleet_speed(int(ships))
    if speed_val <= 0:
        return False
    end_x = spawn[0] + cos_a * speed_val * float(eta)
    end_y = spawn[1] + sin_a * speed_val * float(eta)
    arrival = (end_x, end_y)
    if not path_clears_sun(spawn, arrival, safety=PATH_SUN_SAFETY):
        return False
    if end_x < 0.0 or end_x > BOARD_SIZE or end_y < 0.0 or end_y > BOARD_SIZE:
        return False
    return True


# --- 2. Chain bonus -------------------------------------------------------


def _candidate_post_capture(tgt, world, me, capture_step):
    """If we capture `tgt` at `capture_step`, what's the next target it
    could afford to capture from within CHAIN_LOOKAHEAD_TURNS?

    Returns the best (chain_target, chain_eta, chain_pred_def, chain_ships)
    tuple or None if no chain candidate qualifies.

    "Afford" = the post-capture src has enough ships (the captured
    surplus + production accrued during chain_eta) to overcome the
    chain target's predicted defenders. Uses WorldModel.ships_at for
    the chain target's defender prediction.
    """
    tgt_x = float(tgt.x)
    tgt_y = float(tgt.y)
    tgt_prod = float(tgt.production)
    best = None
    # Iterate over non-mine planets within a Manhattan-ish neighborhood.
    # The post-capture src is at tgt's position; we only consider
    # planets reachable inside CHAIN_LOOKAHEAD_TURNS at speed ~2 (the
    # speed of a small follow-on fleet of 5-15 ships ≈ 1.5-2.5 u/turn).
    # That means we care about planets within ~40 units.
    for p in world.planets_by_id.values():
        if int(p.owner) == me or int(p.id) == int(tgt.id):
            continue
        dx = float(p.x) - tgt_x
        dy = float(p.y) - tgt_y
        d = math.hypot(dx, dy)
        if d > 40.0:
            continue
        # Cheap ETA estimate: assume fleet of ~10 ships → speed ≈ 1.7
        chain_eta = max(1, int(math.ceil(d / 1.7)))
        if chain_eta > CHAIN_LOOKAHEAD_TURNS:
            continue
        # Score chain target by its PV value × production (cheap-marginal
        # equivalent, ignoring the actual ship arithmetic).
        chain_pv = pv_horizon(
            int(world.step) + int(capture_step),
            int(chain_eta),
            gamma=0.99,
            t_total=EPISODE_STEPS,
        )
        chain_ecv = float(p.production) * float(chain_pv)
        if best is None or chain_ecv > best[0]:
            best = (chain_ecv, p, chain_eta)
    if best is None:
        return None
    return best


# --- 3. Composite bonus call (single function consumed by the agent) -----


def compound_bonus(src, tgt, ships, eta, world, model, me,
                   anchor_xy, mission_book,
                   wait_N: int = 0) -> float:
    """Sum of (rotation + chain + carryforward) bonuses for one candidate.

    Each component is bounded; the total is intended to add ≤ ~50% of
    the cheap_marginal_value it augments. The chain and rotation pieces
    are model-derived (Rule 40 — no constant caps); the carryforward
    piece is a TTL-decayed score-stability nudge.

    Returns 0.0 for the BOUNCE branch (ships ≤ predicted defenders) so
    we never reward a doomed capture for being "in the right place."
    """
    # Skip bonus computation for reinforce candidates — the carryforward
    # bonus still applies but rotation/chain don't (we already own it).
    if int(tgt.owner) == me:
        return mission_book.carryforward_bonus(int(src.id), int(tgt.id))

    arrival_step = wait_N + eta
    pred_owner = model.owner_at(int(tgt.id), arrival_step)
    pred_ships = float(model.ships_at(int(tgt.id), arrival_step) or 0.0)
    # Only credit bonuses for actual captures, not bounces.
    if pred_owner == me or ships <= pred_ships:
        return 0.0

    omega = float(world.omega)

    # Rotation alignment: planets drifting toward our cluster centroid
    # are easier to defend → bonus. Scaled by production.
    align = rotation_alignment(
        [tgt.id, tgt.owner, tgt.x, tgt.y, tgt.radius, tgt.ships, tgt.production],
        omega, anchor_xy, horizon=ROTATION_HORIZON,
    )
    rotation_bonus = ROTATION_BONUS_WEIGHT * float(tgt.production) * float(align)

    # Chain bonus: if this capture unlocks a follow-on capture inside
    # CHAIN_LOOKAHEAD_TURNS, credit a fraction of the chain target's
    # ECV. Encourages picking captures whose POSITION matters, not
    # just whose intrinsic production matters.
    chain = _candidate_post_capture(tgt, world, me, capture_step=arrival_step)
    chain_bonus = 0.0
    if chain is not None:
        chain_ecv, _chain_target, _chain_eta = chain
        chain_bonus = CHAIN_BONUS_WEIGHT * chain_ecv

    # Carryforward bonus: if we already committed to this (src, tgt)
    # last turn, small stability nudge to prefer continuing the plan.
    carry_bonus = mission_book.carryforward_bonus(int(src.id), int(tgt.id))

    return rotation_bonus + chain_bonus + carry_bonus
