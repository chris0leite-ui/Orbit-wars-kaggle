"""Post-chooser relay / forward-staging pass (Pass-2).

For each of MY planets that didn't fire this turn but holds a real
garrison, find the (relay R, target T) pair with the shortest
**two-leg** path S → R → T where:

  * R is one of MY planets (the waypoint)
  * T is an opponent or neutral planet (the eventual capture)

S launches at R now; R is expected to launch at T on a later turn
once the merged garrison can clear T's defender. Pass-2 only commits
leg-1; leg-2 is rediscovered organically on R's future turn.

Gates (Phase A — no explicit EV gate, force-sufficiency only):
  * S not in `used_srcs` (pass-1 / earlier post-passes didn't claim it).
  * S.ships > production-scaled reserve (same convention as the
    existing drain passes — `max(production*5, 10)`).
  * ships_to_send >= BASELINE_MIN_SHIPS_LAUNCH (env-var read at call).
  * leg-1 cheap ETA <= BASELINE_MAX_ETA (env-var read at call).
  * leg-1 fate passes `predict_fleet_fate` (outcome="target",
    hit_planet_id == R.id — no sun / OOB / wrong-planet hits).
  * R is predicted to still be mine at leg-1 arrival
    (`model.owner_at(R.id, leg1_eta) == my_id`).
  * Leg-2 segment doesn't pass through the sun (cheap segment-to-
    centre distance check; full fate not needed because R re-validates
    on its future turn).
  * Force-sufficiency: merged garrison at R after leg-1 arrival must
    exceed T's predicted defender at total_eta.

Disabled by default; opt-in via `BASELINE_RELAY=1` env var.
"""

from __future__ import annotations

import math
import os

from agents.baseline.spearhead import SpearheadContext, cos_alignment
from lib.fleet import speed as fleet_speed
from lib.geometry import BOARD_SIZE, CENTER, SUN_RADIUS
from lib.orbit import is_orbiting, predict_relative
from lib.trajectory import predict_fleet_fate


# Float cushion around the sun for the cheap leg-2 segment check. Leg-1
# is validated by full predict_fleet_fate; leg-2 only needs a rough
# "don't pick a path that obviously hits the sun" filter because R will
# re-validate on its future turn.
LEG2_SUN_CUSHION = 2.0

# Reserve floor matches drain_stagnant_rear / emit_sniper_strikes:
# leave behind enough to defend a few production cycles.
RESERVE_MULT = 5
RESERVE_FLOOR = 10

# Limit fate-validated candidate relays per source. Empirically, the
# nearest few friendly planets dominate the "shortest 2-leg path"
# optimum — including all N-1 friendlies would N² scale the expensive
# predict_fleet_fate calls and blow the 1000ms env wallclock cap.
# Env-overridable via BASELINE_RELAY_K_RELAYS.
DEFAULT_K_NEAREST_RELAYS = 5

# Relay-specific fleet-speed floor. Relay-staged fleets travel two legs;
# leg-1 must move fast enough that R hasn't moved on by arrival. 2.5
# board units/turn requires ~30 ships. Stops production-2 sources from
# shipping 4-5 ship fleets at speed ~1.5 that lose force-sufficiency.
DEFAULT_MIN_RELAY_SPEED = 2.5


def _segment_distance_to_center(ax: float, ay: float,
                                bx: float, by: float) -> float:
    """Shortest distance from segment (ax,ay)->(bx,by) to (CENTER, CENTER)."""
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 0.0:
        return math.hypot(ax - CENTER, ay - CENTER)
    t = ((CENTER - ax) * dx + (CENTER - ay) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(px - CENTER, py - CENTER)


def emit_relay_forward(moves, planets, my_id: int, world, model,
                       ctx: SpearheadContext | None = None) -> list:
    """Append two-leg relay launches for idle sources with a downstream
    target. Returns moves + new launches.

    Each move is `[src_id, angle, ships]`. Idempotent: re-derives
    `used_srcs` from `moves` so it doesn't double-ship.

    `ctx` (optional): SpearheadContext from `agents.baseline.spearhead`.
    When provided and BASELINE_RELAY_SPEARHEAD=1, the R-selection
    tiebreak gets a directional bonus that prefers large relays aligned
    with each source's nearest-opponent direction. Default fall-through
    matches the legacy min-total-eta tiebreak.
    """
    if os.environ.get("BASELINE_RELAY", "0") != "1":
        return moves

    max_eta = int(os.environ.get("BASELINE_MAX_ETA", "40"))
    min_ships = int(os.environ.get("BASELINE_MIN_SHIPS_LAUNCH", "2"))
    k_relays = int(os.environ.get("BASELINE_RELAY_K_RELAYS",
                                  str(DEFAULT_K_NEAREST_RELAYS)))
    min_speed = float(os.environ.get("BASELINE_RELAY_MIN_SPEED",
                                     str(DEFAULT_MIN_RELAY_SPEED)))
    spearhead_on = (
        ctx is not None
        and os.environ.get("BASELINE_RELAY_SPEARHEAD", "0") == "1"
    )
    spearhead_alpha = float(os.environ.get(
        "BASELINE_RELAY_SPEARHEAD_ALPHA", "1.5",
    ))

    used_srcs: set[int] = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass

    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return moves  # need at least one relay candidate distinct from S
    foreign_planets = [p for p in planets if int(p.owner) != my_id]
    if not foreign_planets:
        return moves  # no target → nothing to relay toward

    omega = float(getattr(world, "omega", 0.0) or 0.0)

    extras = []
    for src in my_planets:
        if int(src.id) in used_srcs:
            continue
        prod = int(src.production)
        reserve = max(prod * RESERVE_MULT, RESERVE_FLOOR)
        ships_to_send = int(src.ships) - reserve
        if ships_to_send < min_ships:
            continue
        if fleet_speed(ships_to_send) < min_speed:
            continue
        if model.incoming_enemy_eta(int(src.id), my_id) is not None:
            continue

        best_score = None
        best_relay = None
        best_angle = None
        src_opp_xy = ctx.nearest_opp_xy.get(int(src.id)) if spearhead_on else None

        # K-nearest pre-filter: rank friendly relay candidates by distance,
        # keep the closest k_relays. Cuts the expensive predict_fleet_fate
        # cost by a factor relative to iterating all my_planets.
        speed_sr = fleet_speed(ships_to_send)
        relay_candidates: list[tuple[float, object, int]] = []
        for relay in my_planets:
            if int(relay.id) == int(src.id):
                continue
            d_sr = math.hypot(float(relay.x) - float(src.x),
                              float(relay.y) - float(src.y))
            cheap_eta1 = int(math.ceil(d_sr / speed_sr)) if speed_sr > 0 else 999
            if cheap_eta1 > max_eta:
                continue
            relay_candidates.append((d_sr, relay, cheap_eta1))
        relay_candidates.sort(key=lambda t: t[0])
        relay_candidates = relay_candidates[:k_relays]

        for d_sr, relay, cheap_eta1 in relay_candidates:
            angle_sr = math.atan2(float(relay.y) - float(src.y),
                                  float(relay.x) - float(src.x))
            try:
                fate1 = predict_fleet_fate(src, relay, angle_sr,
                                           ships_to_send, world)
            except Exception:
                continue
            if fate1.outcome != "target" or fate1.hit_planet_id != relay.id:
                continue
            leg1_eta = int(fate1.step)
            if leg1_eta > max_eta:
                continue

            # R must still be mine at arrival.
            relay_owner = model.owner_at(int(relay.id), leg1_eta)
            if relay_owner is not None and relay_owner != my_id:
                continue

            # Predicted garrison at R when our ships arrive (same owner →
            # ships merge). Fallback if model is None outside its horizon.
            r_garrison = model.ships_at(int(relay.id), leg1_eta)
            if r_garrison is None:
                r_garrison = float(relay.ships) + leg1_eta * float(relay.production)
            merged_force = float(r_garrison) + float(ships_to_send)

            # R's position at our leg-1 arrival (used as leg-2 origin).
            r_tuple = [relay.id, relay.owner, relay.x, relay.y,
                       relay.radius, relay.ships, relay.production]
            if is_orbiting(r_tuple) and omega != 0.0 and leg1_eta > 0:
                r_arr_x, r_arr_y = predict_relative(r_tuple, omega, leg1_eta)
            else:
                r_arr_x, r_arr_y = float(relay.x), float(relay.y)

            speed_rt = fleet_speed(int(round(merged_force)))
            if speed_rt <= 0:
                continue

            # Spearhead front-bonus: per-(src, relay), constant across
            # this relay's target loop. Rewards picking a high-production
            # relay aligned with src's nearest-opp direction. Rectified
            # cosine: rear relays get no bonus (not a penalty — ETA
            # already penalises them).
            front_bonus = 0.0
            if spearhead_on and src_opp_xy is not None:
                front_bonus = spearhead_alpha * float(relay.production) * \
                    cos_alignment(
                        float(src.x), float(src.y),
                        r_arr_x, r_arr_y,
                        src_opp_xy[0], src_opp_xy[1],
                    )

            for tgt in foreign_planets:
                # First-pass leg-2 ETA from T's current position.
                d_rt0 = math.hypot(float(tgt.x) - r_arr_x,
                                   float(tgt.y) - r_arr_y)
                approx_eta2 = int(math.ceil(d_rt0 / speed_rt))

                # Refine using T's position at total ETA (single iteration).
                t_tuple = [tgt.id, tgt.owner, tgt.x, tgt.y,
                           tgt.radius, tgt.ships, tgt.production]
                if is_orbiting(t_tuple) and omega != 0.0:
                    t_arr_x, t_arr_y = predict_relative(
                        t_tuple, omega, leg1_eta + approx_eta2,
                    )
                else:
                    t_arr_x, t_arr_y = float(tgt.x), float(tgt.y)
                d_rt = math.hypot(t_arr_x - r_arr_x, t_arr_y - r_arr_y)
                leg2_eta = int(math.ceil(d_rt / speed_rt))
                if leg2_eta <= 0:
                    continue
                total_eta = leg1_eta + leg2_eta
                score = float(total_eta) - front_bonus
                if best_score is not None and score >= best_score:
                    continue

                # Cheap sun-crossing rejection for leg-2.
                sun_d = _segment_distance_to_center(
                    r_arr_x, r_arr_y, t_arr_x, t_arr_y,
                )
                if sun_d < SUN_RADIUS + LEG2_SUN_CUSHION:
                    continue
                # OOB rejection (target outside board → bad relay choice).
                if (t_arr_x < 0.0 or t_arr_x > BOARD_SIZE
                        or t_arr_y < 0.0 or t_arr_y > BOARD_SIZE):
                    continue

                # Force-sufficiency at T at total_eta.
                t_def = model.ships_at(int(tgt.id), total_eta)
                if t_def is None:
                    if int(tgt.owner) == -1:
                        t_def = float(tgt.ships)
                    else:
                        t_def = float(tgt.ships) + total_eta * float(tgt.production)
                if merged_force <= float(t_def):
                    continue

                best_score = score
                best_relay = relay
                best_angle = angle_sr

        if best_relay is None:
            continue

        extras.append([int(src.id), float(best_angle), int(ships_to_send)])
        used_srcs.add(int(src.id))

    return list(moves) + extras
