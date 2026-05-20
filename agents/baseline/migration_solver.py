"""Migration solver — closed-form own→own ship repositioning.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §15.

Fills the missing candidate class identified by the Slice 8c
inspect (audit/2026-05-20-slice8c-validation.md): when no capture
candidate has positive Δ-favor, the chooser should consider
REPOSITIONING ships from a rear-line own planet to a front-line
own planet that has capture opportunities but lacks the ships.

The differential chooser correctly assigns Δ-favor = 0 to own→own
ship moves (no ownership change, no production change at the
leaf). That's mathematically right. But the *capture-EV unlocked*
by the migration is a separate positive quantity. This solver
computes that quantity in closed form and emits migration
candidates with it as the `cheap_delta` field. The chooser uses
that value directly for migration moves (special-cased).

Algorithm:
  1. For each of our planets P, compute the best capture-EV
     available NOW given P's current ship count
     (`compute_capture_ev_per_planet`).
  2. For each pair (P_src, P_dst) of our planets, compute the
     hypothetical capture-EV at P_dst if it had P_src's ships
     transferred in. The migration's value = ΔEV − migration
     cost, PV-discounted by the migration ETA.
  3. Emit candidates above an epsilon, sorted by value desc,
     dedup 1-per-source.
"""

from __future__ import annotations

import math

# Single-line imports below — bundler constraint (see proposer.py:71-76).
from lib.fleet import speed as fleet_speed
from lib.scoring import pv_horizon
from lib.world_model import WAVE_LOOKAHEAD


EPISODE_END: int = 500
DEFAULT_GAMMA: float = 0.99
MIN_MIGRATION_SHIPS: int = 5
MIGRATION_VALUE_EPSILON: float = 1.0
DEFAULT_K_MAX_MIGRATIONS: int = 5


def _best_capture_ev_for_planet(P, ship_count: int, world, model, me: int,
                                step: int, gamma: float = DEFAULT_GAMMA) -> float:
    """Closed-form best-capture-EV at planet P, GIVEN a hypothetical
    ship count (which may differ from P.ships).

    For each non-mine target T within reach:
      - eta = ceil(flight_distance / fleet_speed(ship_count))
      - capture_size = max(MIN, predicted_garrison_at_arrival + 1)
      - if ship_count >= capture_size: EV = T.production × pv_horizon
      - else: 0

    Returns the max EV across all feasible targets, or 0 if none.

    Used by both `compute_capture_ev_per_planet` (passing P.ships)
    and `propose_migrations` (passing augmented ship count).
    """
    if int(ship_count) < MIN_MIGRATION_SHIPS:
        return 0.0
    spd = fleet_speed(int(ship_count))
    if spd <= 0:
        return 0.0

    best_ev = 0.0
    for T in world.planets_by_id.values():
        if int(T.owner) == int(me):
            continue
        if int(T.id) == int(P.id):
            continue
        dist = math.hypot(float(P.x) - float(T.x), float(P.y) - float(T.y))
        flight = max(0.0, dist - float(P.radius) - float(T.radius) - 0.1)
        eta = int(math.ceil(flight / spd))
        time_remaining = EPISODE_END - int(step) - eta
        if time_remaining <= 0:
            continue
        pred_garrison = model.ships_at(int(T.id), eta)
        if pred_garrison is None:
            continue
        cap_size = max(MIN_MIGRATION_SHIPS,
                       int(math.ceil(float(pred_garrison))) + 1)
        if int(ship_count) < cap_size:
            continue
        pv = pv_horizon(int(step), int(eta), gamma=float(gamma),
                        t_total=EPISODE_END)
        gross_ev = float(int(T.production)) * float(pv)
        if gross_ev > best_ev:
            best_ev = gross_ev
    return best_ev


def compute_capture_ev_per_planet(world, model, me: int,
                                  gamma: float = DEFAULT_GAMMA) -> dict:
    """For each of our planets, the best capture-EV achievable NOW.

    Returns `{planet_id: ev}`. EV uses the planet's CURRENT ship
    count for both feasibility and fleet speed.
    """
    step = int(getattr(world, "step", 0) or 0)
    ev: dict = {}
    for P in world.planets_by_id.values():
        if int(P.owner) != int(me):
            continue
        ev[int(P.id)] = _best_capture_ev_for_planet(
            P, int(P.ships), world, model, int(me), step, gamma=gamma,
        )
    return ev


def _threat_reserve(P, world, model, me: int) -> int:
    """Ships P needs to keep for defending against in-flight enemy
    fleets. Mirrors `proposer._source_survives_launch`'s threat-force
    accounting (sum of opp ships inbound within
    `time_to_enemy_threat + WAVE_LOOKAHEAD`).

    Returns 0 if no threat. The "reserve" is what we don't drain
    when proposing migrations from this planet.
    """
    threat_eta = model.time_to_enemy_threat(int(P.id), int(me), world)
    if threat_eta is None:
        return 0
    threat_force = 0
    for (eta_arr, owner, sh) in model.ledger.get(int(P.id), []):
        if int(owner) == int(me):
            continue
        if int(eta_arr) <= int(threat_eta) + WAVE_LOOKAHEAD:
            threat_force += int(sh)
    return int(threat_force)


def propose_migrations(world, model, me: int,
                       *,
                       gamma: float = DEFAULT_GAMMA,
                       k_max: int = DEFAULT_K_MAX_MIGRATIONS,
                       epsilon: float = MIGRATION_VALUE_EPSILON,
                       ) -> list:
    """Emit closed-form own→own migration candidates.

    Returns a list of tuples matching the proposer's prerank format:
        `(value, src, tgt, ships, angle, eta, horizon, wait_N=0)`
    where `value` is the migration's ΔCapture-EV (the score the
    chooser uses directly for migration moves).

    Constraints:
      - At least 2 of our planets must exist.
      - Source planet must have ships above its threat-reserve.
      - Destination must NOT be under enemy threat (those are
        defensive reinforces, handled separately).
      - Migration must increase the destination's capture-EV
        (otherwise migrating is wasteful).
      - Migration value (PV-discounted) must exceed `epsilon`.
      - One migration per source per turn (greedy dedup).
      - Cap at `k_max` migrations per turn.
    """
    step = int(getattr(world, "step", 0) or 0)
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(me)]
    if len(my_planets) < 2:
        return []

    base_ev = compute_capture_ev_per_planet(world, model, int(me), gamma)

    candidates: list = []
    for P_src in my_planets:
        reserve = _threat_reserve(P_src, world, model, int(me))
        available = int(P_src.ships) - int(reserve)
        if available < MIN_MIGRATION_SHIPS:
            continue

        src_ev = base_ev.get(int(P_src.id), 0.0)

        for P_dst in my_planets:
            if int(P_dst.id) == int(P_src.id):
                continue
            # Skip destinations under enemy threat — those are
            # defensive-reinforce territory (W2 / proposer handles).
            if model.time_to_enemy_threat(int(P_dst.id), int(me), world) is not None:
                continue

            # Migration ETA
            dist = math.hypot(
                float(P_src.x) - float(P_dst.x),
                float(P_src.y) - float(P_dst.y),
            )
            flight = max(0.0, dist - float(P_src.radius)
                                 - float(P_dst.radius) - 0.1)
            spd = fleet_speed(int(available))
            if spd <= 0:
                continue
            mig_eta = int(math.ceil(flight / spd))
            if mig_eta <= 0:
                continue
            # Don't migrate to a destination that won't have time to
            # use the ships before the game ends.
            time_remaining = EPISODE_END - step - mig_eta
            if time_remaining <= 0:
                continue

            # EV at destination: pre vs post migration.
            pre_ev = float(base_ev.get(int(P_dst.id), 0.0))
            post_ships = int(P_dst.ships) + int(available)
            post_ev = _best_capture_ev_for_planet(
                P_dst, post_ships, world, model, int(me), step, gamma=gamma,
            )
            delta_ev = post_ev - pre_ev
            if delta_ev <= 0:
                continue  # destination doesn't unlock new value

            # PV-discount the migration's eta (we wait `mig_eta` ticks
            # before the dst can use the ships).
            pv_mig = pv_horizon(step, mig_eta, gamma=float(gamma),
                                t_total=EPISODE_END)
            # Opportunity cost: src's own EV is lost.
            cost = float(src_ev)
            value = (delta_ev - cost) * float(pv_mig)

            if value <= epsilon:
                continue

            angle = math.atan2(
                float(P_dst.y) - float(P_src.y),
                float(P_dst.x) - float(P_src.x),
            )
            # Tuple format mirrors `proposer.propose()` output:
            # (value, src, tgt, ships, angle, eta, horizon, wait_N)
            candidates.append((
                float(value),
                P_src,
                P_dst,
                int(available),
                float(angle),
                int(mig_eta),
                int(mig_eta + 2),
                0,  # wait_N
            ))

    # Greedy 1-per-source dedup, sorted by value desc.
    candidates.sort(key=lambda c: -float(c[0]))
    used_srcs: set = set()
    out: list = []
    for c in candidates:
        sid = int(c[1].id)
        if sid in used_srcs:
            continue
        used_srcs.add(sid)
        out.append(c)
        if len(out) >= int(k_max):
            break
    return out
