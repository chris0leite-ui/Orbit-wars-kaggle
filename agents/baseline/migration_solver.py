"""Migration solver — closed-form own→own ship repositioning.

Ported from origin/claude/strategy-framework-design-OyoYR-rebased
(commit 175f790, Slice 9). Plan reference:
/root/.claude/plans/so-you-have-multiple-tranquil-papert.md Phase 3.

Matches the Claws ladder pattern documented from
audit/live-episodes/52827111/episode-77164175 (Claws's 4P win): the
opponent stockpiles ships on prod≥4 planets, sends a large fleet to
a recently-captured low-prod staging planet, then immediately
re-launches from there toward a richer target. The capture-EV is
unlocked by repositioning, not by the source's own capture range —
exactly the candidate class our proposer was missing.

Prior A/B (on OyoYR-rebased + chooser_differential) scored Wlo=0.102
n=16 — failed gate but layered on a different chooser stack. This
port re-evaluates on our modular baseline chooser after the Phase 1
+ Phase 2 fixes have landed.

Default-disabled via env var BASELINE_MIGRATION=1 so we can A/B
cleanly against the post-Phase-2 head.

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
# WAVE_LOOKAHEAD inlined locally — defined in OyoYR lib.world_model but not
# on this branch. Same semantic value (12 turns) as that branch.
WAVE_LOOKAHEAD = 12


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


# ---------------------------------------------------------------------------
# Phase 4: defensive migration toward threatened own planets.
# ---------------------------------------------------------------------------


def _defensive_deficit(P_dst, world, model, me: int) -> tuple[int, int, int]:
    """Return (threat_eta, total_enemy_force, my_garrison_at_eta) for P_dst.

    `threat_eta` is None if P_dst is not under threat.
    `total_enemy_force` sums every in-flight enemy fleet arriving within
    `threat_eta + WAVE_LOOKAHEAD` turns.
    `my_garrison_at_eta` is the predicted garrison at threat_eta from
    production and any in-flight friendly fleets already inbound.
    """
    threat_eta = model.time_to_enemy_threat(int(P_dst.id), int(me), world)
    if threat_eta is None:
        return (None, 0, int(P_dst.ships))

    window = int(threat_eta) + WAVE_LOOKAHEAD
    enemy_force = 0
    my_inbound = 0
    for (eta_arr, owner, sh) in model.ledger.get(int(P_dst.id), []):
        if int(owner) == int(me) and int(eta_arr) <= int(threat_eta):
            my_inbound += int(sh)
        elif int(owner) != int(me) and int(eta_arr) <= window:
            enemy_force += int(sh)
    my_garrison = int(P_dst.ships) + int(P_dst.production) * int(threat_eta) + my_inbound
    return (int(threat_eta), int(enemy_force), int(my_garrison))


def propose_defensive_migrations(world, model, me: int,
                                 *,
                                 gamma: float = DEFAULT_GAMMA,
                                 k_max: int = DEFAULT_K_MAX_MIGRATIONS,
                                 epsilon: float = MIGRATION_VALUE_EPSILON,
                                 safety: int = 2,
                                 ) -> list:
    """Emit closed-form defensive migrations TOWARD threatened own planets.

    Mirrors `propose_migrations` but inverted: each candidate is a
    rescue mission from a safe source to a threatened destination. The
    value is the present-value of the destination's remaining production
    if it survives (otherwise it would fall and that PV is lost).

    Constraints:
      - Destination must be under enemy threat AND have a non-trivial
        deficit (enemy force > current garrison + safety).
      - Source must have spare ships above its own threat reserve.
      - Reinforce must ARRIVE in time: mig_eta <= threat_eta.
      - Reinforce amount must close the deficit (else partial rescues
        bleed ships and still lose the planet).
      - One migration per source per turn (greedy dedup).
      - Cap at `k_max` candidates per turn.

    Returns the same 8-tuple shape as `propose_migrations` so the
    chooser can treat both code paths uniformly.
    """
    step = int(getattr(world, "step", 0) or 0)
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(me)]
    if len(my_planets) < 2:
        return []

    candidates: list = []
    for P_dst in my_planets:
        threat_eta, enemy_force, my_garrison = _defensive_deficit(
            P_dst, world, model, int(me),
        )
        if threat_eta is None or enemy_force <= 0:
            continue
        deficit = enemy_force + int(safety) - my_garrison
        if deficit <= 0:
            continue  # destination holds without help

        # PV of saving this planet — its production over remaining turns.
        time_remaining = EPISODE_END - step - int(threat_eta)
        if time_remaining <= 0:
            continue
        pv_save = pv_horizon(step, int(threat_eta), gamma=float(gamma),
                             t_total=EPISODE_END)
        save_value = float(int(P_dst.production)) * float(pv_save)
        if save_value <= 0:
            continue

        for P_src in my_planets:
            if int(P_src.id) == int(P_dst.id):
                continue
            reserve = _threat_reserve(P_src, world, model, int(me))
            available = int(P_src.ships) - int(reserve)
            # Need to cover the deficit; partial reinforces don't help
            # (the planet still falls and the ships are wasted).
            if available < deficit:
                continue

            dist = math.hypot(
                float(P_src.x) - float(P_dst.x),
                float(P_src.y) - float(P_dst.y),
            )
            flight = max(0.0, dist - float(P_src.radius)
                                 - float(P_dst.radius) - 0.1)
            ships_send = max(MIN_MIGRATION_SHIPS, int(deficit))
            spd = fleet_speed(int(ships_send))
            if spd <= 0:
                continue
            mig_eta = int(math.ceil(flight / spd))
            if mig_eta <= 0 or mig_eta > int(threat_eta):
                continue  # too slow to arrive in time

            angle = math.atan2(
                float(P_dst.y) - float(P_src.y),
                float(P_dst.x) - float(P_src.x),
            )
            value = save_value  # PV-discount already baked into save_value
            if value <= epsilon:
                continue
            candidates.append((
                float(value),
                P_src,
                P_dst,
                int(ships_send),
                float(angle),
                int(mig_eta),
                int(mig_eta + 2),
                0,  # wait_N — fire-now rescue
            ))

    candidates.sort(key=lambda c: -float(c[0]))
    used_srcs: set = set()
    used_dsts: set = set()
    out: list = []
    for c in candidates:
        sid = int(c[1].id)
        tid = int(c[2].id)
        if sid in used_srcs or tid in used_dsts:
            continue
        used_srcs.add(sid)
        used_dsts.add(tid)
        out.append(c)
        if len(out) >= int(k_max):
            break
    return out
