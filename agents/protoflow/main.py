"""protoflow — flow-field "converging streams" target selector (PROBE, not a submission).

Tests the PI's hypothesis: treat the board as a value field and let many of our
planets converge winning-sized waves onto the SAME target. The mechanical reason
this should work is combat rule 1 — same-owner fleets that arrive on the same
turn are SUMMED before the fight — so a synchronized cohort can capture a target
no single planet could take alone (the teamwork-coalition idea generalized from
exactly-two planets to an N-source arrival-turn cohort).

Design discipline (why this is NOT the three dead analytic scorers):
- The value field never collapses to zero: a target's `winnability` has a hard
  positive floor (WIN_FLOOR), so there is always *some* pull toward the enemy's
  softest planet. No death spiral.
- The field stays reactive: a target's pull is discounted by a reach race — how
  much sooner we can reach it than the nearest enemy planet can. Win the race →
  strong pull; lose it → faint pull. This is what keeps "don't overexpose".
- Every launch is a winning-sized wave (solo, or a same-arrival-turn cohort that
  meets the capture floor). We never under-send into a defended target (no bounce
  waste). When nothing is winnable, sources HOLD and accumulate.

Anti-inertia is the thing the probe measures: `scripts/protoflow_probe.py` reads
the per-turn trace below to report launches/game, idle-fraction, and
convergence-turns/game — the midgame-goes-flat failure that killed the ancestors
shows up as a high idle-fraction.

This file imports lib/* directly (fine for local A/B via fast.py /
kaggle_environments; it is NOT bundled for Kaggle).
"""
from __future__ import annotations

import math
from collections import defaultdict

from lib.intent import World
from lib.aim import aim_orbiting
from lib.fleet import travel_time
from lib.trajectory import predict_fleet_fate

EPISODE_STEPS = 500
WIN_FLOOR = 0.10       # winnability never reaches zero -> field never goes flat
RACE_SCALE = 5.0       # turns; steepness of the reach-race sigmoid
THREAT_ANGLE = 0.45    # rad (~26 deg); enemy fleet is "incoming" if heading within this of the bearing
THREAT_RANGE = 60.0    # board units; only nearby enemy fleets count as a committed threat

# Module-global per-game trace for the probe runner (convergence instrumentation).
# Each entry: {"step", "launches": [(src_id, tgt_id, ships)], "idle": bool, "sources": int}.
_TRACE: list[dict] = []


def reset_trace() -> None:
    _TRACE.clear()


def get_trace() -> list[dict]:
    return list(_TRACE)


def _sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _parse_fleets(obs):
    raw = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    return [tuple(f) for f in raw]  # (id, owner, x, y, angle, from_planet_id, ships)


def _incoming_threat(planet, enemy_fleets) -> float:
    """Sum of enemy-fleet ships currently heading at `planet` from within range."""
    total = 0.0
    for f in enemy_fleets:
        fx, fy, fang, fships = f[2], f[3], f[4], f[6]
        d = math.hypot(planet.x - fx, planet.y - fy)
        if d > THREAT_RANGE:
            continue
        bearing = math.atan2(planet.y - fy, planet.x - fx)
        diff = abs(((bearing - fang + math.pi) % (2.0 * math.pi)) - math.pi)
        if diff < THREAT_ANGLE:
            total += fships
    return total


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    me = world.my_id
    step = world.step
    omega = world.omega
    remain = max(1, EPISODE_STEPS - step)

    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if p.owner == me]
    enemy_planets = [p for p in planets if p.owner != me and p.owner != -1]
    enemy_fleets = [f for f in _parse_fleets(obs) if f[1] != me and f[1] != -1]

    if not my_planets:
        _TRACE.append({"step": step, "launches": [], "idle": True, "sources": 0})
        return []

    # --- spare / hold: a planet holds enough to cover its committed incoming threat ---
    threat = {p.id: _incoming_threat(p, enemy_fleets) for p in my_planets}
    spare = {p.id: max(0, int(p.ships) - int(math.ceil(threat[p.id]))) for p in my_planets}

    offense = [p for p in planets if p.owner != me]                 # every non-owned planet
    defense = [p for p in my_planets if threat[p.id] > p.ships]     # my planets out-threatened

    # --- aim cache: (angle, arrival_xy, eta) for each (source, target) ---
    aims: dict[tuple[int, int], object] = {}

    def aim_for(s, t):
        key = (s.id, t.id)
        if key in aims:
            return aims[key]
        if s.id == t.id:
            aims[key] = None
            return None
        ships_nom = max(1, spare.get(s.id, 0) or 1)
        res = aim_orbiting((s.x, s.y), s.radius, t, t.radius, ships_nom, omega)
        aims[key] = res
        return res

    def our_best_eta(t):
        best = None
        for s in my_planets:
            a = aim_for(s, t)
            if a is not None and (best is None or a[2] < best):
                best = a[2]
        return best

    def enemy_best_eta(t):
        best = None
        for ep in enemy_planets:
            tt = travel_time((ep.x, ep.y), (t.x, t.y), max(1, int(ep.ships)))
            if best is None or tt < best:
                best = tt
        return best

    # --- pull per offense target: value * reach-race winnability (floored) ---
    pull: dict[int, float] = {}
    for t in offense:
        oe = our_best_eta(t)
        if oe is None:
            continue  # unreachable this turn
        ee = enemy_best_eta(t)
        win = 1.0 if ee is None else WIN_FLOOR + (1.0 - WIN_FLOOR) * _sigmoid((ee - oe) / RACE_SCALE)
        val = t.production * remain
        if t.owner != -1:           # flipping an enemy planet denies + gains (roi_enemy2x lesson)
            val *= 2.0
        pull[t.id] = val * win

    # Priority order: defend first, then offense by descending pull.
    ordered = [("def", d) for d in defense]
    ordered += [("off", t) for t in sorted((t for t in offense if t.id in pull),
                                            key=lambda t: -pull[t.id])]

    def capture_floor(t, arrival_turn):
        grow = 0.0 if t.owner == -1 else t.production * arrival_turn   # neutrals don't produce
        return int(math.ceil(t.ships + grow)) + 1

    moves: list[list] = []
    launches: list[tuple] = []

    for kind, t in ordered:
        cands = []
        for s in my_planets:
            if spare.get(s.id, 0) <= 0:
                continue
            a = aim_for(s, t)
            if a is None:
                continue
            cands.append((s, a[0], a[2]))  # (source, angle, eta)
        if not cands:
            continue

        # Group candidates by arrival turn: combat sums same-turn same-owner arrivals.
        by_turn: dict[int, list] = defaultdict(list)
        for s, angle, eta in cands:
            by_turn[int(math.ceil(eta))].append((s, angle, eta))

        if kind == "def":
            need = int(math.ceil(threat[t.id] - t.ships)) + 1
            if need <= 0:
                continue

        # Earliest arrival-turn cohort whose combined spare meets the floor wins the target.
        committed = None
        for turn in sorted(by_turn):
            cohort = by_turn[turn]
            floor = need if kind == "def" else capture_floor(t, turn)
            if sum(spare[s.id] for s, _, _ in cohort) >= floor:
                committed = (cohort, floor)
                break
        if committed is None:
            continue

        cohort, floor = committed
        cohort.sort(key=lambda c: c[2])  # fill nearest-first
        needed = floor
        cohort_arrival = int(math.ceil(cohort[0][2]))
        for s, angle, eta in cohort:
            if needed <= 0:
                break
            send = min(spare[s.id], needed)
            if send <= 0:
                continue
            if predict_fleet_fate(s, t, angle, int(send), world).outcome != "target":
                continue  # path blocked (sun / wrong planet / oob) -> drop this leg
            moves.append([s.id, angle, int(send)])
            launches.append({
                "src": s.id, "tgt": t.id, "ships": int(send),
                "eta": round(eta, 1), "arrive_turn": cohort_arrival,
                "dist": round(math.hypot(t.x - s.x, t.y - s.y), 1),
                "tgt_owner": t.owner, "kind": kind, "floor": floor,
            })
            spare[s.id] -= send
            needed -= send

    _TRACE.append({
        "step": step,
        "launches": launches,
        "idle": len(launches) == 0,
        "sources": len(my_planets),
        "my_planets": len(my_planets),
        "my_ships": int(sum(p.ships for p in my_planets)),
    })
    return moves
