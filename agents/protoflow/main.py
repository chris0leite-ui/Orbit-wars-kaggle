"""protoflow — action-space value field (PROBE, not a submission).

PI reframe (2026-06-04): the field is NOT a spatial potential over planet
positions — it lives in the ACTION SPACE. The thing we score and follow a
gradient through is the set of moves we can actually make from the current
state: which launch trajectories are feasible right now, and which captures
matter most based on production and location. A position-pull sends senseless
tiny/far fleets because it does not know which actions are feasible or worth
it; an action field can never contain such a move, because it is not feasible
or not value-positive.

Build:
  1. FIELD = the feasible-action set. Reuse the champion's `propose()` — it
     returns every move we can make this turn, already trajectory-feasible and
     already sized so each one is a real capture (no 2-ship bouncers, no
     sun-crossers, no hopeless far shots past the reach ceiling). That kills the
     waste of the spatial prototype by construction. A same-arrival cohort
     SUPPLEMENT covers defended targets no single planet can solo-take (combat
     rule 1 sums same-owner same-turn arrivals — the teamwork-coalition regime).
  2. HEIGHT = importance per action: production(target) * hold_horizon *
     location_weight (enemy flip counts double) * winnability (soft reach race,
     floored so the field never freezes) * near_bias (sooner captures are more
     robust -> suppresses overreach).
  3. GRADIENT-FOLLOW = pick a robust, aligned set: greedily take highest-
     importance actions under a per-source budget (a planet keeps a defensive
     reserve and funds only one commitment), merging same-target same-arrival
     legs only when that is what wins the target.

Probe lens (PI): do the champion's good properties — no wasted ships, no two
small fleets, no shots so far the opponent reacts easily — EMERGE NATURALLY
here? `scripts/protoflow_probe.py` reads the per-turn trace below and reports
tiny-fleet / far-shot / idle / convergence rates plus winrate vs light-greedy.

Imports lib/* and agents.baseline.* directly (fine for local A/B; NOT bundled).
"""
from __future__ import annotations

import math
from collections import defaultdict

from lib.intent import World
from lib.world_model import WorldModel
from lib.kinematic_table import KinematicTable
from lib.aim import aim_orbiting
from lib.fleet import travel_time
from lib.trajectory import predict_fleet_fate
from agents.baseline.proposer import propose, capture_size

EPISODE_STEPS = 500
WIN_FLOOR = 0.10       # winnability never reaches zero -> the field never freezes
RACE_SCALE = 6.0       # turns; steepness of the reach-race / near-bias falloff
REACH_CEIL = 26        # feasibility: no capture whose (wait+eta) exceeds this (no overreach)
THREAT_ANGLE = 0.45    # rad (~26 deg); enemy fleet "incoming" if heading within this of the bearing
THREAT_RANGE = 60.0    # board units; only nearby enemy fleets count as a committed threat
MAX_CONVERGE_TARGETS = 8   # bound the cohort supplement scan

# Per-game trace for the probe runner. Each launch is a dict (see bottom).
_TRACE: list[dict] = []
# Last turn's ranked action field, for synthetic-situation calibration.
# Each entry: {"src","tgt","ships","ttc","imp","tgt_owner","prod"}.
_LAST_FIELD: list[dict] = []


def reset_trace() -> None:
    _TRACE.clear()


def get_trace() -> list[dict]:
    return list(_TRACE)


def get_last_field() -> list[dict]:
    return list(_LAST_FIELD)


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


def _enemy_best_reach(t, enemy_planets) -> float | None:
    best = None
    for ep in enemy_planets:
        tt = travel_time((ep.x, ep.y), (t.x, t.y), max(1, int(ep.ships)))
        if best is None or tt < best:
            best = tt
    return best


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    me = world.my_id
    step = world.step
    omega = world.omega
    remain = max(1, EPISODE_STEPS - step)

    world._kt = KinematicTable()
    world._kt.begin_turn(world)
    model = WorldModel.from_world(world)

    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if int(p.owner) == me]
    enemy_planets = [p for p in planets if int(p.owner) not in (-1, me)]
    enemy_fleets = [f for f in _parse_fleets(obs) if f[1] not in (-1, me)]
    if not my_planets:
        _TRACE.append({"step": step, "launches": [], "idle": True,
                       "sources": 0, "my_planets": 0, "my_ships": 0})
        return []

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = enemy_planets + [p for p in planets if int(p.owner) == -1] + threatened_mine

    # --- per-source budget: keep enough to cover committed incoming threat ---
    threat = {p.id: _incoming_threat(p, enemy_fleets) for p in my_planets}
    spare = {p.id: max(0, int(p.ships) - int(math.ceil(threat[p.id]))) for p in my_planets}
    planet_by_id = {int(p.id): p for p in planets}

    # --- reach-race winnability per target (soft, floored; reused for importance) ---
    def winnability(t, our_eta):
        ee = _enemy_best_reach(t, enemy_planets)
        if ee is None:
            return 1.0
        return WIN_FLOOR + (1.0 - WIN_FLOOR) * _sigmoid((ee - our_eta) / RACE_SCALE)

    def importance(t, time_to_cap):
        owner = int(t.owner)
        loc = 2.0 if owner not in (-1, me) else (1.5 if owner == me else 1.0)
        hold = max(1.0, remain - time_to_cap)
        near = 1.0 / (1.0 + time_to_cap / RACE_SCALE)
        return int(t.production) * hold * loc * near * winnability(t, time_to_cap)

    # ============================================================
    # FIELD (1): the champion's feasible solo-capture action set.
    # propose -> (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N)
    # ============================================================
    prerank = propose(my_planets, target_pool, world, model, me, omega,
                      baseline_len=max(2, remain))
    # Score each feasible action by capture-importance; drop overreach (feasibility).
    actions = []
    for _cheap, src, tgt, ships, angle, eta, _hz, wait in prerank:
        ttc = float(wait) + float(eta)
        if ttc > REACH_CEIL:
            continue
        actions.append({
            "src": src, "tgt": tgt, "ships": int(ships), "angle": float(angle),
            "eta": float(eta), "wait": int(wait), "arrive": int(math.ceil(ttc)),
            "imp": importance(tgt, ttc),
        })
    actions.sort(key=lambda a: -a["imp"])
    _LAST_FIELD.clear()
    for a in actions:
        _LAST_FIELD.append({
            "src": int(a["src"].id), "tgt": int(a["tgt"].id), "ships": a["ships"],
            "ttc": round(float(a["wait"]) + float(a["eta"]), 1), "imp": round(a["imp"], 1),
            "tgt_owner": int(a["tgt"].owner), "prod": int(a["tgt"].production),
        })

    moves: list[list] = []
    launches: list[dict] = []
    committed_tgt: set[int] = set()

    def emit(s, t, angle, send, eta, arrive, kind, floor):
        if send <= 0:
            return False
        if predict_fleet_fate(s, t, angle, int(send), world).outcome != "target":
            return False  # path blocked (sun / wrong planet / oob)
        moves.append([int(s.id), float(angle), int(send)])
        launches.append({
            "src": int(s.id), "tgt": int(t.id), "ships": int(send),
            "eta": round(eta, 1), "arrive_turn": int(arrive),
            "dist": round(math.hypot(t.x - s.x, t.y - s.y), 1),
            "tgt_owner": int(t.owner), "kind": kind, "floor": int(floor),
        })
        spare[int(s.id)] -= int(send)
        return True

    # GRADIENT-FOLLOW: take best feasible solo actions under the per-source budget.
    for a in actions:
        tid, sid = int(a["tgt"].id), int(a["src"].id)
        if tid in committed_tgt:
            continue  # target already won this turn
        if spare.get(sid, 0) < a["ships"]:
            continue  # can't solo-fund without overdrawing the defensive reserve
        if emit(a["src"], a["tgt"], a["angle"], a["ships"], a["eta"], a["arrive"],
                "off" if int(a["tgt"].owner) != me else "def", a["ships"]):
            committed_tgt.add(tid)

    # ============================================================
    # FIELD (2): convergence supplement. Defended targets that NO single planet
    # could solo-fund never appear in propose; assemble a same-arrival cohort
    # from remaining spare (combat rule 1 sums the legs). This is the only place
    # multi-stream alignment can arise, and only when it actually wins a target.
    # ============================================================
    solo_tids = {int(a["tgt"].id) for a in actions}
    extra = [t for t in target_pool
             if int(t.owner) != me and int(t.id) not in committed_tgt
             and int(t.id) not in solo_tids]
    # rank the supplement scan by a cheap value proxy so we bound the cost
    extra.sort(key=lambda t: -int(t.production))
    for t in extra[:MAX_CONVERGE_TARGETS]:
        legs = []  # (src, angle, eta)
        for s in my_planets:
            if spare.get(int(s.id), 0) <= 0:
                continue
            res = aim_orbiting((s.x, s.y), s.radius, t, t.radius,
                               max(1, spare[int(s.id)]), omega)
            if res is None:
                continue
            ang, _arr_xy, eta = res
            if eta > REACH_CEIL:
                continue
            legs.append((s, ang, eta))
        if len(legs) < 2:
            continue  # not a convergence opportunity
        by_turn: dict[int, list] = defaultdict(list)
        for s, ang, eta in legs:
            by_turn[int(math.ceil(eta))].append((s, ang, eta))
        for arrive in sorted(by_turn):
            cohort = by_turn[arrive]
            floor = capture_size(cohort[0][0], t, model, omega, me, world)
            if sum(spare[int(s.id)] for s, _, _ in cohort) < floor:
                continue
            cohort.sort(key=lambda c: c[2])  # nearest-first
            need = floor
            for s, ang, eta in cohort:
                if need <= 0:
                    break
                send = min(spare[int(s.id)], need)
                if emit(s, t, ang, send, eta, arrive, "conv", floor):
                    need -= send
            committed_tgt.add(int(t.id))
            break

    _TRACE.append({
        "step": step,
        "launches": launches,
        "idle": len(launches) == 0,
        "sources": len(my_planets),
        "my_planets": len(my_planets),
        "my_ships": int(sum(p.ships for p in my_planets)),
    })
    return moves
