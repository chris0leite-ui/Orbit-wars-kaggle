"""Patient — the simple agent built on the principle the others violated:
**ships on planets compound; ships in flight are wasted.**

Every prior design (phi, expander, swarm) bled ships into perpetual flight
(supply churn, far-flinging) and never built an economy, so a patient
producer out-produced and overran them. Patient does the opposite:

  - HOLD by default: a planet keeps its ships (they produce). We never shuffle
    ships between our own planets.
  - DEFEND reactively: only a planet with an enemy fleet actually inbound and
    larger than its garrison gets reinforced — just enough, from the nearest
    neighbour that can arrive in time. No blanket reserve (that hoards).
  - EXPAND efficiently: a safe planet sends *just enough* to capture the best
    nearby target (production-weighted, near-first), keeping the rest producing.
    Just-enough means almost everything stays home compounding.

Stateless, fast, orbit-aware lead-aim, sun avoidance. No supply, no search.
"""
from __future__ import annotations

import math
import os

CENTER = 50.0
SUN_R = 10.0
MAX_SPD = 6.0
LOG1000 = math.log(1000.0)


def _w(n, d):
    try:
        return float(os.environ.get(n, d))
    except (TypeError, ValueError):
        return d


ECON_K = _w("PA_ECON_K", 25.0)
ENEMY_MULT = _w("PA_ENEMY_MULT", 2.0)
DIST_SCALE = _w("PA_DIST_SCALE", 25.0)
ETA_CAP = _w("PA_ETA_CAP", 24.0)
HEAD_TOL = _w("PA_HEAD_TOL", 0.45)     # radians: a fleet is "inbound" if aimed within this
MARGIN = _w("PA_MARGIN", 2.0)
MIN_SEND = _w("PA_MIN_SEND", 2.0)
DEF_MARGIN = _w("PA_DEF_MARGIN", 1.0)


def _g(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fleet_speed(n):
    n = max(1.0, float(n))
    return 1.0 + (MAX_SPD - 1.0) * (min(1.0, math.log(n) / LOG1000)) ** 1.5


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _sun_blocked(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return _dist(ax, ay, CENTER, CENTER) < SUN_R
    t = max(0.0, min(1.0, ((CENTER - ax) * dx + (CENTER - ay) * dy) / L2))
    return _dist(CENTER, CENTER, ax + t * dx, ay + t * dy) < SUN_R + 0.5


def _predict(pl, t):
    if not pl["orbiting"]:
        return pl["x"], pl["y"]
    th = pl["theta"] + pl["omega"] * t
    return CENTER + pl["r_orb"] * math.cos(th), CENTER + pl["r_orb"] * math.sin(th)


def _aim(ax, ay, tgt, ships):
    speed = max(0.5, fleet_speed(ships))
    eta = _dist(ax, ay, tgt["x"], tgt["y"]) / speed
    for _ in range(6):
        tx, ty = _predict(tgt, eta)
        eta = _dist(ax, ay, tx, ty) / speed
    tx, ty = _predict(tgt, eta)
    return math.atan2(ty - ay, tx - ax), eta, tx, ty


def agent(obs, configuration=None):
    me = int(_g(obs, "player", 0))
    raw = _g(obs, "planets", []) or []
    fleets = _g(obs, "fleets", []) or []
    init = _g(obs, "initial_planets", []) or []
    step = int(_g(obs, "step", 0) or 0)
    angvel = float(_g(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = {int(c) for c in (_g(obs, "comet_planet_ids", []) or [])}
    init_ang = {int(p[0]): math.atan2(float(p[3]) - CENTER, float(p[2]) - CENTER) for p in init}

    P = []
    for p in raw:
        pid, owner = int(p[0]), int(p[1])
        x, y, rad, ships, prod = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        r_orb = _dist(x, y, CENTER, CENTER)
        orbiting = (r_orb + rad < 50.0) and (r_orb > 0.5) and (pid not in comet_ids)
        theta = math.atan2(y - CENTER, x - CENTER)
        omega = angvel
        if orbiting and step > 0 and pid in init_ang:
            d = theta - init_ang[pid]
            omega = ((d + math.pi) % (2 * math.pi) - math.pi) / step
        P.append(dict(id=pid, owner=owner, x=x, y=y, ships=ships, prod=prod,
                      r_orb=r_orb, theta=theta, omega=omega, orbiting=orbiting))

    mine = [p for p in P if p["owner"] == me]
    if not mine:
        return []
    targets = [p for p in P if p["owner"] != me and int(p["id"]) not in comet_ids]
    pos = {p["id"]: p for p in mine}

    # incoming enemy threat per owned planet: enemy fleets aimed at it, by arrival.
    incoming = {p["id"]: 0.0 for p in mine}
    inc_eta = {p["id"]: 1e9 for p in mine}
    for f in fleets:
        fo = int(f[1])
        if fo == me or fo < 0:
            continue
        fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), float(f[6])
        for p in mine:
            ang_to = math.atan2(p["y"] - fy, p["x"] - fx)
            da = abs((fa - ang_to + math.pi) % (2 * math.pi) - math.pi)
            if da < HEAD_TOL:
                incoming[p["id"]] += fs
                eta = _dist(fx, fy, p["x"], p["y"]) / max(0.5, fleet_speed(fs))
                inc_eta[p["id"]] = min(inc_eta[p["id"]], eta)

    # Each planet holds exactly enough to cover its own inbound attack; the rest
    # is free to expand. No binary "freeze" — a planet under light threat still
    # expands its surplus. budget = ships - (inbound it must cover).
    budget = {p["id"]: max(0.0, p["ships"] - incoming[p["id"]] - DEF_MARGIN) for p in mine}
    moves = []

    # 1. reactive defense: a planet whose own garrison can't cover its inbound
    #    attack pulls just enough from the nearest neighbour's surplus, in time.
    for p in mine:
        pid = p["id"]
        deficit = incoming[pid] + DEF_MARGIN - p["ships"]
        if deficit <= 0:
            continue
        donors = sorted(
            (q for q in mine if q["id"] != pid and budget[q["id"]] >= MIN_SEND
             and not _sun_blocked(q["x"], q["y"], p["x"], p["y"])),
            key=lambda q: _dist(q["x"], q["y"], p["x"], p["y"]),
        )
        for q in donors:
            if deficit <= 0:
                break
            ang, eta, tx, ty = _aim(q["x"], q["y"], p, budget[q["id"]])
            if eta > inc_eta[pid] + 1.0 or _sun_blocked(q["x"], q["y"], tx, ty):
                continue
            give = min(budget[q["id"]], math.ceil(deficit))
            if give < MIN_SEND:
                continue
            moves.append([q["id"], ang, int(give)])
            budget[q["id"]] -= give
            deficit -= give

    # 2. efficient expansion: send JUST ENOUGH surplus to take the best nearby
    #    target; everything else stays home producing.
    def value(t, d):
        mult = ENEMY_MULT if t["owner"] >= 0 else 1.0
        return ECON_K * t["prod"] * mult / (1.0 + d / DIST_SCALE)

    cands = []
    for s in mine:
        if budget[s["id"]] < MIN_SEND:
            continue
        for t in targets:
            if _sun_blocked(s["x"], s["y"], t["x"], t["y"]):
                continue
            d = _dist(s["x"], s["y"], t["x"], t["y"])
            ang, eta, tx, ty = _aim(s["x"], s["y"], t, budget[s["id"]])
            if eta > ETA_CAP or _sun_blocked(s["x"], s["y"], tx, ty):
                continue
            grow = t["prod"] * math.ceil(eta) if t["owner"] >= 0 else 0.0
            cost = math.ceil(t["ships"] + grow + MARGIN)
            cands.append((value(t, d), cost, t["id"], s["id"], ang))

    cands.sort(reverse=True, key=lambda c: c[0])
    taken = set()
    for val, cost, tid, sid, ang in cands:
        if tid in taken:
            continue
        if budget[sid] >= cost and cost >= MIN_SEND:
            moves.append([sid, ang, int(cost)])
            budget[sid] -= cost
            taken.add(tid)

    return moves
