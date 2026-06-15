"""Expander — a simple, fast, fully-ours agent distilling the positional finding:
**aggressive economic expansion + tempo + force concentration.**

Three steps, no search, stateless, ~0 ms/turn:

  1. DEFEND minimally: each owned planet reserves only the ships an incoming
     enemy fleet actually threatens — nothing more (caution hurts).
  2. EXPAND aggressively: rank every capturable planet by economic value
     (production it adds / denies, minus the ships it costs, minus a small
     tempo penalty), and take the best ones with just-enough ships from the
     nearest planet that can afford it.
  3. CONCENTRATE: rear planets with leftover ships stream toward the front
     (the owned planet nearest the enemy frontier) so force massings build
     where the fighting is — the classic Planet-Wars reinforcement win.

No reach/options term, no risk/caution term, no forward search — the weight
sweep showed those hurt. Correct tactics (orbit-aware lead-aim recovered from
initial_planets, just-enough capture sizing, sun avoidance) are what a simple
agent needs to be good.
"""
from __future__ import annotations

import math
import os

CENTER = 50.0
SUN_R = 10.0
MAX_SPD = 6.0
LOG1000 = math.log(1000.0)


def _w(name, d):
    try:
        return float(os.environ.get(name, d))
    except (TypeError, ValueError):
        return d


ECON_K = _w("EXP_ECON_K", 25.0)        # turns of production a planet is worth
ENEMY_MULT = _w("EXP_ENEMY_MULT", 2.0)  # capturing enemy also denies their economy
ETA_W = _w("EXP_ETA_W", 0.6)           # tempo: prefer sooner arrivals
ETA_CAP = _w("EXP_ETA_CAP", 45.0)      # ignore targets we can't reach within this
RESERVE_D = _w("EXP_RESERVE_D", 20.0)  # enemy fleets within this threaten a planet
MARGIN = _w("EXP_MARGIN", 1.0)         # extra ships over the garrison when capturing
MIN_SEND = _w("EXP_MIN_SEND", 2.0)
REINFORCE_FRAC = _w("EXP_REINFORCE_FRAC", 1.0)   # share of rear leftover to push forward


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
                      r_orb=r_orb, theta=theta, omega=omega, orbiting=orbiting,
                      comet=(pid in comet_ids)))

    mine = [p for p in P if p["owner"] == me]
    if not mine:
        return []
    targets = [p for p in P if p["owner"] != me and not p["comet"]]
    enemy_xy = [(p["x"], p["y"]) for p in P if p["owner"] >= 0 and p["owner"] != me]

    # 1. minimal defense: reserve only against incoming enemy fleets.
    reserve = {}
    for p in mine:
        thr = sum(float(f[6]) for f in fleets
                  if int(f[1]) != me and int(f[1]) >= 0
                  and _dist(float(f[2]), float(f[3]), p["x"], p["y"]) < RESERVE_D)
        reserve[p["id"]] = thr
    budget = {p["id"]: max(0.0, p["ships"] - reserve[p["id"]]) for p in mine}
    pos = {p["id"]: p for p in mine}

    moves = []

    # 2. aggressive economic expansion: rank targets by econ value via their
    #    nearest affordable source; take the best with just-enough ships.
    scored = []
    for t in targets:
        best = None
        for s in mine:
            if _sun_blocked(s["x"], s["y"], t["x"], t["y"]):
                continue
            d = _dist(s["x"], s["y"], t["x"], t["y"])
            if best is None or d < best[1]:
                best = (s, d)
        if best is None:
            continue
        s, _d = best
        ang, eta, tx, ty = _aim(s["x"], s["y"], t, max(budget[s["id"]], 1.0))
        if eta > ETA_CAP or _sun_blocked(s["x"], s["y"], tx, ty):
            continue
        grow = t["prod"] * math.ceil(eta) if t["owner"] >= 0 else 0.0
        cost = t["ships"] + grow + MARGIN
        mult = ENEMY_MULT if t["owner"] >= 0 else 1.0
        score = ECON_K * t["prod"] * mult - cost - ETA_W * eta
        scored.append((score, cost, t, s["id"], ang))

    scored.sort(key=lambda r: r[0], reverse=True)
    used_target = set()
    for score, cost, t, sid, ang in scored:
        if score <= 0 or t["id"] in used_target:
            continue
        send = math.ceil(cost)
        if budget[sid] >= send and send >= MIN_SEND:
            moves.append([sid, ang, int(send)])
            budget[sid] -= send
            used_target.add(t["id"])

    # 3. force concentration: rear planets push leftover toward the front
    #    (the owned planet nearest the enemy). Skipped if no enemy seen yet.
    if enemy_xy:
        def near_enemy(px, py):
            return min(_dist(px, py, ex, ey) for ex, ey in enemy_xy)
        front = min(mine, key=lambda p: near_enemy(p["x"], p["y"]))
        for s in mine:
            if s["id"] == front["id"]:
                continue
            spare = budget[s["id"]]
            if spare < MIN_SEND:
                continue
            # only push if this planet is meaningfully behind the front
            if near_enemy(s["x"], s["y"]) <= near_enemy(front["x"], front["y"]) + 5.0:
                continue
            if _sun_blocked(s["x"], s["y"], front["x"], front["y"]):
                continue
            send = math.floor(spare * REINFORCE_FRAC)
            if send < MIN_SEND:
                continue
            ang, eta, tx, ty = _aim(s["x"], s["y"], front, send)
            if _sun_blocked(s["x"], s["y"], tx, ty):
                continue
            moves.append([s["id"], ang, int(send)])
            budget[s["id"]] -= send

    return moves
