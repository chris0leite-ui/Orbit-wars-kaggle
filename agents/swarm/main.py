"""Swarm — a simple, fully-ours agent built around FORCE CONCENTRATION, the
thing the single-source heuristics missed.

Reactive single-source greedy stalls: it only takes a planet one planet can
afford alone, so it prunes the far/expensive planets producer takes and gets
out-expanded. Swarm fixes the structure, not a knob:

  1. DEFEND: hold against fleets actually inbound to each planet.
  2. CONCENTRATE: rank objectives by economic value; for each, POOL ships from
     the nearest planets until the pooled force can take it (cumulative waves
     chip a neutral garrison down), and fire the whole coalition. Only commit
     when the pool can actually take it — no partial bleed.
  3. SUPPLY: planets with leftover ships and no objective of their own stream
     toward their nearest more-forward planet, so the frontier always has ammo.

Stateless, fast, orbit-aware lead-aim, sun avoidance. No search, no vendored
code.
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


ECON_K = _w("SW_ECON_K", 25.0)
ENEMY_MULT = _w("SW_ENEMY_MULT", 2.0)
DIST_SCALE = _w("SW_DIST_SCALE", 25.0)   # value falls off with distance (tempo)
ETA_CAP = _w("SW_ETA_CAP", 40.0)
RESERVE_D = _w("SW_RESERVE_D", 22.0)
MARGIN = _w("SW_MARGIN", 2.0)
MIN_SEND = _w("SW_MIN_SEND", 2.0)
SUPPLY_GAP = _w("SW_SUPPLY_GAP", 6.0)    # supply only to a planet this much closer to the enemy
MAX_OBJ = int(_w("SW_MAX_OBJ", 6))       # concentrate on the few best objectives/turn


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
    targets = [p for p in P if p["owner"] != me and (int(p["id"]) not in comet_ids)]
    enemy_xy = [(p["x"], p["y"]) for p in P if p["owner"] >= 0 and p["owner"] != me]

    # 1. defend: reserve ships against fleets actually inbound to each planet.
    reserve = {}
    for p in mine:
        thr = sum(float(f[6]) for f in fleets
                  if int(f[1]) != me and int(f[1]) >= 0
                  and _dist(float(f[2]), float(f[3]), p["x"], p["y"]) < RESERVE_D)
        reserve[p["id"]] = thr
    budget = {p["id"]: max(0.0, p["ships"] - reserve[p["id"]]) for p in mine}

    def nearest_mine_dist(t):
        return min(_dist(s["x"], s["y"], t["x"], t["y"]) for s in mine)

    # 2. concentrate: rank objectives by economic value, pool force to take each.
    def value(t):
        mult = ENEMY_MULT if t["owner"] >= 0 else 1.0
        return ECON_K * t["prod"] * mult / (1.0 + nearest_mine_dist(t) / DIST_SCALE)

    objectives = sorted(targets, key=value, reverse=True)[:MAX_OBJ]
    moves = []
    for t in objectives:
        # candidate sources: nearest first, sun-clear, with spare.
        srcs = sorted(
            (s for s in mine if budget[s["id"]] >= MIN_SEND
             and not _sun_blocked(s["x"], s["y"], t["x"], t["y"])),
            key=lambda s: _dist(s["x"], s["y"], t["x"], t["y"]),
        )
        if not srcs:
            continue
        # size against the slowest contributor's arrival (enemy garrison regrows).
        eta_far = _aim(srcs[-1]["x"], srcs[-1]["y"], t, MIN_SEND)[1]
        if _aim(srcs[0]["x"], srcs[0]["y"], t, budget[srcs[0]["id"]])[1] > ETA_CAP:
            continue
        grow = t["prod"] * math.ceil(eta_far) if t["owner"] >= 0 else 0.0
        need = math.ceil(t["ships"] + grow + MARGIN)
        # allocate from nearest sources until the pool covers the cost.
        alloc, acc = [], 0
        for s in srcs:
            if acc >= need:
                break
            give = min(int(budget[s["id"]]), need - acc)
            if give < MIN_SEND:
                continue
            alloc.append((s, give))
            acc += give
        if acc < need:
            continue                       # can't take it even pooled -> leave for supply
        for s, give in alloc:
            ang, eta, tx, ty = _aim(s["x"], s["y"], t, give)
            if _sun_blocked(s["x"], s["y"], tx, ty):
                continue
            moves.append([s["id"], ang, int(give)])
            budget[s["id"]] -= give

    # 3. supply: leftover ships flow to the nearest more-forward owned planet.
    if enemy_xy:
        def ne(px, py):
            return min(_dist(px, py, ex, ey) for ex, ey in enemy_xy)
        for s in mine:
            spare = budget[s["id"]]
            if spare < MIN_SEND:
                continue
            fwd = [o for o in mine if o["id"] != s["id"]
                   and ne(o["x"], o["y"]) < ne(s["x"], s["y"]) - SUPPLY_GAP
                   and not _sun_blocked(s["x"], s["y"], o["x"], o["y"])]
            if not fwd:
                continue
            dest = min(fwd, key=lambda o: _dist(s["x"], s["y"], o["x"], o["y"]))
            ang, eta, tx, ty = _aim(s["x"], s["y"], dest, spare)
            if _sun_blocked(s["x"], s["y"], tx, ty):
                continue
            moves.append([s["id"], ang, int(spare)])
            budget[s["id"]] -= spare

    return moves
