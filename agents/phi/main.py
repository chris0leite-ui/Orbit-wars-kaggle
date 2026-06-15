"""Vanilla positional agent (clean-room) — a legible instrument for the
positional-game thesis.

It does NOT reuse producer. The whole agent is two ideas:

  1. A position value  Phi(state)  = (my economy + my material)
                                     - (strongest opponent's economy + material)
     where economy = production * ECON_K (production compounds, so it is worth
     ~ECON_K turns of future ships) and material = ships on planets + in fleets.

  2. Pick the launches that most increase Phi (delta-Phi greedy). Capturing a
     planet raises my economy by its production (and, if it's an enemy's, denies
     theirs — so it counts double in the differential), costs the ships the
     garrison destroys, gains "reach" if it opens new neutral frontier, and is
     penalised if it's exposed (we likely can't hold it).

Everything positional in the framework appears as a named, tunable weight:
ECON_K (compounding), REACH_W (options/space), RISK_W (durability), ETA_W
(tempo). Tune via env (PHI_ECON_K=... etc.) to read off what actually matters.

Stateless: a pure function of the observation each turn (orbit motion is
recovered from initial_planets), so the same callable can play both seats in
self-play with no shared state.
"""
from __future__ import annotations

import math
import os

CENTER = 50.0
SUN_R = 10.0
MAX_SPD = 6.0
LOG1000 = math.log(1000.0)
GAME_LEN = 500


def _w(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- positional weights (the knobs the thesis is about) --------------------
ECON_K = _w("PHI_ECON_K", 25.0)      # turns of production a planet is worth (compounding)
REACH_W = _w("PHI_REACH_W", 3.0)     # value of new neutral frontier a capture opens (options/space)
RISK_W = _w("PHI_RISK_W", 1.0)       # penalty for exposed captures we may not hold (durability)
ETA_W = _w("PHI_ETA_W", 0.5)         # prefer sooner arrivals (tempo)
FRONTIER_D = _w("PHI_FRONTIER_D", 28.0)   # "near" radius for frontier/options
RESERVE_D = _w("PHI_RESERVE_D", 22.0)     # keep ships home vs enemy fleets within this
THREAT_D = _w("PHI_THREAT_D", 30.0)       # enemy planets within this contest a target
ETA_CAP = _w("PHI_ETA_CAP", 40.0)         # ignore targets we can't reach within this many turns


def _g(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fleet_speed(n: float) -> float:
    n = max(1.0, float(n))
    r = min(1.0, math.log(n) / LOG1000)
    return 1.0 + (MAX_SPD - 1.0) * r ** 1.5


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _seg_center_dist(ax, ay, bx, by):
    """Distance from the sun (board centre) to segment AB — for sun avoidance."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return _dist(ax, ay, CENTER, CENTER)
    t = ((CENTER - ax) * dx + (CENTER - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return _dist(CENTER, CENTER, ax + t * dx, ay + t * dy)


def _predict(pl, t):
    """Position of planet pl in t turns (orbit recovered per-planet)."""
    if not pl["orbiting"]:
        return pl["x"], pl["y"]
    th = pl["theta"] + pl["omega"] * t
    return CENTER + pl["r_orb"] * math.cos(th), CENTER + pl["r_orb"] * math.sin(th)


def _lead_aim(ax, ay, tgt, ships):
    """Angle + ETA to intercept a (possibly orbiting) target from (ax,ay)."""
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

    planets = []
    for p in raw:
        pid, owner = int(p[0]), int(p[1])
        x, y, rad, ships, prod = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        r_orb = _dist(x, y, CENTER, CENTER)
        orbiting = (r_orb + rad < 50.0) and (r_orb > 0.5) and (pid not in comet_ids)
        theta = math.atan2(y - CENTER, x - CENTER)
        omega = angvel
        if orbiting and step > 0 and pid in init_ang:        # recover signed omega exactly
            d = theta - init_ang[pid]
            d = (d + math.pi) % (2 * math.pi) - math.pi
            omega = d / step
        planets.append(dict(id=pid, owner=owner, x=x, y=y, rad=rad, ships=ships,
                            prod=prod, r_orb=r_orb, theta=theta, omega=omega,
                            orbiting=orbiting, comet=(pid in comet_ids)))

    mine = [p for p in planets if p["owner"] == me]
    if not mine:
        return []
    targets = [p for p in planets if p["owner"] != me and not p["comet"]]
    enemies = [p for p in planets if p["owner"] >= 0 and p["owner"] != me]
    neutrals = [p for p in planets if p["owner"] == -1 and not p["comet"]]

    # Defensive reserve per owned planet: enemy fleet ships bearing down on it.
    reserve = {}
    for p in mine:
        thr = 0.0
        for f in fleets:
            if int(f[1]) != me and int(f[1]) >= 0:
                if _dist(float(f[2]), float(f[3]), p["x"], p["y"]) < RESERVE_D:
                    thr += float(f[6])
        reserve[p["id"]] = thr

    # Frontier helper: production a target uniquely opens (neutrals near it that
    # are NOT already near something we own) — the "options/space" term.
    my_xy = [(p["x"], p["y"]) for p in mine]

    def frontier_gain(t):
        g = 0.0
        for q in neutrals:
            if q["id"] == t["id"]:
                continue
            if _dist(q["x"], q["y"], t["x"], t["y"]) < FRONTIER_D:
                if all(_dist(q["x"], q["y"], mx, my) > FRONTIER_D for mx, my in my_xy):
                    g += q["prod"]
        return g

    # Exposure helper: enemy force that can contest a target beyond what we hold.
    def exposure(t, defender_after):
        force = 0.0
        for e in enemies:
            if _dist(e["x"], e["y"], t["x"], t["y"]) < THREAT_D:
                force += e["ships"]
        return max(0.0, force - defender_after)

    # ---- generate delta-Phi candidates -----------------------------------
    cands = []
    for a in mine:
        spare = a["ships"] - reserve[a["id"]]
        if spare < 2:
            continue
        for t in targets:
            # size the send: clear the garrison at arrival (+1), capped by spare
            sp_guess = fleet_speed(spare)
            eta0 = _dist(a["x"], a["y"], t["x"], t["y"]) / max(0.5, sp_guess)
            if eta0 > ETA_CAP:
                continue
            grow = t["prod"] * math.ceil(eta0) if t["owner"] >= 0 else 0.0
            defenders = t["ships"] + grow
            send = min(spare, math.ceil(defenders) + 1)
            if send <= defenders:                  # can't take it with what we have
                continue
            angle, eta, tx, ty = _lead_aim(a["x"], a["y"], t, send)
            if eta > ETA_CAP:
                continue
            if _seg_center_dist(a["x"], a["y"], tx, ty) < SUN_R + 0.5:   # would cross the sun
                continue
            grow = t["prod"] * math.ceil(eta) if t["owner"] >= 0 else 0.0
            defenders = t["ships"] + grow
            if send <= defenders:
                continue
            defender_after = send - defenders      # ours left holding it

            econ = ECON_K * t["prod"]              # economy we gain
            deny = ECON_K * t["prod"] if t["owner"] >= 0 else 0.0   # differential: deny enemy
            reach = REACH_W * frontier_gain(t)
            risk = RISK_W * exposure(t, defender_after)
            material = defenders                   # ships the garrison destroys
            dphi = econ + deny + reach - material - risk - ETA_W * eta
            if dphi > 0:
                cands.append((dphi, a["id"], t["id"], angle, int(send)))

    # ---- greedy select by delta-Phi --------------------------------------
    cands.sort(reverse=True, key=lambda c: c[0])
    budget = {p["id"]: p["ships"] - reserve[p["id"]] for p in mine}
    taken = set()
    moves = []
    for dphi, aid, tid, angle, send in cands:
        if tid in taken:
            continue
        if budget.get(aid, 0) >= send:
            moves.append([aid, angle, send])
            budget[aid] -= send
            taken.add(tid)

    return moves
