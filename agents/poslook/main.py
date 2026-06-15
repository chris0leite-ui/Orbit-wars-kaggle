"""poslook — the agent the reactive ones weren't: it LOOKS AHEAD.

Producer's idea (simulate ~H steps forward assuming a static opponent, pick the
action that scores best at the horizon) but with a POSITIONAL objective:

    not  "how do I have the most ships in 18 steps"
    but  "how do I have the best POSITION vs my opponent in 18 steps"

Position value at the horizon (differential, vs the strongest opponent):

    Phi = sum_my (ships + ECON*prod)  -  sum_opp (ships + ECON*prod)  [+ in-flight]

ECON*prod credits a planet's ongoing production as a compounding asset, so a
capture that won't repay its ships within 18 steps but gives lasting economy
scores POSITIVE — which raw ship-flow misses.

Each turn: greedily add the launch that most increases simulated Phi (re-sim
after each commit so later launches see earlier ones), against a static
opponent (existing fleets resolve; the opponent launches nothing new).

Clean-room, stateless. Event-based forward sim (fleet arrivals + production +
combat) — accurate enough to choose well, simple enough to read.
"""
from __future__ import annotations

import math
import os

CENTER = 50.0
SUN_R = 10.0
MAX_SPD = 6.0
LOG1000 = math.log(1000.0)


def _wf(n, d):
    try:
        return float(os.environ.get(n, d))
    except (TypeError, ValueError):
        return d


H = int(_wf("PL_H", 18))            # lookahead horizon (steps)
ECON = _wf("PL_ECON", 6.0)          # production weight in the position value
ETA_CAP = _wf("PL_ETA_CAP", 24.0)
MAX_TGT = int(_wf("PL_MAX_TGT", 7))  # nearest targets considered per source
MAX_WAVES = int(_wf("PL_MAX_WAVES", 12))
MARGIN = _wf("PL_MARGIN", 1.0)
MIN_SEND = _wf("PL_MIN_SEND", 2.0)


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
    for _ in range(5):
        tx, ty = _predict(tgt, eta)
        eta = _dist(ax, ay, tx, ty) / speed
    tx, ty = _predict(tgt, eta)
    return math.atan2(ty - ay, tx - ax), eta, tx, ty


def _combat(owner, ships, arrivals):
    """Resolve arrivals (dict owner->ships) onto a planet (owner, ships).
    Returns (new_owner, new_ships). 2-player-exact; 4P approximated by top-two."""
    if not arrivals:
        return owner, ships
    groups = sorted(arrivals.items(), key=lambda kv: kv[1], reverse=True)
    if len(groups) >= 2 and groups[0][1] == groups[1][1]:
        return owner, ships                      # tie among top attackers: all destroyed
    atk_owner, atk = groups[0]
    if len(groups) >= 2:
        atk -= groups[1][1]                      # largest fights second-largest
    if atk <= 0:
        return owner, ships
    if atk_owner == owner:
        return owner, ships + atk                # reinforcement
    if atk > ships:
        return atk_owner, atk - ships            # capture
    return owner, ships - atk                    # held


def _simulate(planets, in_flight, me, opp_set, launches):
    """Forward-sim H steps under a static opponent; return position value Phi.

    planets: list of dicts (id,owner,ships,prod). in_flight/launches: lists of
    (arr_step, target_id, owner, ships). Opponent adds nothing new (static).
    """
    owner = {p["id"]: p["owner"] for p in planets}
    ships = {p["id"]: p["ships"] for p in planets}
    prod = {p["id"]: p["prod"] for p in planets}
    # bucket arrivals by step
    arr = [dict() for _ in range(H + 2)]
    flight_after = []   # (owner, ships) still travelling past H
    for (k, tid, o, s) in in_flight + launches:
        kk = int(math.ceil(k))
        if 1 <= kk <= H:
            d = arr[kk].setdefault(tid, {})
            d[o] = d.get(o, 0.0) + s
        elif kk > H:
            flight_after.append((o, s))
    for k in range(1, H + 1):
        for pid in owner:
            if owner[pid] >= 0:
                ships[pid] += prod[pid]          # production
        for tid, byowner in arr[k].items():
            if tid not in owner:
                continue
            owner[tid], ships[tid] = _combat(owner[tid], ships[tid], byowner)
    phi = 0.0
    for pid in owner:
        if owner[pid] == me:
            phi += ships[pid] + ECON * prod[pid]
        elif owner[pid] in opp_set:
            phi -= ships[pid] + ECON * prod[pid]
    for (o, s) in flight_after:
        phi += s if o == me else (-s if o in opp_set else 0.0)
    return phi


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
        x, y, rad, sh, pr = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        r_orb = _dist(x, y, CENTER, CENTER)
        orbiting = (r_orb + rad < 50.0) and (r_orb > 0.5) and (pid not in comet_ids)
        theta = math.atan2(y - CENTER, x - CENTER)
        omega = angvel
        if orbiting and step > 0 and pid in init_ang:
            d = theta - init_ang[pid]
            omega = ((d + math.pi) % (2 * math.pi) - math.pi) / step
        P.append(dict(id=pid, owner=owner, x=x, y=y, ships=sh, prod=pr,
                      r_orb=r_orb, theta=theta, omega=omega, orbiting=orbiting))

    mine = [p for p in P if p["owner"] == me]
    if not mine:
        return []
    by_id = {p["id"]: p for p in P}
    opp_set = {p["owner"] for p in P if p["owner"] >= 0 and p["owner"] != me}
    targets = [p for p in P if p["owner"] != me and p["id"] not in comet_ids]

    # infer existing fleets' (target, eta) so the sim accounts for them.
    in_flight = []
    for f in fleets:
        fo, fx, fy, fa, fs = int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[6])
        if fo < 0:
            continue
        best, bestd = None, 1e9
        for p in P:
            ang_to = math.atan2(p["y"] - fy, p["x"] - fx)
            da = abs((fa - ang_to + math.pi) % (2 * math.pi) - math.pi)
            if da < 0.4:
                d = _dist(fx, fy, p["x"], p["y"])
                if d < bestd:
                    best, bestd = p["id"], d
        if best is not None:
            in_flight.append((bestd / max(0.5, fleet_speed(fs)), best, fo, fs))

    # base position value (do-nothing).
    base_phi = _simulate(P, in_flight, me, opp_set, [])

    # candidate launches: each source -> nearest MAX_TGT targets, just-enough.
    cand = []
    for s in mine:
        near = sorted(targets, key=lambda t: _dist(s["x"], s["y"], t["x"], t["y"]))[:MAX_TGT]
        for t in near:
            if _sun_blocked(s["x"], s["y"], t["x"], t["y"]):
                continue
            ang, eta, tx, ty = _aim(s["x"], s["y"], t, max(s["ships"], 1.0))
            if eta > ETA_CAP or _sun_blocked(s["x"], s["y"], tx, ty):
                continue
            grow = t["prod"] * math.ceil(eta) if t["owner"] >= 0 else 0.0
            need = math.ceil(t["ships"] + grow + MARGIN)
            cand.append(dict(src=s["id"], tgt=t["id"], ang=ang, eta=eta, need=need))

    # greedy: repeatedly add the launch that most improves simulated Phi.
    budget = {p["id"]: p["ships"] for p in mine}
    committed = []          # (eta, tgt, owner, ships) for sim
    taken = set()
    moves = []
    cur_phi = base_phi
    for _ in range(MAX_WAVES):
        best, best_gain, best_send = None, 1e-6, 0
        for c in cand:
            if c["tgt"] in taken:
                continue
            send = c["need"]
            if budget[c["src"]] < max(send, MIN_SEND):
                continue
            trial = committed + [(c["eta"], c["tgt"], me, float(send))]
            phi = _simulate(P, in_flight, me, opp_set, trial)
            gain = phi - cur_phi
            if gain > best_gain:
                best, best_gain, best_send = c, gain, send
        if best is None:
            break
        committed.append((best["eta"], best["tgt"], me, float(best_send)))
        budget[best["src"]] -= best_send
        taken.add(best["tgt"])
        cur_phi += best_gain
        moves.append([best["src"], best["ang"], int(best_send)])
    return moves
