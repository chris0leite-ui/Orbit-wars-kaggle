"""Condensed light approximation of the v3.5.1 teacher (top_tier_mirror).

The teacher's strength = predictive DEFENSE (reinforce planets that will fall)
+ hold-aware OFFENSE (production x hold-time / (ships+dist), skip unaffordable
/unholdable) + per-source COORDINATION + aggressive sizing + spoiler-the-leader.
The only expensive part is the WorldModel timeline; we replace it with cheap
projection (ships + production*eta) and the in-flight `fleets` list. No
WorldModel, no learned model -> fast.
"""
from __future__ import annotations
import math

# Teacher constants (lib/missions/snipe.py)
AGG_FRACTION = 0.7
AGG_RESERVE = 5
AGG_MIN_GARRISON = 12
LEADER_MULT = 1.5
EPISODE_STEPS = 500
SUN_X = SUN_Y = 50.0
SUN_R = 10.0
MIN_SRC = 8


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def fleet_speed(ships: float) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + 5.0 * (math.log(max(ships, 1.0)) / math.log(1000.0)) ** 1.5
    return min(6.0, max(1.0, s))


def _crosses_sun(sx, sy, tx, ty) -> bool:
    """Does the segment src->tgt pass within the sun radius?"""
    dx, dy = tx - sx, ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return False
    u = ((SUN_X - sx) * dx + (SUN_Y - sy) * dy) / L2
    u = max(0.0, min(1.0, u))
    cx, cy = sx + u * dx, sy + u * dy
    return (cx - SUN_X) ** 2 + (cy - SUN_Y) ** 2 < (SUN_R + 1.0) ** 2


def _fleet_target(f, planets, me):
    """Which planet is enemy fleet f aimed at (nearest angular match)?"""
    fx, fy, fa = float(f[2]), float(f[3]), float(f[4])
    best, bestd = None, 0.20
    for p in planets:
        a = math.atan2(float(p[3]) - fy, float(p[2]) - fx)
        d = abs((a - fa + math.pi) % (2 * math.pi) - math.pi)
        if d < bestd:
            bestd, best = d, p
    return best


class CondensedPolicy:
    def __init__(self, num_seats: int = 2):
        self.num_seats = num_seats

    def __call__(self, obs):
        me = int(_read(obs, "player", 0))
        planets = _read(obs, "planets", []) or []
        fleets = _read(obs, "fleets", []) or []
        step = int(_read(obs, "step", 0))
        mine = [p for p in planets if int(p[1]) == me]
        targets = [p for p in planets if int(p[1]) != me]
        if not mine or not targets:
            return []

        # --- strength -> leader / spoiler ---
        strength = {}
        for p in planets:
            o = int(p[1])
            if o >= 0:
                strength[o] = strength.get(o, 0.0) + float(p[5])
        for f in fleets:
            o = int(f[1])
            if o >= 0:
                strength[o] = strength.get(o, 0.0) + float(f[6])
        my_str = strength.get(me, 0.0)
        leader = max((o for o in strength if o != me),
                     key=lambda o: strength[o], default=None)
        my_rank = 1 + sum(1 for o, s in strength.items() if o != me and s > my_str)
        spoiler = self.num_seats >= 3 and my_rank >= 2 and leader is not None

        # --- incoming enemy threats per my planet (from in-flight fleets) ---
        threats: dict[int, list] = {}
        pid = {int(p[0]): p for p in planets}
        for f in fleets:
            fo = int(f[1])
            if fo == me or fo < 0:
                continue
            tgt = _fleet_target(f, planets, me)
            if tgt is None or int(tgt[1]) != me:
                continue
            d = math.hypot(float(tgt[2]) - float(f[2]), float(tgt[3]) - float(f[3]))
            eta = max(1, int(math.ceil(d / fleet_speed(float(f[6])))))
            threats.setdefault(int(tgt[0]), []).append((eta, float(f[6])))

        missions = []  # (score, src_id, tgt_id, ships, eta, angle)

        # --- DEFENSE: reinforce planets that will fall ---
        for d in mine:
            inc = sorted(threats.get(int(d[0]), []))
            if not inc:
                continue
            garr = float(d[5]); prod = float(d[6])
            cum = 0.0; t_loss = None; atk = 0.0
            for (eta, sh) in inc:
                cum += sh
                if cum > garr + prod * eta:
                    t_loss = eta; atk = cum; break
            if t_loss is None:
                continue
            for s in mine:
                if int(s[0]) == int(d[0]) or float(s[5]) < 2:
                    continue
                cost = int(atk) + 1
                dist = math.hypot(float(d[2]) - float(s[2]), float(d[3]) - float(s[3]))
                eta = max(1, int(math.ceil(dist / fleet_speed(cost))))
                if eta >= t_loss or cost > float(s[5]):
                    continue
                if _crosses_sun(float(s[2]), float(s[3]), float(d[2]), float(d[3])):
                    continue
                hold = max(1.0, EPISODE_STEPS - step - eta)
                score = (prod * hold) / (cost + dist + 1.0)
                ang = math.atan2(float(d[3]) - float(s[3]), float(d[2]) - float(s[2]))
                missions.append((score, int(s[0]), int(d[0]), cost, eta, ang))

        # --- OFFENSE: hold-aware snipes ---
        for s in mine:
            s_sh = float(s[5])
            if s_sh < MIN_SRC:
                continue
            sx, sy = float(s[2]), float(s[3])
            for t in targets:
                tx, ty = float(t[2]), float(t[3])
                if _crosses_sun(sx, sy, tx, ty):
                    continue
                dist = math.hypot(tx - sx, ty - sy)
                t_sh, t_pr, t_owner = float(t[5]), float(t[6]), int(t[1])
                target_min = max(1, int(t_sh) + 1)
                if s_sh > AGG_MIN_GARRISON:
                    frac = max(1, int(s_sh * AGG_FRACTION))
                    cap = max(1, int(s_sh) - AGG_RESERVE)
                    agg = max(target_min, min(frac, cap))
                else:
                    agg = target_min
                eta = max(1, int(math.ceil(dist / fleet_speed(max(agg, 1)))))
                # cheap garrison projection at arrival (neutrals don't grow)
                arr = t_sh + (t_pr * eta if t_owner >= 0 else 0.0)
                needed = int(math.ceil(arr)) + 1
                if needed > s_sh:
                    continue  # genuinely can't afford the capture (don't bounce)
                ships = min(int(s_sh), max(agg, needed))  # send enough to WIN
                hold = max(1.0, EPISODE_STEPS - step - eta)
                value = t_pr * hold
                priority = 1.0
                if spoiler and t_owner == leader:
                    priority *= LEADER_MULT
                score = priority * value / (ships + dist + 1.0)
                ang = math.atan2(ty - sy, tx - sx)
                missions.append((score, int(s[0]), int(t[0]), ships, eta, ang))

        # --- COORDINATE: one per source, best-first, target over-commit ledger ---
        by_src: dict[int, list] = {}
        for m in missions:
            by_src.setdefault(m[1], []).append(m)
        for sid in by_src:
            by_src[sid].sort(key=lambda m: -m[0])
        order = sorted(by_src.keys(), key=lambda sid: -by_src[sid][0][0])
        pending: dict[int, float] = {}
        action = []
        for sid in order:
            for m in by_src[sid]:
                score, src_id, tgt_id, ships, eta, ang = m
                need = float(pid[tgt_id][5]) + 1.0
                if pending.get(tgt_id, 0.0) >= need:
                    continue
                action.append([src_id, float(ang), int(ships)])
                pending[tgt_id] = pending.get(tgt_id, 0.0) + ships
                break
        return action
