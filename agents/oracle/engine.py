"""Oracle agent — exact-physics engine core.

Port of the parity-tested ledger World (agents/ledger/main.py, gated by
tests/test_ledger_forecast.py): per-tick exact planet positions (orbits +
comet paths), exact fleet flight with swept-disk collision in engine order,
production, and combat — giving every planet's (owner, ships) timeline over
the horizon absent new launches. Decision logic lives elsewhere; this module
adds only the injection hooks (walk_with) that the feature extractor and the
plan evaluator share.

Engine tick order mirrored from kaggle_environments orbit_wars:
  launches -> production -> fleet moves with swept-disk collision (planets
  in list order, then out-of-bounds, then sun) -> planet rotation / comet
  path advance -> combat.
"""

import math

# ---------------------------------------------------------------- constants
BOARD = 100.0
CENTER = 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
LOG1000 = math.log(1000.0)
EPISODE_END = 498          # interpreter flags DONE at step >= episodeSteps-2

DEFAULT_HORIZON = 90


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + 5.0 * (math.log(ships) / LOG1000) ** 1.5
    return s if s < 6.0 else 6.0


def _g(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


class World:
    """Per-turn snapshot + exact position table + ledger."""

    __slots__ = (
        "step", "me", "n_planets", "pid", "owner0", "px", "py", "pr",
        "ships0", "prod", "pos", "alive_until", "idx_of", "fleets",
        "omega", "horizon", "arrivals", "pre_owner", "pre_ships",
        "post_owner", "post_ships", "is_comet", "remaining", "flights",
    )

    def __init__(self, obs, horizon=DEFAULT_HORIZON, step=None):
        self.step = int(_g(obs, "step", 0) or 0) if step is None else int(step)
        self.me = int(_g(obs, "player", 0) or 0)
        planets = [list(p) for p in (_g(obs, "planets", []) or [])]
        self.fleets = [list(f) for f in (_g(obs, "fleets", []) or [])]
        self.omega = float(_g(obs, "angular_velocity", 0.03) or 0.03)
        self.horizon = min(horizon, max(8, EPISODE_END - self.step))
        self.remaining = max(0, EPISODE_END - self.step)

        n = len(planets)
        self.n_planets = n
        self.pid = [p[0] for p in planets]
        self.owner0 = [p[1] for p in planets]
        self.px = [float(p[2]) for p in planets]
        self.py = [float(p[3]) for p in planets]
        self.pr = [float(p[4]) for p in planets]
        self.ships0 = [int(p[5]) for p in planets]
        self.prod = [int(p[6]) for p in planets]
        self.idx_of = {p[0]: i for i, p in enumerate(planets)}

        comet_ids = set(_g(obs, "comet_planet_ids", []) or [])
        self.is_comet = [p[0] in comet_ids for p in planets]

        comet_path = {}
        for group in (_g(obs, "comets", []) or []):
            gids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
            gpaths = group["paths"] if isinstance(group, dict) else group.paths
            gidx = group["path_index"] if isinstance(group, dict) else group.path_index
            for i, cpid in enumerate(gids):
                comet_path[cpid] = (gpaths[i], gidx)

        H = self.horizon
        # pos[i] = list of (x, y) for dt = 0..H. alive_until[i] = last tick
        # the planet is collidable (comets: the tick they vanish; they sit
        # at their final path point during that tick).
        self.pos = []
        self.alive_until = []
        for i in range(n):
            x0, y0 = self.px[i], self.py[i]
            if self.is_comet[i]:
                entry = comet_path.get(self.pid[i])
                if entry is None:
                    self.pos.append([(x0, y0)] * (H + 1))
                    self.alive_until.append(H)
                    continue
                path, k = entry
                last = len(path) - 1
                row = []
                for dt in range(H + 1):
                    j = k + dt
                    if j <= last:
                        pt = path[j]
                        row.append((float(pt[0]), float(pt[1])))
                    else:
                        pt = path[last]
                        row.append((float(pt[0]), float(pt[1])))
                self.pos.append(row)
                self.alive_until.append(min(H, last - k + 1))
                continue
            dx, dy = x0 - CENTER, y0 - CENTER
            orb_r = math.hypot(dx, dy)
            if orb_r + self.pr[i] < ROT_LIMIT and orb_r > 1e-9:
                a0 = math.atan2(dy, dx)
                w = self.omega
                row = [
                    (CENTER + orb_r * math.cos(a0 + w * dt),
                     CENTER + orb_r * math.sin(a0 + w * dt))
                    for dt in range(H + 1)
                ]
                self.pos.append(row)
            else:
                self.pos.append([(x0, y0)] * (H + 1))
            self.alive_until.append(1 << 30)

    # ------------------------------------------------------------- physics
    def fly(self, x, y, angle, ships, start_dt):
        """Exact flight: returns (planet_idx, hit_tick) or (None, death_tick).

        Mirrors the engine: each tick the fleet segment is tested against
        every planet's swept disk in list order (first hit wins), then
        out-of-bounds, then the sun.
        """
        H = self.horizon
        v = fleet_speed(ships)
        vx = math.cos(angle) * v
        vy = math.sin(angle) * v
        pos = self.pos
        pr = self.pr
        alive = self.alive_until
        n = self.n_planets
        for dt in range(start_dt, H + 1):
            nx = x + vx
            ny = y + vy
            lox, hix = (x, nx) if x < nx else (nx, x)
            loy, hiy = (y, ny) if y < ny else (ny, y)
            for i in range(n):
                if dt > alive[i]:
                    continue
                p0 = pos[i][dt - 1]
                p1 = pos[i][dt]
                r = pr[i]
                if p0[0] < p1[0]:
                    pminx = p0[0] - r; pmaxx = p1[0] + r
                else:
                    pminx = p1[0] - r; pmaxx = p0[0] + r
                if hix < pminx or lox > pmaxx:
                    continue
                if p0[1] < p1[1]:
                    pminy = p0[1] - r; pmaxy = p1[1] + r
                else:
                    pminy = p1[1] - r; pmaxy = p0[1] + r
                if hiy < pminy or loy > pmaxy:
                    continue
                d0x = x - p0[0]; d0y = y - p0[1]
                dvx = (nx - x) - (p1[0] - p0[0])
                dvy = (ny - y) - (p1[1] - p0[1])
                a = dvx * dvx + dvy * dvy
                b = 2.0 * (d0x * dvx + d0y * dvy)
                c = d0x * d0x + d0y * d0y - r * r
                if a < 1e-12:
                    if c <= 0.0:
                        return (i, dt)
                    continue
                disc = b * b - 4.0 * a * c
                if disc < 0.0:
                    continue
                sq = math.sqrt(disc)
                t1 = (-b - sq) / (2.0 * a)
                t2 = (-b + sq) / (2.0 * a)
                if t2 >= 0.0 and t1 <= 1.0:
                    return (i, dt)
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                return (None, dt)
            if _seg_dist_to_sun(x, y, nx, ny) < SUN_R:
                return (None, dt)
            x, y = nx, ny
        return (None, H + 1)

    # ------------------------------------------------------------- ledger
    def build_ledger(self):
        """Forecast all in-flight fleets, then walk every planet's future.

        Produces, for each planet i and tick dt in 0..H:
          pre_owner / pre_ships  — state after production, before combat
          post_owner / post_ships — state after combat
        flights — one record per in-flight fleet:
          (owner, ships, end_dt, hit_planet_idx_or_None)
        """
        H = self.horizon
        n = self.n_planets
        arrivals = [dict() for _ in range(n)]   # i -> {dt: {owner: ships}}
        flights = []
        for f in self.fleets:
            hit, dt = self.fly(float(f[2]), float(f[3]), float(f[4]),
                               int(f[6]), 1)
            flights.append((int(f[1]), int(f[6]), dt, hit))
            if hit is not None:
                slot = arrivals[hit].setdefault(dt, {})
                slot[f[1]] = slot.get(f[1], 0) + int(f[6])
        self.arrivals = arrivals
        self.flights = flights

        self.pre_owner = []
        self.pre_ships = []
        self.post_owner = []
        self.post_ships = []
        for i in range(n):
            po, ps, qo, qs = self._walk_planet(i, self.owner0[i],
                                               self.ships0[i], 0, None, 0)
            self.pre_owner.append(po)
            self.pre_ships.append(ps)
            self.post_owner.append(qo)
            self.post_ships.append(qs)

    def _walk_planet(self, i, owner, ships, extra_owner, extra_at, extra_n,
                     start_dt=1, extra_list=None):
        """Walk planet i's future from (owner, ships) at tick start_dt-1.

        Optionally injects extra arrivals — either one (extra_n ships for
        extra_owner at tick extra_at) or a list of (dt, owner, n) — used to
        price candidate launches and coalitions exactly.
        Returns four arrays indexed 0..H (entries < start_dt-1 are None).
        """
        H = self.horizon
        alive = self.alive_until[i]
        prod = self.prod[i]
        arr = self.arrivals[i]
        extras = {}
        if extra_at is not None:
            extras.setdefault(extra_at, []).append((extra_owner, extra_n))
        if extra_list:
            for e_dt, e_o, e_n in extra_list:
                extras.setdefault(e_dt, []).append((e_o, e_n))
        po = [None] * (H + 1)
        ps = [None] * (H + 1)
        qo = [None] * (H + 1)
        qs = [None] * (H + 1)
        po[start_dt - 1] = qo[start_dt - 1] = owner
        ps[start_dt - 1] = qs[start_dt - 1] = ships
        is_comet = self.is_comet[i]
        for dt in range(start_dt, H + 1):
            if dt > alive or (dt == alive and is_comet):
                # the engine removes an expiring comet (with its garrison,
                # and any fleets arriving that tick) at the END of the tick
                # its path index overruns — so the post-tick state is gone
                owner = -2
                ships = 0
                po[dt] = qo[dt] = owner
                ps[dt] = qs[dt] = ships
                continue
            if owner >= 0:
                ships += prod
            po[dt] = owner
            ps[dt] = ships
            slot = arr.get(dt)
            ex = extras.get(dt)
            if ex is not None:
                slot = dict(slot) if slot else {}
                for e_o, e_n in ex:
                    slot[e_o] = slot.get(e_o, 0) + e_n
            if slot and dt == alive and self.is_comet[i]:
                # arrivals on a comet's vanish tick die with it
                slot = None
            if slot:
                owner, ships = _resolve_combat(owner, ships, slot)
            qo[dt] = owner
            qs[dt] = ships
        return po, ps, qo, qs

    def walk_with(self, overrides):
        """Per-planet timelines with injected launches.

        overrides: {planet_idx: (ships_delta_at_t0, [(dt, owner, n), ...])}
        — ships_delta for departures from that planet this turn (negative),
        the list for extra arrivals. Returns {planet_idx: (po, ps, qo, qs)}
        for the overridden planets only; un-overridden planets keep the
        base self.post_* timelines.
        """
        out = {}
        for i, (delta, extra) in overrides.items():
            out[i] = self._walk_planet(
                i, self.owner0[i], self.ships0[i] + delta, 0, None, 0,
                extra_list=extra or None)
        return out

    # --------------------------------------------------------- aim + verify
    def aim_at(self, src, tgt, ships):
        """Fixed-point intercept of target tgt from source src.

        Returns (angle, launch_x, launch_y, est_dt) or None.
        """
        sx, sy = self.px[src], self.py[src]
        sr = self.pr[src]
        tr = self.pr[tgt]
        v = fleet_speed(ships)
        tx, ty = self.pos[tgt][0]
        dt_est = 1
        for _ in range(6):
            d = math.hypot(tx - sx, ty - sy) - sr - tr - 0.1
            if d < 0:
                d = 0.0
            new_est = max(1, int(math.ceil(d / v)))
            if new_est > self.horizon:
                return None
            p = self.pos[tgt][new_est]
            if new_est == dt_est and abs(p[0] - tx) < 0.3 and abs(p[1] - ty) < 0.3:
                break
            tx, ty = p
            dt_est = new_est
        angle = math.atan2(ty - sy, tx - sx)
        lx = sx + math.cos(angle) * (sr + 0.1)
        ly = sy + math.sin(angle) * (sr + 0.1)
        return (angle, lx, ly, dt_est)

    def verified_shot(self, src, tgt, ships):
        """Aim, then verify with exact flight. Returns (angle, hit_dt) or None."""
        aim = self.aim_at(src, tgt, ships)
        if aim is None:
            return None
        angle, lx, ly, dt_est = aim
        hit, dt = self.fly(lx, ly, angle, ships, 1)
        if hit == tgt:
            return (angle, dt)
        # one retry: aim at the position one tick later (helps fast orbiters)
        if dt_est + 1 <= self.horizon:
            tx, ty = self.pos[tgt][dt_est + 1]
            sx, sy = self.px[src], self.py[src]
            angle2 = math.atan2(ty - sy, tx - sx)
            lx2 = sx + math.cos(angle2) * (self.pr[src] + 0.1)
            ly2 = sy + math.sin(angle2) * (self.pr[src] + 0.1)
            hit2, dt2 = self.fly(lx2, ly2, angle2, ships, 1)
            if hit2 == tgt:
                return (angle2, dt2)
        return None


def _seg_dist_to_sun(ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(ax - CENTER, ay - CENTER)
    t = ((CENTER - ax) * dx + (CENTER - ay) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(ax + t * dx - CENTER, ay + t * dy - CENTER)


def _resolve_combat(owner, ships, by_owner):
    """Exact engine combat: largest arrival fights second largest; the
    difference fights (or joins) the garrison. Ties annihilate."""
    items = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_owner, top_ships = items[0]
    if len(items) > 1:
        second = items[1][1]
        surv = top_ships - second
        if top_ships == second:
            surv = 0
        surv_owner = top_owner if surv > 0 else -1
    else:
        surv_owner, surv = top_owner, top_ships
    if surv > 0:
        if owner == surv_owner:
            ships += surv
        else:
            ships -= surv
            if ships < 0:
                owner = surv_owner
                ships = -ships
    return owner, ships


def safe_launch(world, i, owner=None):
    """Max ships launchable from planet i now while it survives all
    currently-known incoming fleets (binary search over the ledger walk).
    Returns (max_launch, doomed)."""
    who = world.owner0[i] if owner is None else owner
    g0 = world.ships0[i]
    if g0 <= 0:
        return 0, False
    arr = world.arrivals[i]
    if not arr:
        return g0, False
    has_enemy = any(o != who for slot in arr.values() for o in slot)
    if not has_enemy:
        return g0, False

    def survives(launch):
        _, _, qo, _ = world._walk_planet(i, who, g0 - launch, 0, None, 0)
        return all(o == who for o in qo if o is not None)

    if not survives(0):
        return g0, True          # doomed even at full garrison
    lo, hi = 0, g0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if survives(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo, False


def required_ships(world, tgt, arr_dt, attacker):
    """Min ships arriving at tick arr_dt for `attacker` to own planet tgt at
    the end of that tick. Returns (needed, neutral_garrison_killed) or None."""
    pre_o = world.pre_owner[tgt][arr_dt]
    pre_s = world.pre_ships[tgt][arr_dt]
    if pre_o is None or pre_o == -2:
        return None
    slot = world.arrivals[tgt].get(arr_dt, {})
    mine_already = slot.get(attacker, 0)
    other_max = 0
    for o, s in slot.items():
        if o != attacker and s > other_max:
            other_max = s
    if pre_o == attacker:
        return None
    need_total = other_max + pre_s + 1
    n = need_total - mine_already
    if n < 1:
        n = 1
    neutral_killed = pre_s if pre_o == -1 else 0
    return n, neutral_killed
