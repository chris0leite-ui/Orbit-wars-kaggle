"""Ledger agent — exact-future economist for Orbit Wars.

Single self-contained file: this IS the submission (no bundler).

Premise (first principles):
  Every combat in this game destroys ships 1:1 on both sides, so fighting
  the enemy never moves the final-score differential. The differential is
  moved only by:
    1. production flow  — planets held x turns held,
    2. neutral garrisons paid to expand (a pure sink),
    3. waste            — sun hits, out-of-bounds, comet departures,
                          tie annihilations.
  The game is deterministic and fully observable, so the exact future of
  every planet (owner + garrison per tick, given all in-flight fleets) is
  computable. This agent builds that future every turn ("the ledger") and
  greedily buys the highest net-present-value actions priced exactly off
  it: captures, snipes, defenses, evacuations. Snipe timing, defense
  timing and dodge behaviour all fall out of the one mechanism.

Physics is mirrored from the engine (kaggle_environments orbit_wars):
  per tick: launches -> production -> fleet moves with swept-disk
  collision (planets checked in list order, then out-of-bounds, then sun)
  -> planet rotation / comet path advance -> combat.
"""

import math
import time

# ---------------------------------------------------------------- constants
BOARD = 100.0
CENTER = 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
LOG1000 = math.log(1000.0)
EPISODE_END = 498          # interpreter flags DONE at step >= episodeSteps-2

# ------------------------------------------------------------------- knobs
HORIZON = 90               # forecast depth in ticks
HOLD_TICKS = 8             # captured planet must survive this long vs known fleets
RACE_DISCOUNT = 0.35       # value multiplier when enemy can reach a neutral first
TRAVEL_EPS = 0.012         # per-ship-per-tick tempo cost (tie-break toward near)
GAMMA = 0.97               # forecast decay per tick of flight time: the
                           # no-new-launches ledger loses validity with depth
TIME_BUDGET = 0.70         # seconds per turn before we stop adding missions
COMET_EVAC_MARGIN = 2      # evacuate owned comets this many ticks before exit


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + 5.0 * (math.log(ships) / LOG1000) ** 1.5
    return s if s < 6.0 else 6.0


def _g(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


# ------------------------------------------------------------------- world
class World:
    """Per-turn snapshot + exact position table + ledger."""

    __slots__ = (
        "step", "me", "n_planets", "pid", "owner0", "px", "py", "pr",
        "ships0", "prod", "pos", "alive_until", "idx_of", "fleets",
        "omega", "horizon", "arrivals", "pre_owner", "pre_ships",
        "post_owner", "post_ships", "is_comet", "remaining",
    )

    def __init__(self, obs):
        self.step = int(_g(obs, "step", 0) or 0)
        self.me = int(_g(obs, "player", 0) or 0)
        planets = [list(p) for p in (_g(obs, "planets", []) or [])]
        self.fleets = [list(f) for f in (_g(obs, "fleets", []) or [])]
        self.omega = float(_g(obs, "angular_velocity", 0.03) or 0.03)
        self.horizon = min(HORIZON, max(8, EPISODE_END - self.step))
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

        # Comet path lookup: pid -> (path list, current index)
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
                    # Comet with no group data (shouldn't happen): static.
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
                # vanish tick: when index overruns path end
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
                # AABB prune
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
                # swept pair quadratic
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
            # sun: distance from segment (x,y)-(nx,ny) to center
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
        """
        H = self.horizon
        n = self.n_planets
        arrivals = [dict() for _ in range(n)]   # i -> {dt: {owner: ships}}
        for f in self.fleets:
            hit, dt = self.fly(float(f[2]), float(f[3]), float(f[4]),
                               int(f[6]), 1)
            if hit is not None:
                slot = arrivals[hit].setdefault(dt, {})
                slot[f[1]] = slot.get(f[1], 0) + int(f[6])
        self.arrivals = arrivals

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


# --------------------------------------------------------------- decisions
def _safe_launch(world, i):
    """Max ships launchable from my planet i now while it survives all
    currently-known incoming fleets. Doomed planets return full garrison
    (evacuation beats donating a 1:1 garrison fight only when the ships
    have somewhere better to be — the allocator decides)."""
    g0 = world.ships0[i]
    if g0 <= 0:
        return 0, False
    arr = world.arrivals[i]
    if not arr:
        return g0, False
    me = world.me
    has_enemy = any(o != me for slot in arr.values() for o in slot)
    if not has_enemy:
        return g0, False

    def survives(launch):
        _, _, qo, _ = world._walk_planet(i, me, g0 - launch, 0, None, 0)
        return all(o == me for o in qo if o is not None)

    if not survives(0):
        return g0, True          # doomed even at full garrison
    lo, hi = 0, g0               # survives(lo) True; find max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if survives(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo, False


def _required_ships(world, tgt, arr_dt, hold):
    """Min ships arriving at tick arr_dt for ME to own planet tgt at the
    end of that tick AND hold it for `hold` ticks vs known fleets.
    Returns (needed, neutral_garrison_killed) or None if impossible/pointless."""
    me = world.me
    pre_o = world.pre_owner[tgt][arr_dt]
    pre_s = world.pre_ships[tgt][arr_dt]
    if pre_o is None or pre_o == -2:
        return None
    slot = world.arrivals[tgt].get(arr_dt, {})
    mine_already = slot.get(me, 0)
    other_max = 0
    for o, s in slot.items():
        if o != me and s > other_max:
            other_max = s
    if pre_o == me:
        return None
    # need: my total strictly tops other arrivals, and the surplus
    # strictly exceeds the garrison
    need_total = other_max + pre_s + 1
    n = need_total - mine_already
    if n < 1:
        n = 1
    # hold check vs already-known later arrivals
    for _ in range(4):
        _, _, qo, qs = world._walk_planet(
            tgt, world.post_owner[tgt][arr_dt - 1] if arr_dt > 0 else world.owner0[tgt],
            world.post_ships[tgt][arr_dt - 1] if arr_dt > 0 else world.ships0[tgt],
            me, arr_dt, n, start_dt=arr_dt)
        end = min(arr_dt + hold, world.horizon)
        ok = all(qo[t] == me for t in range(arr_dt, end + 1) if qo[t] is not None)
        if ok:
            break
        deficit = 0
        for t in range(arr_dt, end + 1):
            if qo[t] is not None and qo[t] != me:
                deficit = max(deficit, qs[t] + 1)
        n += max(1, deficit)
        if n > 4000:
            return None
    neutral_killed = pre_s if pre_o == -1 else 0
    return n, neutral_killed


def _reserve(world, i, budget_hint):
    """Garrison to keep at my planet i against a feasible enemy first strike.

    For each enemy planet e: if e launched its whole garrison at i right
    now, it lands at tick eta_e. My cover by then = garrison + production
    + already-inbound friendly fleets + neighbours' garrisons that could
    respond after seeing the launch (launch visible next tick, so a helper
    must have flight time <= eta_e - 1). Reserve the worst-case deficit.
    Interior planets therefore reserve ~0; frontier planets keep real
    garrisons. Helpers are counted optimistically (the same helper covers
    several planets) — first-order just-in-time defense.
    """
    me = world.me
    g = world.ships0[i]
    if g <= 0:
        return 0
    ix, iy = world.px[i], world.py[i]
    ir = world.pr[i]
    prod = world.prod[i]
    inbound = 0
    for dt, slot in world.arrivals[i].items():
        inbound += slot.get(me, 0)
    worst = 0
    for e in range(world.n_planets):
        o = world.owner0[e]
        if o < 0 or o == me:
            continue
        se = world.ships0[e]
        if se <= worst:                       # can't beat current worst
            continue
        d = math.hypot(world.px[e] - ix, world.py[e] - iy) \
            - world.pr[e] - ir - 0.1
        eta_e = max(1, int(math.ceil(max(d, 0.0) / fleet_speed(se))))
        # cover excludes my garrison: the reserve IS the garrison part
        cover = prod * eta_e + inbound
        if cover >= se:
            continue
        for j in range(world.n_planets):
            if j == i or world.owner0[j] != me:
                continue
            gj = world.ships0[j]
            if gj <= 0:
                continue
            dj = math.hypot(world.px[j] - ix, world.py[j] - iy) \
                - world.pr[j] - ir - 0.1
            if dj <= 0:
                continue
            if math.ceil(dj / fleet_speed(gj)) + 1 <= eta_e:
                cover += gj
                if cover >= se:
                    break
        deficit = se - cover
        if deficit > worst:
            worst = deficit
    return min(g, max(0, worst))


def _enemy_best_eta(world, tgt, size_hint):
    """Fastest tick any enemy planet could land `size_hint` ships on tgt."""
    me = world.me
    best = 1 << 30
    tx, ty = world.px[tgt], world.py[tgt]
    v = fleet_speed(max(size_hint, 1))
    for j in range(world.n_planets):
        o = world.owner0[j]
        if o == me or o < 0:
            continue
        if world.ships0[j] < size_hint:
            continue
        d = math.hypot(world.px[j] - tx, world.py[j] - ty) \
            - world.pr[j] - world.pr[tgt]
        eta = max(1, int(math.ceil(max(d, 0.0) / v)))
        if eta < best:
            best = eta
    return best


def agent(obs, configuration=None):
    t0 = time.perf_counter()
    world = World(obs)
    me = world.me
    n = world.n_planets
    if n == 0:
        return []
    world.build_ledger()

    mine = [i for i in range(n) if world.owner0[i] == me]
    if not mine:
        return []

    # ----- per-source launchable budget (defense reservations baked in)
    budget = {}
    doomed = {}
    for i in mine:
        avail, is_doomed = _safe_launch(world, i)
        if not is_doomed and avail > 0 and not world.is_comet[i]:
            avail = min(avail, world.ships0[i] - _reserve(world, i, avail))
        budget[i] = max(0, avail)
        doomed[i] = is_doomed

    moves = []
    committed = {}            # src -> ships committed this turn

    def can_spend(src, k):
        return budget[src] - committed.get(src, 0) >= k

    def spend(src, k, angle):
        committed[src] = committed.get(src, 0) + k
        moves.append([world.pid[src], angle, int(k)])

    # ----- 1. mandatory: evacuate owned comets about to leave
    for i in mine:
        if not world.is_comet[i]:
            continue
        exit_dt = world.alive_until[i]
        if exit_dt > COMET_EVAC_MARGIN + 1:
            continue
        g = world.ships0[i]
        if g <= 0:
            continue
        # send everything to the best reachable planet (prefer own/neutral)
        best = None
        for j in range(n):
            if j == i or world.is_comet[j]:
                continue
            shot = world.verified_shot(i, j, g)
            if shot is None:
                continue
            angle, dt = shot
            pref = 0 if world.post_owner[j][dt] == me else 1
            key = (pref, dt)
            if best is None or key < best[0]:
                best = (key, angle)
        if best is not None:
            spend(i, g, best[1])
            budget[i] = 0

    # ----- 2. defense: rescue own planets that the ledger says will fall
    fall = []                 # (fall_tick, planet)
    for i in range(n):
        if world.owner0[i] != me or world.is_comet[i]:
            continue
        for t in range(1, world.horizon + 1):
            o = world.post_owner[i][t]
            if o is not None and o != me and o != -2:
                fall.append((t, i))
                break
    fall.sort()
    for fall_t, i in fall:
        if doomed.get(i):
            continue          # this is one of mine that even full garrison can't save... handled below
        # find the cheapest single reinforcement that saves it
        saved = False
        helpers = []
        for s in mine:
            if s == i or budget[s] - committed.get(s, 0) <= 0:
                continue
            d = math.hypot(world.px[s] - world.px[i], world.py[s] - world.py[i])
            helpers.append((d, s))
        helpers.sort()
        for _, s in helpers:
            cap = budget[s] - committed.get(s, 0)
            # binary search smallest reinforcement that keeps the planet
            lo, hi = 1, cap
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                shot = world.verified_shot(s, i, mid)
                if shot is None:
                    break
                angle, dt = shot
                if dt > fall_t:
                    break     # arrives too late at any size (bigger=faster, try bigger)
                slot = world.arrivals[i].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + mid
                _, _, qo, _ = world._walk_planet(i, me, world.ships0[i], 0, None, 0)
                slot[me] -= mid
                if not slot[me]:
                    del slot[me]
                if all(o == me for o in qo if o is not None):
                    found = (mid, angle, dt)
                    hi = mid - 1
                else:
                    lo = mid + 1
            if found is not None:
                k, angle, dt = found
                spend(s, k, angle)
                slot = world.arrivals[i].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + k
                # refresh this planet's ledger rows
                po, ps, qo, qs = world._walk_planet(i, me, world.ships0[i], 0, None, 0)
                world.pre_owner[i] = po
                world.pre_ships[i] = ps
                world.post_owner[i] = qo
                world.post_ships[i] = qs
                saved = True
                break
        if not saved:
            # cannot save: free the garrison for missions (evacuation)
            budget[i] = world.ships0[i]
            doomed[i] = True

    # ----- 3. offense: price all (source, target) captures off the ledger
    if time.perf_counter() - t0 < TIME_BUDGET:
        live_enemies = len({world.owner0[i] for i in range(n)
                            if world.owner0[i] >= 0 and world.owner0[i] != me})
        deny_mult = 1.0 + (1.0 / max(1, live_enemies)) if live_enemies else 1.0
        candidates = []
        for s in mine:
            cap = budget[s] - committed.get(s, 0)
            if cap <= 0:
                continue
            sx, sy = world.px[s], world.py[s]
            for tgt in range(n):
                if tgt == s:
                    continue
                last_owner = world.post_owner[tgt][world.horizon]
                if last_owner == me:
                    continue
                if world.alive_until[tgt] < 4:
                    continue
                d = math.hypot(world.px[tgt] - sx, world.py[tgt] - sy)
                # coarse reachability prune
                eta0 = max(1, int(d / 6.0))
                if eta0 > world.horizon - 2:
                    continue
                # size guess from ledger at coarse eta
                size_guess = max(1, int((d - world.pr[s] - world.pr[tgt])
                                        / fleet_speed(max(1, cap))))
                size_guess = min(size_guess, world.horizon)
                pre_s = world.pre_ships[tgt][size_guess]
                if pre_s is None:
                    continue
                candidates.append((d, s, tgt))
        candidates.sort()

        # group candidate sources per target, nearest first
        by_target = {}
        for d, s, tgt in candidates:
            by_target.setdefault(tgt, []).append((d, s))

        def price_target(tgt, arr_dt, n_need, neutral_killed):
            pre_o = world.pre_owner[tgt][arr_dt]
            remaining = max(0, world.remaining - arr_dt)
            if world.is_comet[tgt]:
                flow = min(remaining,
                           max(0, world.alive_until[tgt] - arr_dt - 1))
                value = float(flow) - neutral_killed
            elif pre_o == -1:
                value = world.prod[tgt] * remaining - float(neutral_killed)
                race = _enemy_best_eta(world, tgt,
                                       world.pre_ships[tgt][arr_dt] + 1)
                if race < arr_dt:
                    value *= RACE_DISCOUNT
            else:
                value = deny_mult * world.prod[tgt] * remaining
            value *= GAMMA ** arr_dt
            value -= TRAVEL_EPS * n_need * arr_dt
            return value

        def plan_attack(tgt):
            """Best affordable strike on tgt, or a 'wish' if worth saving for.

            Returns (value, density, plan, wish_members) where plan is
            [(src, angle, n, dt), ...] when affordable, else None and
            wish_members lists the sources whose budgets should be banked.
            Coalition pricing is exact: every member's shot is verified
            with real flight physics and the joint arrival schedule is
            re-walked through the target's ledger.
            """
            srcs = by_target.get(tgt)
            if not srcs:
                return None
            members = []          # (src, spare, shot at full spare)
            for d, s in sorted(srcs):
                spare = budget[s] - committed.get(s, 0)
                if spare <= 0:
                    continue
                shot = world.verified_shot(s, tgt, max(1, spare))
                if shot is None:
                    continue
                members.append((s, spare, shot))
                if len(members) >= 5:
                    break
            if not members:
                return None
            # try growing coalitions: 1 source, then 2, ... nearest first
            for k in range(1, len(members) + 1):
                group = members[:k]
                total_spare = sum(sp for _, sp, _ in group)
                latest = max(shot[1] for _, _, shot in group)
                rq = _required_ships(world, tgt, latest, HOLD_TICKS)
                if rq is None:
                    return None
                n_need, neutral_killed = rq
                if n_need > total_spare:
                    if k == len(members):
                        # unaffordable even all-in: bank toward it if the
                        # garrisons will cover it within a few turns
                        value = price_target(tgt, latest, n_need,
                                             neutral_killed)
                        growth = sum(world.prod[s] for s, _, _ in group)
                        turns_short = ((n_need - total_spare) /
                                       max(1, growth))
                        if value > 0 and turns_short <= 6:
                            return (value, value / n_need, None,
                                    [s for s, _, _ in group])
                        return None
                    continue      # try a bigger coalition
                # allocate shares nearest-first, re-verify each at its size
                plan = []
                left = n_need
                sched = []
                for s, sp, _ in group:
                    take = min(sp, left)
                    if take <= 0:
                        continue
                    shot = world.verified_shot(s, tgt, take)
                    if shot is None:
                        plan = None
                        break
                    angle, dt = shot
                    plan.append((s, angle, take, dt))
                    sched.append((dt, me, take))
                    left -= take
                if plan is None or left > 0:
                    continue
                # exact joint check: does this schedule actually take + hold?
                _, _, qo, _ = world._walk_planet(
                    tgt, world.owner0[tgt], world.ships0[tgt], 0, None, 0,
                    extra_list=sched)
                arr_dt = max(dt for dt, _, _ in sched)
                lastt = min(arr_dt + HOLD_TICKS, world.horizon)
                if not all(qo[t] in (me, None) for t in range(arr_dt,
                                                              lastt + 1)):
                    continue
                value = price_target(tgt, arr_dt, n_need, neutral_killed)
                if value <= 0:
                    return None
                return (value, value / n_need, plan, None)
            return None

        # price every target, buy best-density first, reprice as we commit;
        # unaffordable-but-best plans bank their sources (saving emerges)
        order = []
        for tgt in by_target:
            if time.perf_counter() - t0 > TIME_BUDGET:
                break
            res = plan_attack(tgt)
            if res is not None:
                order.append((res[1], res[0], tgt))
        order.sort(reverse=True)
        banked = False
        for _, _, tgt in order:
            if time.perf_counter() - t0 > TIME_BUDGET:
                break
            res = plan_attack(tgt)   # reprice with current budgets
            if res is None:
                continue
            value, density, plan, wish = res
            if plan is None:
                if not banked and wish:
                    banked = True
                    for s in wish:
                        committed[s] = budget[s]    # freeze for this turn
                continue
            for s, angle, take, dt in plan:
                spend(s, take, angle)
                slot = world.arrivals[tgt].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + take

    # final safety: never exceed the actual garrison
    by_src = {}
    out = []
    for pid, angle, k in moves:
        i = world.idx_of[pid]
        used = by_src.get(pid, 0)
        room = world.ships0[i] - used
        if room <= 0:
            continue
        k = min(k, room)
        if k < 1:
            continue
        by_src[pid] = used + k
        out.append([pid, float(angle), int(k)])
    return out
