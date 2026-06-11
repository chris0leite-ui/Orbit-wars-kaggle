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
PRESSURE_EPS = 0.03        # extra per-ship-per-tick cost scaled by military
                           # pressure: in-flight ships cannot change course,
                           # so committed capital is the army you lack when
                           # the enemy's wave lands — long missions get
                           # expensive exactly when the war is close
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
        "resp_scale",
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

        # Shopping commitment: an enemy garrison is a credible RESPONDER
        # to my captures only to the extent it is not already committed to
        # buying neutrals (each ship can be spent once). Assign each
        # neutral to whoever is nearer; if an owner's claimable pool costs
        # more than their stock, their stock is committed. The scale
        # returns to 1.0 as the pool empties — the war phase arms itself.
        me = self.me
        basket = {}
        stock = {}
        for j in range(n):
            o = self.owner0[j]
            if o >= 0 and o != me:
                stock[o] = stock.get(o, 0) + self.ships0[j]
        if stock:
            for t in range(n):
                if self.owner0[t] != -1 or self.is_comet[t]:
                    continue
                tx, ty = self.px[t], self.py[t]
                dm = 1e18
                d_by = {}
                for j in range(n):
                    o = self.owner0[j]
                    if o == -1 or o == -2:
                        continue
                    d = math.hypot(self.px[j] - tx, self.py[j] - ty)
                    if o == me:
                        if d < dm:
                            dm = d
                    elif d < d_by.get(o, 1e18):
                        d_by[o] = d
                for o, d in d_by.items():
                    if d < dm:
                        basket[o] = basket.get(o, 0) + self.ships0[t] + 1
        # Halve the commitment effect and keep a substantial floor: real
        # opponents do interrupt shopping to defend sometimes; treating
        # them as fully disarmed over-extends (floor 0.2 reshuffled as
        # many games as it fixed in A/B).
        self.resp_scale = {
            o: max(0.55, 1.0 - 0.5 * basket.get(o, 0) / max(1, s))
            for o, s in stock.items()
        }

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
    # strictly exceeds the garrison. Holding afterwards is priced, not
    # required (flow-duration valuation + rollout veto).
    need_total = other_max + pre_s + 1
    n = need_total - mine_already
    if n < 1:
        n = 1
    neutral_killed = pre_s if pre_o == -1 else 0
    return n, neutral_killed


def _reserves(world):
    """Garrison each of my planets must keep against feasible enemy waves.

    Threat model: the enemy can synchronize launches from several planets
    to land together at tick T (each contributor needs eta <= T). For each
    of my planets and each candidate T, the wave is the sum of enemy
    garrisons within reach; my cover is production during the flight,
    friendly fleets already inbound, and ONE assigned helper neighbour
    (each of my planets backs up only its nearest friendly planet, so the
    same garrison is never promised to two defenses). Reserve the worst
    deficit over T. Interior planets reserve ~0; the frontier keeps real
    garrisons.

    Returns dict planet_index -> reserve. Empty reserves when I am
    collapsing (defending a lost position parks the bank; all-in is
    strictly better since combat trades 1:1).
    """
    me = world.me
    n = world.n_planets
    mine = [i for i in range(n) if world.owner0[i] == me
            and not world.is_comet[i]]
    enemies = [e for e in range(n) if world.owner0[e] >= 0
               and world.owner0[e] != me]
    if not mine or not enemies:
        return {i: 0 for i in mine}
    if len(mine) * 3 < len(enemies):
        return {i: 0 for i in mine}          # collapsed: all-in mode

    # assign each of my planets as helper to its nearest friendly planet
    helper_for = {}
    for j in mine:
        best = None
        for i in mine:
            if i == j:
                continue
            d = math.hypot(world.px[j] - world.px[i],
                           world.py[j] - world.py[i])
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            helper_for.setdefault(best[1], []).append((best[0], j))

    T_MAX = 14
    reserves = {}
    for i in mine:
        g = world.ships0[i]
        ix, iy, ir = world.px[i], world.py[i], world.pr[i]
        prod = world.prod[i]
        inbound = 0
        for dt, slot in world.arrivals[i].items():
            if dt <= T_MAX:
                inbound += slot.get(me, 0) - sum(
                    v for o, v in slot.items() if o != me and o >= 0)
        helpers = sorted(helper_for.get(i, []))[:2]
        worst = 0
        for e in enemies:
            se = world.ships0[e]
            if se <= worst:
                continue
            d = math.hypot(world.px[e] - ix, world.py[e] - iy) \
                - world.pr[e] - ir - 0.1
            eta = max(1, int(math.ceil(max(d, 0.0) / fleet_speed(se))))
            if eta > T_MAX:
                continue
            cover = prod * eta + inbound
            for hd, j in helpers:
                h_eta = max(1, int(math.ceil(
                    max(hd - world.pr[j] - ir - 0.1, 0.0)
                    / fleet_speed(max(1, world.ships0[j])))))
                if h_eta + 1 <= eta:
                    cover += world.ships0[j]
            deficit = se - cover
            if deficit > worst:
                worst = deficit
        reserves[i] = min(g, max(0, worst))
    return reserves


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


def _rollout_score(world, base_arrivals, extra_landings, K, flow_weight):
    """Event-driven joint walk of ALL planets K ticks ahead, with a
    reactive opponent model on BOTH sides (pre-emptive just-in-time
    reinforcement and re-snipes of fresh captures). Fleet flight reduces
    to (landing planet, landing tick) because planet motion is
    action-independent; reactions use clean-path analytic etas.

    base_arrivals: {planet: {dt: {owner: ships}}} known in-flight fleets.
    extra_landings: [(tgt, dt, ships)] my decision-set under evaluation.
    Returns my_score - opp_score where score = banked ships + in-flight
    ships + flow_weight * production.
    """
    me = world.me
    n = world.n_planets
    owner = list(world.owner0)
    ships = list(world.ships0)
    prod = world.prod
    alive = world.alive_until
    is_comet = world.is_comet
    pos = world.pos

    sched = {}                  # dt -> {tgt: {owner: ships}}
    for i, by_dt in base_arrivals.items():
        for dt, slot in by_dt.items():
            if dt <= K + 12:
                tgt_slot = sched.setdefault(dt, {}).setdefault(i, {})
                for o, v in slot.items():
                    tgt_slot[o] = tgt_slot.get(o, 0) + v
    for tgt, dt, v in extra_landings:
        if dt <= K + 12:
            tgt_slot = sched.setdefault(dt, {}).setdefault(tgt, {})
            tgt_slot[me] = tgt_slot.get(me, 0) + v

    reacted = set()

    def react_launch(side, tgt, need, now, deadline):
        """Side launches `need` ships at tgt, landing by `deadline`, from
        its nearest able planet — or split across the two nearest if no
        single planet suffices. Returns True if (fully) launched."""
        tx, ty = pos[tgt][min(deadline, world.horizon)]
        reach = []
        for j in range(n):
            if owner[j] != side or j == tgt:
                continue
            avail = ships[j] - 1
            if avail < 1:
                continue
            d = math.hypot(pos[j][now][0] - tx, pos[j][now][1] - ty) \
                - world.pr[j] - world.pr[tgt]
            eta = max(1, int(math.ceil(
                max(d, 0.0) / fleet_speed(max(need, 2)))))
            if now + 1 + eta > deadline:
                continue
            reach.append((d, j, avail))
        if not reach:
            return False
        reach.sort()
        total = 0
        used = []
        for d, j, avail in reach[:2]:
            take = min(avail, need - total)
            if take > 0:
                used.append((j, take))
                total += take
            if total >= need:
                break
        if total < need:
            return False
        land = now + 1 + max(1, int(math.ceil(max(reach[0][0], 0.0)
                                              / fleet_speed(need))))
        for j, take in used:
            ships[j] -= take
        tgt_slot = sched.setdefault(land, {}).setdefault(tgt, {})
        tgt_slot[side] = tgt_slot.get(side, 0) + total
        return True

    for dt in range(1, K + 1):
        for i in range(n):
            if owner[i] >= 0:
                ships[i] += prod[i]
            if is_comet[i] and dt >= alive[i] and owner[i] != -2:
                owner[i] = -2
                ships[i] = 0
        flips = []
        slot_map = sched.get(dt)
        if slot_map:
            for i, by_owner in slot_map.items():
                if owner[i] == -2:
                    continue
                old = owner[i]
                o2, s2 = _resolve_combat(old, ships[i], by_owner)
                owner[i] = o2
                ships[i] = s2
                if o2 != old and o2 >= 0:
                    flips.append((i, old, o2))
        # ---- reactions (both sides)
        # R2: re-snipe fresh captures
        for i, old, new in flips:
            if old < 0:
                continue
            key = ("resnipe", i, new)
            if key in reacted:
                continue
            reacted.add(key)
            need = ships[i] + prod[i] * 4 + 2
            react_launch(old, i, need, dt, dt + 8)
        # R1: pre-emptive JIT defense against scheduled hostile arrivals
        for fdt in range(dt + 1, min(dt + 7, K + 12)):
            fmap = sched.get(fdt)
            if not fmap:
                continue
            for i, by_owner in fmap.items():
                d_side = owner[i]
                if d_side < 0:
                    continue
                hostile = sum(v for o, v in by_owner.items() if o != d_side)
                if hostile <= 0:
                    continue
                key = ("jit", i, fdt)
                if key in reacted:
                    continue
                cover = ships[i] + prod[i] * (fdt - dt) \
                    + by_owner.get(d_side, 0)
                if hostile <= cover:
                    continue
                reacted.add(key)
                react_launch(d_side, i, hostile - cover + 1, dt, fdt)

    # in-flight ships beyond K still belong to their owner
    inflight = {}
    for dt in range(K + 1, K + 13):
        for i, by_owner in sched.get(dt, {}).items():
            for o, v in by_owner.items():
                inflight[o] = inflight.get(o, 0) + v
    my_score = 0.0
    opp_score = 0.0
    for i in range(n):
        if owner[i] == me:
            my_score += ships[i] + flow_weight * prod[i]
        elif owner[i] >= 0:
            opp_score += ships[i] + flow_weight * prod[i]
    my_score += inflight.get(me, 0)
    for o, v in inflight.items():
        if o != me and o >= 0:
            opp_score += v
    return my_score - opp_score


def _response_curve(world, tgt):
    """Enemy mass that can land on tgt by tick t, launched AFTER my fleet
    becomes visible (launch happens 1 tick after mine). Sorted cumulative
    [(land_tick, cumulative_mass)]. Conservative: counts each enemy
    garrison at full strength for every target it could reach."""
    me = world.me
    tx, ty, tr = world.px[tgt], world.py[tgt], world.pr[tgt]
    scale = getattr(world, "resp_scale", {})
    items = []
    for e in range(world.n_planets):
        o = world.owner0[e]
        if o < 0 or o == me or e == tgt:
            continue
        se = int(world.ships0[e] * scale.get(o, 1.0))
        if se <= 0:
            continue
        d = math.hypot(world.px[e] - tx, world.py[e] - ty) \
            - world.pr[e] - tr - 0.1
        land = 1 + max(1, int(math.ceil(max(d, 0.0) / fleet_speed(se))))
        items.append((land, se))
    items.sort()
    out = []
    cum = 0
    for land, se in items:
        cum += se
        out.append((land, cum))
    return out


def agent(obs, configuration=None):
    try:
        return _agent_inner(obs, configuration)
    except Exception:
        # never throw a game on an internal error: an empty action is a
        # legal no-op turn; production continues and we re-plan next tick
        return []


def _agent_inner(obs, configuration=None):
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
    defense_budget = {}
    doomed = {}
    reserves = _reserves(world)
    for i in mine:
        avail, is_doomed = _safe_launch(world, i)
        defense_budget[i] = max(0, avail)   # reserves ARE spendable on defense
        if not is_doomed and avail > 0 and not world.is_comet[i]:
            avail = min(avail, world.ships0[i] - reserves.get(i, 0))
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
            if s == i:
                continue
            spendable = defense_budget.get(s, 0) - committed.get(s, 0)
            if spendable <= 0:
                continue
            d = math.hypot(world.px[s] - world.px[i], world.py[s] - world.py[i])
            helpers.append((d, s))
        helpers.sort()
        # coalition rescue: helpers contribute nearest-first (a single
        # avalanche wave often exceeds any one neighbour's spare but not
        # several together). Each contribution is tentative; if even the
        # full coalition cannot save the planet, everything rolls back —
        # never feed a lost defense.
        tentative = []        # (src, k, angle, dt)

        def survives_now():
            _, _, qo, _ = world._walk_planet(i, me, world.ships0[i],
                                             0, None, 0)
            return all(o == me for o in qo if o is not None)

        for _, s in helpers:
            cap = defense_budget[s] - committed.get(s, 0) \
                - sum(k for src, k, _, _ in tentative if src == s)
            if cap <= 0:
                continue
            # binary search the smallest contribution from THIS helper
            # that (together with earlier tentative ones) saves the planet
            lo, hi = 1, cap
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                shot = world.verified_shot(s, i, mid)
                if shot is None:
                    break
                angle, dt = shot
                if dt > fall_t:
                    break     # arrives too late at any size
                slot = world.arrivals[i].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + mid
                ok = survives_now()
                slot[me] -= mid
                if not slot[me]:
                    del slot[me]
                if ok:
                    found = (mid, angle, dt)
                    hi = mid - 1
                else:
                    lo = mid + 1
            if found is not None:
                k, angle, dt = found
                tentative.append((s, k, angle, dt))
                slot = world.arrivals[i].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + k
                saved = True
                break
            # this helper alone (plus earlier ones) is not enough: commit
            # its full spare to the coalition and move to the next helper
            shot = world.verified_shot(s, i, cap)
            if shot is None or shot[1] > fall_t:
                continue
            angle, dt = shot
            tentative.append((s, cap, angle, dt))
            slot = world.arrivals[i].setdefault(dt, {})
            slot[me] = slot.get(me, 0) + cap
        if saved:
            for s, k, angle, dt in tentative:
                spend(s, k, angle)
            # refresh this planet's ledger rows
            po, ps, qo, qs = world._walk_planet(i, me, world.ships0[i],
                                                0, None, 0)
            world.pre_owner[i] = po
            world.pre_ships[i] = ps
            world.post_owner[i] = qo
            world.post_ships[i] = qs
        else:
            # roll back tentative coalition arrivals
            for s, k, angle, dt in tentative:
                slot = world.arrivals[i].get(dt)
                if slot:
                    slot[me] = slot.get(me, 0) - k
                    if not slot[me]:
                        del slot[me]
        if not saved:
            # cannot save: free the garrison for missions (evacuation)
            budget[i] = world.ships0[i]
            doomed[i] = True

    # ----- 3. offense: price all (source, target) captures off the ledger
    # military pressure: enemy garrison mass deliverable into my zone within
    # ~12 ticks, relative to my total garrison
    my_total = sum(world.ships0[i] for i in mine) + 1
    near_enemy = 0
    for e in range(n):
        o = world.owner0[e]
        if o < 0 or o == me:
            continue
        se = world.ships0[e]
        if se <= 0:
            continue
        v_e = fleet_speed(se)
        for i in mine:
            d = math.hypot(world.px[e] - world.px[i],
                           world.py[e] - world.py[i]) - world.pr[e] - world.pr[i]
            if d / v_e <= 12.0:
                near_enemy += se
                break
    pressure = near_enemy / my_total
    my_prod_total = sum(world.prod[i] for i in mine)
    prod_by_owner = {}
    for i in range(n):
        o = world.owner0[i]
        if o >= 0 and o != me:
            prod_by_owner[o] = prod_by_owner.get(o, 0) + world.prod[i]
    # reference = the strongest single opponent (in 2P this is THE
    # opponent; in free-for-all ranking against the leader is what counts)
    their_prod_total = max(prod_by_owner.values()) if prod_by_owner else 0
    if my_prod_total < 0.9 * their_prod_total:
        pressure = 0.0      # production-behind: converting bank into
                            # production is mandatory; no liquidity tax

    offense_start = len(moves)
    base_arrivals = {i: {dt: dict(slot)
                         for dt, slot in world.arrivals[i].items()}
                     for i in range(n)}
    purchases = []            # [(move list, [(tgt, dt, ships), ...]), ...]
    if time.perf_counter() - t0 < TIME_BUDGET:
        live_enemies = len({world.owner0[i] for i in range(n)
                            if world.owner0[i] >= 0 and world.owner0[i] != me})
        deny_mult = 1.0 + (1.0 / max(1, live_enemies)) if live_enemies else 1.0
        # free-for-all: fighting any one opponent is negative-sum for both
        # of us relative to the bystanders — expand and defend instead,
        # except for cheap opportunistic snipes
        ffa = live_enemies >= 2
        # projected final score per player as the board stands: only the
        # projected WINNER matters (score is ranked; 2nd pays like last)
        proj = {}
        for j in range(n):
            o = world.owner0[j]
            if o >= 0:
                proj[o] = proj.get(o, 0.0) + world.ships0[j] \
                    + world.prod[j] * world.remaining
        for f in world.fleets:
            if f[1] >= 0:
                proj[f[1]] = proj.get(f[1], 0.0) + f[6]
        rivals = {o: v for o, v in proj.items() if o != me}
        leader = max(rivals, key=rivals.get) if rivals else None
        i_lead = leader is None or proj.get(me, 0.0) >= rivals[leader]
        # behind-endgame gambit: when projection-behind late, quiet play
        # locks in the loss — risk becomes free (applies in 2P too)
        gambit = (not i_lead) and world.remaining < 120
        if gambit:
            pressure = 0.0
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

        resp_cache = {}

        def hold_ticks(tgt, arr_dt, surplus):
            """How long my capture survives the enemy's feasible response:
            first tick where their landable mass (launched after seeing my
            fleet) exceeds my surplus + production since capture."""
            curve = resp_cache.get(tgt)
            if curve is None:
                curve = _response_curve(world, tgt)
                resp_cache[tgt] = curve
            prod = world.prod[tgt]
            for land, cum in curve:
                t = max(land, arr_dt + 1)
                if cum > surplus + prod * (t - arr_dt):
                    return max(1, t - arr_dt)
            return None          # no feasible response: hold to the end

        def price_target(tgt, arr_dt, n_need, neutral_killed, surplus,
                         flow_known=None):
            pre_o = world.pre_owner[tgt][arr_dt]
            remaining = max(0, world.remaining - arr_dt)
            held = hold_ticks(tgt, arr_dt, surplus)
            flow = remaining if held is None else min(remaining, held)
            if flow_known is not None:
                flow = min(flow, flow_known)
            if world.is_comet[tgt]:
                flow = min(flow, max(0, world.alive_until[tgt] - arr_dt - 1))
                value = float(flow) - neutral_killed
            elif pre_o == -1:
                value = world.prod[tgt] * flow - float(neutral_killed)
                race = _enemy_best_eta(world, tgt,
                                       world.pre_ships[tgt][arr_dt] + 1)
                if race < arr_dt:
                    value *= RACE_DISCOUNT
            else:
                value = deny_mult * world.prod[tgt] * flow
                if ffa:
                    if pre_o == leader and not i_lead:
                        pass          # hitting the projected winner is the
                                      # only attack that changes my outcome
                    elif n_need > 3 * world.prod[tgt] + 6:
                        value *= 0.2  # brawling non-leaders helps the leader
            value *= GAMMA ** arr_dt
            value -= (TRAVEL_EPS + PRESSURE_EPS * pressure) * n_need * arr_dt
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
                                             neutral_killed, 1, None)
                        growth = sum(world.prod[s] for s, _, _ in group)
                        turns_short = ((n_need - total_spare) /
                                       max(1, growth))
                        if value > 0 and turns_short <= 6:
                            return (value, value / n_need, None,
                                    [s for s, _, _ in group])
                        return None
                    continue      # try a bigger coalition
                # allocate shares nearest-first and iterate to consistency:
                # smaller shares fly slower, land later, face a regrown
                # garrison — recompute the requirement at the actual
                # arrival ticks and grow the demand until the joint walk
                # confirms the capture (or the coalition can't afford it).
                plan = None
                sched = None
                qo = qs = None
                arr_dt = None
                demand = n_need
                for _ in range(3):
                    trial = []
                    left = demand
                    tsched = []
                    for s, sp, _ in group:
                        take = min(sp, left)
                        if take <= 0:
                            continue
                        shot = world.verified_shot(s, tgt, take)
                        if shot is None:
                            trial = None
                            break
                        angle, dt = shot
                        trial.append((s, angle, take, dt))
                        tsched.append((dt, me, take))
                        left -= take
                    if trial is None or left > 0:
                        break
                    _, _, tqo, tqs = world._walk_planet(
                        tgt, world.owner0[tgt], world.ships0[tgt], 0,
                        None, 0, extra_list=tsched)
                    t_arr = max(dt for dt, _, _ in tsched)
                    if tqo[t_arr] == me:
                        plan, sched, qo, qs, arr_dt = (
                            trial, tsched, tqo, tqs, t_arr)
                        break
                    # capture failed: grow demand by the observed deficit
                    demand += max(1, tqs[t_arr] + 1)
                    if demand > total_spare:
                        break
                if plan is None:
                    continue          # try a bigger coalition
                n_need = demand
                surplus = qs[arr_dt]
                # flow limit from KNOWN fleets (ledger walk with my plan)
                flow_known = None
                for t in range(arr_dt + 1, world.horizon + 1):
                    if qo[t] is not None and qo[t] != me:
                        flow_known = t - arr_dt
                        break
                value = price_target(tgt, arr_dt, n_need, neutral_killed,
                                     surplus, flow_known)
                # the analytic response curve assumes the enemy's WHOLE
                # army answers each target; their response budget is in
                # fact shared across my simultaneous attacks. Admit
                # slightly-negative plans — the rollout veto (which spends
                # enemy garrisons exactly) makes the final call.
                admit_floor = -0.15 * n_need if (not ffa or gambit) \
                    else 0.0
                if value <= admit_floor:
                    continue          # a bigger coalition may hold longer
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
                order.append((res[0], res[1], tgt))
        order.sort(reverse=True)
        # bank toward the best unaffordable plan ONLY if it clearly
        # dominates the best affordable one — otherwise saving starves the
        # opening on maps where the top target is just out of reach while
        # excellent affordable targets sit next door
        best_affordable_value = 0.0
        best_wish_value = 0.0
        for v, _, tgt2 in order:
            res2 = None
            if best_affordable_value == 0.0 or best_wish_value == 0.0:
                res2 = plan_attack(tgt2)
            if res2 is None:
                continue
            if res2[2] is not None and best_affordable_value == 0.0:
                best_affordable_value = res2[0]
            elif res2[2] is None and res2[3] and best_wish_value == 0.0:
                best_wish_value = res2[0]
            if best_affordable_value and best_wish_value:
                break
        allow_banking = best_wish_value > 1.5 * best_affordable_value
        banked = False
        for _, _, tgt in order:
            if time.perf_counter() - t0 > TIME_BUDGET:
                break
            res = plan_attack(tgt)   # reprice with current budgets
            if res is None:
                continue
            value, density, plan, wish = res
            if plan is None:
                if allow_banking and not banked and wish:
                    banked = True
                    for s in wish:
                        committed[s] = budget[s]    # freeze for this turn
                continue
            start = len(moves)
            landings = []
            for s, angle, take, dt in plan:
                spend(s, take, angle)
                slot = world.arrivals[tgt].setdefault(dt, {})
                slot[me] = slot.get(me, 0) + take
                landings.append((tgt, dt, take))
            purchases.append((moves[start:], landings))

    # ----- 4. rollout selection: validate the offense against a reactive
    # opponent; keep the best of {all, drop-one each, defense-only}
    if purchases and time.perf_counter() - t0 < TIME_BUDGET:
        K = 18
        flow_weight = float(min(world.remaining, 60))
        variants = [list(range(len(purchases)))]
        if len(purchases) > 1:
            for drop in range(len(purchases)):
                variants.append([k for k in range(len(purchases))
                                 if k != drop])
        variants.append([])
        best_keep = None
        best_score = None
        for keep in variants:
            extra = []
            for k in keep:
                extra.extend(purchases[k][1])
            score = _rollout_score(world, base_arrivals, extra, K,
                                   flow_weight)
            if best_score is None or score > best_score + 1e-9:
                best_score = score
                best_keep = keep
            if time.perf_counter() - t0 > TIME_BUDGET:
                break
        if best_keep is not None and len(best_keep) < len(purchases):
            moves = moves[:offense_start]
            for k in best_keep:
                moves.extend(purchases[k][0])

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
