"""Oracle agent — feature extraction (shared by training and runtime).

The value model sees the exact no-new-launches forecast, not just the
current frame: per-player planet/production/garrison/in-flight/score
trajectories probed at several future depths, plus posture (concentration,
frontline geometry), expansion economics (the neutral pool and who reaches
it first), and phase. A candidate plan (our launches, an opponent's reply)
is injected into the ledger before extraction, so leaf states during search
run through this same function.

Two entry points with identical output (gated by tests/test_oracle_engine.py):
  extract(world)                       — one-shot (dataset building)
  FeatureContext(world).leaf(...)      — incremental (runtime search,
                                         hundreds of leaves per turn)

Anything added here must be computable at runtime from a single observation
— no rating, no opponent identity, no replay-only fields.
"""

import math

from .engine import fleet_speed

# forecast probe depths (ticks ahead)
PROBES = (0, 4, 8, 16, 32)
N_NEUTRAL_PROBE = 8


def feature_names():
    names = []
    for slot in ("me", "opp", "rest"):
        for q in ("planets", "prod", "garrison", "inflight", "score"):
            for dt in PROBES:
                names.append(f"{slot}_{q}_t{dt}")
    for slot in ("me", "opp"):
        names += [f"{slot}_top1_share", f"{slot}_top3_share",
                  f"{slot}_n_sources", f"{slot}_biggest_garrison"]
    names += ["centroid_dist", "frontline_dist", "me_to_neutral_eta",
              "opp_to_neutral_eta"]
    for k in ("count", "garrison", "prod", "my_first", "price_ratio"):
        names.append(f"neutral_{k}")
    names += ["step_frac", "remaining_frac", "players_alive",
              "total_ships_log", "total_prod", "comet_count",
              "comet_ticks_left", "share_score_t0", "share_score_t32",
              "share_prod_t0", "share_prod_t32"]
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)


def _players_present(world):
    seen = set()
    for o in world.owner0:
        if o >= 0:
            seen.add(o)
    for f in world.fleets:
        if f[1] >= 0:
            seen.add(int(f[1]))
    seen.add(world.me)
    return seen


class FeatureContext:
    """Precomputed base tables for fast repeated leaf extraction."""

    def __init__(self, world):
        self.world = world
        w = world
        n = w.n_planets
        self.probes = [min(dt, w.horizon) for dt in PROBES]
        # base (owner, ships) per planet per probe
        self.base_cell = [
            [(w.post_owner[i][dt], w.post_ships[i][dt])
             for dt in self.probes]
            for i in range(n)
        ]
        self.players = _players_present(w)

        # current score per player (for opponent ranking): garrison + inflight
        cur = {p: 0.0 for p in self.players}
        for i in range(n):
            o, s = self.base_cell[i][0]
            if o in cur:
                cur[o] += s
        for (o, s, end, hit) in w.flights:
            if end > 0 and o in cur:
                cur[o] += s
        self.cur = cur
        me = w.me
        opps = sorted((p for p in self.players if p != me),
                      key=lambda p: -cur.get(p, 0))
        self.opp1 = opps[0] if opps else None
        self.rest = set(opps[1:])

        # ---- static blocks that do not depend on injected launches ----
        self._static_geo = self._geometry_block()
        self._static_neutral = self._neutral_block()
        # globals: step/remaining/comets are static; alive/totals leaf-dependent
        self._comet_static = self._comet_block()

    # ----------------------------------------------------------- blocks
    def _geometry_block(self):
        w = self.world
        me, opp1 = w.me, self.opp1
        n = w.n_planets

        def centroid(p):
            sx = sy = wt_sum = 0.0
            for i in range(n):
                if w.owner0[i] == p:
                    wt = w.prod[i] + 0.5
                    sx += w.px[i] * wt
                    sy += w.py[i] * wt
                    wt_sum += wt
            if wt_sum == 0:
                return None
            return sx / wt_sum, sy / wt_sum

        cme = centroid(me)
        copp = centroid(opp1) if opp1 is not None else None
        cdist = (math.hypot(cme[0] - copp[0], cme[1] - copp[1])
                 if cme and copp else 100.0)
        fd = 200.0
        for i in range(n):
            if w.owner0[i] != me:
                continue
            for j in range(n):
                if opp1 is None or w.owner0[j] != opp1:
                    continue
                d = math.hypot(w.px[i] - w.px[j], w.py[i] - w.py[j])
                if d < fd:
                    fd = d
        self._cme = cme
        return [cdist, fd]

    def _neutral_block(self):
        w = self.world
        me, opp1 = w.me, self.opp1
        n = w.n_planets
        neutrals = [i for i in range(n)
                    if w.owner0[i] == -1 and not w.is_comet[i]
                    and w.alive_until[i] >= 8]
        mine_idx = [i for i in range(n) if w.owner0[i] == me]
        opp_idx = [i for i in range(n) if opp1 is not None
                   and w.owner0[i] == opp1]

        def best_eta(srcs, t, size):
            if not srcs:
                return 99.0
            v = fleet_speed(max(1, size))
            best = 99.0
            tx, ty = w.px[t], w.py[t]
            for s in srcs:
                d = math.hypot(w.px[s] - tx, w.py[s] - ty) \
                    - w.pr[s] - w.pr[t]
                eta = max(1.0, math.ceil(max(d, 0.0) / v))
                if eta < best:
                    best = eta
            return best

        if neutrals and mine_idx:
            cm = self._cme or (50.0, 50.0)
            neutrals.sort(key=lambda t: math.hypot(w.px[t] - cm[0],
                                                   w.py[t] - cm[1]))
            probe_n = neutrals[:N_NEUTRAL_PROBE]
            my_first = 0
            etas_me, etas_opp, prices = [], [], []
            for t in probe_n:
                size = w.ships0[t] + 1
                em = best_eta(mine_idx, t, size)
                eo = best_eta(opp_idx, t, size)
                etas_me.append(em)
                etas_opp.append(eo)
                prices.append((w.ships0[t] + 1.0) / (w.prod[t] + 0.25))
                if em < eo:
                    my_first += 1
            return [sum(etas_me) / len(etas_me),
                    sum(etas_opp) / len(etas_opp),
                    float(len(neutrals)),
                    float(sum(w.ships0[t] for t in neutrals)),
                    float(sum(w.prod[t] for t in neutrals)),
                    float(my_first),
                    sum(prices) / len(prices)]
        return [99.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _comet_block(self):
        # fixed window so the feature is identical between the dataset
        # extraction horizon and the (longer) runtime planning horizon
        w = self.world
        n = w.n_planets
        win = min(36, w.horizon)
        comets = [i for i in range(n) if w.is_comet[i]
                  and w.alive_until[i] <= win]
        return [float(sum(1 for i in range(n) if w.is_comet[i])),
                (sum(w.alive_until[i] for i in comets) / len(comets))
                if comets else float(win)]

    # ------------------------------------------------------------- leaf
    def leaf(self, overrides=None, extra_flights=None):
        """Feature vector with optional injected launches.

        overrides: {planet_idx: (ships_delta_t0, [(dt, owner, n), ...])}
        extra_flights: [(owner, ships, end_dt, hit_idx_or_None), ...]
        """
        w = self.world
        n = w.n_planets
        me = w.me
        opp1, rest = self.opp1, self.rest
        probes = self.probes
        nP = len(probes)

        cell = self.base_cell
        over_cell = {}
        if overrides:
            walked = w.walk_with(overrides)
            for i, (po, ps, qo, qs) in walked.items():
                over_cell[i] = [(qo[dt], qs[dt]) for dt in probes]

        def cell_at(i, k):
            oc = over_cell.get(i)
            return oc[k] if oc is not None else cell[i][k]

        flights = w.flights if not extra_flights \
            else (w.flights + list(extra_flights))

        # trajectory sums per slot
        def slot_key(o):
            if o == me:
                return 0
            if o == opp1:
                return 1
            if o in rest:
                return 2
            return -1

        planets_s = [[0.0] * nP for _ in range(3)]
        prod_s = [[0.0] * nP for _ in range(3)]
        garr_s = [[0.0] * nP for _ in range(3)]
        for i in range(n):
            pr = w.prod[i]
            ci = over_cell.get(i) or cell[i]
            for k in range(nP):
                o, s = ci[k]
                sk = slot_key(o)
                if sk >= 0:
                    planets_s[sk][k] += 1.0
                    prod_s[sk][k] += pr
                    garr_s[sk][k] += s

        infl_s = [[0.0] * nP for _ in range(3)]
        for (o, s, end, hit) in flights:
            sk = slot_key(o)
            if sk >= 0:
                for k in range(nP):
                    if end > probes[k]:
                        infl_s[sk][k] += s

        feats = []
        for sk in range(3):
            feats += planets_s[sk]
            feats += prod_s[sk]
            feats += garr_s[sk]
            feats += infl_s[sk]
            feats += [garr_s[sk][k] + infl_s[sk][k] for k in range(nP)]

        # posture (current frame, with overrides applied)
        for p in (me, opp1):
            if p is None:
                feats += [0.0, 0.0, 0.0, 0.0]
                continue
            garrisons = sorted(
                (cell_at(i, 0)[1] for i in range(n)
                 if cell_at(i, 0)[0] == p), reverse=True)
            tot = sum(garrisons) or 1.0
            top1 = garrisons[0] if garrisons else 0.0
            feats += [top1 / tot, sum(garrisons[:3]) / tot,
                      float(len(garrisons)), float(top1)]

        feats += self._static_geo
        feats += self._static_neutral

        # globals
        cur = {p: 0.0 for p in self.players}
        for i in range(n):
            o, s = cell_at(i, 0)
            if o in cur:
                cur[o] += s
        for (o, s, end, hit) in flights:
            if end > 0 and o in cur:
                cur[o] += s
        alive = sum(1 for p in self.players if cur.get(p, 0) > 0)
        total_ships = sum(cur.values()) or 1.0
        total_prod = sum(w.prod[i] for i in range(n)
                         if w.owner0[i] != -2) or 1.0
        feats += [w.step / 500.0, w.remaining / 500.0, float(alive),
                  math.log1p(total_ships), float(total_prod)]
        feats += self._comet_static

        # share features at t0 and the last probe
        last = nP - 1
        for k in (0, last):
            tot_g = sum(garr_s[sk][k] + infl_s[sk][k] for sk in range(3))
            mine_g = garr_s[0][k] + infl_s[0][k]
            feats.append(mine_g / tot_g if tot_g > 0 else 0.0)
        for k in (0, last):
            tot_p = sum(prod_s[sk][k] for sk in range(3))
            feats.append(prod_s[0][k] / tot_p if tot_p > 0 else 0.0)
        # reorder: share_score_t0, share_score_t32, share_prod_t0, share_prod_t32
        return feats


def extract(world, plan_overrides=None, extra_flights=None):
    """One-shot extraction (dataset building path)."""
    return FeatureContext(world).leaf(plan_overrides, extra_flights)
