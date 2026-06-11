"""Oracle agent — candidate generation + adversarial plan search.

Per turn:
  1. Exact ledger (engine.World) — every planet's (owner, ships) timeline.
  2. Generate my candidate waves: concentrated full-drain strikes, exact
     minimum-capture strikes, multi-source coalitions, just-in-time defense
     of planets the ledger says fall, comet evacuation, frontline regroup.
  3. Model each live opponent's replies: their best base waves plus the two
     reactions our wave provokes (reinforce its target / counter-snipe the
     thinned source), all generated from THEIR seat with the same machinery.
  4. Score = trained value net on the exact post-injection forecast.
     Each candidate is priced under the WORST reply (maximin); a candidate
     joins the plan only if its worst case beats the null plan's worst case.
  5. Greedy portfolio: commit best candidate, deduct spares, re-price the
     rest on top of the committed plan, repeat under the time budget.

All ship counts integer; every launch verified by exact flight simulation
(never feeds the sun, never overshoots off-board).
"""

import math
import os
import time

from .engine import (World, fleet_speed, safe_launch, required_ships)
from .features import FeatureContext
from .value import ValueNet

# knobs (env-overridable for A/B)
def _f(name, dflt):
    try:
        return float(os.environ.get(name, dflt))
    except Exception:
        return dflt

HORIZON = int(_f("ORACLE_HORIZON", 48))
TARGETS_PER_SOURCE = int(_f("ORACLE_TPS", 8))
MAX_WAVES = int(_f("ORACLE_MAX_WAVES", 5))
TIME_BUDGET = _f("ORACLE_TIME_BUDGET", 0.55)
EPS_GAIN = _f("ORACLE_EPS_GAIN", 1e-4)
CAPTURE_MARGIN = int(_f("ORACLE_CAPTURE_MARGIN", 3))
REPLY_TOP = int(_f("ORACLE_REPLY_TOP", 3))
MIN_SEND = int(_f("ORACLE_MIN_SEND", 3))


class Cand:
    __slots__ = ("launches", "kind", "tgt", "quick", "score")

    def __init__(self, launches, kind, tgt):
        self.launches = launches      # [(src_idx, ships, angle, hit_dt, tgt_idx)]
        self.kind = kind
        self.tgt = tgt
        self.quick = 0.0
        self.score = None


def _spares(world, owner):
    """Launchable surplus per planet of `owner` (survives known incoming)."""
    sp = {}
    for i in range(world.n_planets):
        if world.owner0[i] != owner:
            continue
        s, doomed = safe_launch(world, i, owner)
        if world.is_comet[i] and world.alive_until[i] <= world.horizon:
            s = world.ships0[i]          # comet exits soon: all ships launchable
        sp[i] = (s, doomed)
    return sp


def _shot(world, src, tgt, ships):
    """Verified launch record or None."""
    if ships < 1:
        return None
    vs = world.verified_shot(src, tgt, ships)
    if vs is None:
        return None
    angle, dt = vs
    return (src, int(ships), angle, dt, tgt)


def _gen_candidates(world, sp, me, now):
    """Candidate waves for `me` given spare map sp."""
    n = world.n_planets
    cands = []
    sources = [i for i, (s, d) in sp.items() if s >= MIN_SEND]
    if not sources:
        return cands

    # target shortlist per source: nearest non-mine planets
    enemy_or_neutral = [t for t in range(n)
                        if world.owner0[t] != me
                        and world.post_owner[t][min(4, world.horizon)] != -2]
    # my planets that fall in the base forecast (defense targets)
    falling = []
    for t in range(n):
        if world.owner0[t] != me:
            continue
        po = world.post_owner[t]
        for dt in range(1, min(world.horizon, 24) + 1):
            if po[dt] is not None and po[dt] != me and po[dt] != -2:
                falling.append((t, dt))
                break

    seen_pairs = set()
    for s in sources:
        spare = sp[s][0]
        sx, sy = world.px[s], world.py[s]
        near = sorted(enemy_or_neutral,
                      key=lambda t: math.hypot(world.px[t] - sx,
                                               world.py[t] - sy))
        for t in near[:TARGETS_PER_SOURCE]:
            if (s, t) in seen_pairs:
                continue
            seen_pairs.add((s, t))
            # estimate arrival for the required-size lookup
            aim = world.aim_at(s, t, max(spare, 1))
            if aim is None:
                continue
            est_dt = aim[3]
            req = required_ships(world, t, min(est_dt, world.horizon), me)
            # full-drain concentrated wave
            rec = _shot(world, s, t, spare)
            if rec is not None:
                # does the full wave actually capture at its true arrival?
                req_at = required_ships(world, t, rec[3], me)
                if req_at is None or spare >= req_at[0]:
                    cands.append(Cand([rec], "wave", t))
                elif world.owner0[t] >= 0 and world.owner0[t] != me:
                    # damage wave against an enemy stack (1:1 trade) — only
                    # let the value net see it; usually rejected
                    cands.append(Cand([rec], "press", t))
            # exact-size capture (cheaper, keeps the rest home)
            if req is not None:
                need = req[0] + CAPTURE_MARGIN
                if MIN_SEND <= need < spare:
                    rec2 = _shot(world, s, t, need)
                    if rec2 is not None:
                        req_at = required_ships(world, t, rec2[3], me)
                        if req_at is not None and need >= req_at[0]:
                            cands.append(Cand([rec2], "take", t))

    # coalitions for rich targets no single source captures
    by_tgt = {}
    for c in cands:
        if c.kind == "wave":
            by_tgt.setdefault(c.tgt, []).append(c)
    coalition_tgts = []
    for t in enemy_or_neutral:
        if world.prod[t] < 2 or t in by_tgt:
            continue
        coalition_tgts.append(t)
    for t in coalition_tgts[:6]:
        tx, ty = world.px[t], world.py[t]
        near_src = sorted(sources,
                          key=lambda s2: math.hypot(world.px[s2] - tx,
                                                    world.py[s2] - ty))[:3]
        total = sum(sp[s2][0] for s2 in near_src)
        if total < MIN_SEND:
            continue
        recs = []
        worst_dt = 0
        for s2 in near_src:
            rec = _shot(world, s2, t, sp[s2][0])
            if rec is not None:
                recs.append(rec)
                worst_dt = max(worst_dt, rec[3])
        if len(recs) >= 2 and worst_dt <= world.horizon:
            req = required_ships(world, t, worst_dt, me)
            if req is not None and sum(r[1] for r in recs) >= req[0]:
                cands.append(Cand(recs, "coal", t))

    # just-in-time defense of falling planets
    for (t, flip_dt) in falling[:4]:
        tx, ty = world.px[t], world.py[t]
        helpers = sorted((s2 for s2 in sources if s2 != t),
                         key=lambda s2: math.hypot(world.px[s2] - tx,
                                                   world.py[s2] - ty))[:2]
        recs = []
        for s2 in helpers:
            rec = _shot(world, s2, t, sp[s2][0])
            if rec is not None and rec[3] <= flip_dt:
                recs.append(rec)
        if recs:
            cands.append(Cand(recs[:1], "def", t))
            if len(recs) > 1:
                cands.append(Cand(recs, "def2", t))

    # comet evacuation: doomed garrisons fly to the nearest keepable planet
    for s in sources:
        if not (world.is_comet[s] and world.alive_until[s] <= 12):
            continue
        spare = sp[s][0]
        if spare < 1:
            continue
        sx, sy = world.px[s], world.py[s]
        homes = sorted(
            (t for t in range(n) if t != s
             and world.post_owner[t][min(world.horizon, 8)] == me),
            key=lambda t: math.hypot(world.px[t] - sx, world.py[t] - sy))
        for t in homes[:2]:
            rec = _shot(world, s, t, spare)
            if rec is not None:
                cands.append(Cand([rec], "evac", t))
                break

    # regroup: rear mass walks toward the frontline
    enemies = [j for j in range(n)
               if world.owner0[j] >= 0 and world.owner0[j] != me]
    if enemies and len(sources) >= 2:
        def front_d(i):
            return min(math.hypot(world.px[i] - world.px[j],
                                  world.py[i] - world.py[j])
                       for j in enemies)
        mine_all = [i for i in range(n) if world.owner0[i] == me
                    and not world.is_comet[i]]
        if len(mine_all) >= 2:
            front = min(mine_all, key=front_d)
            rears = sorted(sources, key=front_d, reverse=True)
            for s in rears[:2]:
                if s == front or front_d(s) < front_d(front) + 5:
                    continue
                rec = _shot(world, s, front, sp[s][0])
                if rec is not None:
                    cands.append(Cand([rec], "regroup", front))

    return cands


def _opponent_replies(world, me, my_cand, opp_state):
    """Reply set one opponent could fly NEXT turn, given my candidate."""
    opp, osp, base_waves = opp_state
    replies = [[]]                                  # null reply
    for w_ in base_waves[:2]:
        replies.append([w_])
    if my_cand is not None and my_cand.launches:
        # reinforce the target of my biggest launch
        big = max(my_cand.launches, key=lambda r: r[1])
        tgt, hit_dt = big[4], big[3]
        if world.owner0[tgt] == opp:
            tx, ty = world.px[tgt], world.py[tgt]
            helpers = sorted(
                (j for j, (s, d) in osp.items() if s >= 1 and j != tgt),
                key=lambda j: math.hypot(world.px[j] - tx,
                                         world.py[j] - ty))
            for j in helpers[:2]:
                rec = _shot(world, j, tgt, osp[j][0])
                if rec is not None and rec[3] + 1 <= hit_dt:
                    replies.append([(rec[0], rec[1], rec[2],
                                     rec[3] + 1, rec[4])])
                    break
        # counter-snipe my thinned source
        src = big[0]
        left = world.ships0[src] - sum(r[1] for r in my_cand.launches
                                       if r[0] == src)
        sx, sy = world.px[src], world.py[src]
        snipers = sorted(
            (j for j, (s, d) in osp.items() if s >= 1),
            key=lambda j: math.hypot(world.px[j] - sx, world.py[j] - sy))
        for j in snipers[:1]:
            sz = osp[j][0]
            rec = _shot(world, j, src, sz)
            if rec is not None:
                need = left + world.prod[src] * (rec[3] + 1) + 1
                if sz >= need:
                    replies.append([(rec[0], rec[1], rec[2],
                                     rec[3] + 1, rec[4])])
                break
    return replies[:1 + REPLY_TOP]


def _merge_overrides(world, launch_sets, me_set_owner):
    """Build walk_with overrides + extra flight records from launch lists.

    launch_sets: [(owner, [launch records])]
    """
    deltas = {}
    extras = {}
    flights = []
    for owner, recs in launch_sets:
        for (src, ships, angle, hit_dt, tgt) in recs:
            deltas[src] = deltas.get(src, 0) - ships
            extras.setdefault(tgt, []).append((hit_dt, owner, ships))
            flights.append((owner, ships, hit_dt, tgt))
    overrides = {}
    touched = set(deltas) | set(extras)
    for i in touched:
        overrides[i] = (deltas.get(i, 0), extras.get(i, []))
    return overrides, flights


class Planner:
    def __init__(self):
        self.net = ValueNet()

    def act(self, obs):
        t0 = time.time()
        world = World(obs, horizon=HORIZON)
        world.build_ledger()
        me = world.me
        ctx = FeatureContext(world)

        sp_me = _spares(world, me)
        cands = _gen_candidates(world, sp_me, me, t0)
        if not cands:
            return []

        # opponent precomputation (live opponents, strongest first)
        opp_states = []
        for opp in sorted(self_opponents(world, me),
                          key=lambda p: -player_score(world, p)):
            osp = _spares(world, opp)
            base = _opp_base_waves(world, opp, osp)
            opp_states.append((opp, osp, base))
        opp_states = opp_states[:2]

        # quick prior: value of each candidate under no reply
        feats = []
        for c in cands:
            ov, fl = _merge_overrides(world, [(me, c.launches)], me)
            feats.append(ctx.leaf(ov, fl))
        null_feat = ctx.leaf(None, None)
        scores = self.net.batch(feats + [null_feat])
        for c, s in zip(cands, scores[:-1]):
            c.quick = float(s)
        null_quick = float(scores[-1])

        cands.sort(key=lambda c: -c.quick)
        committed = []          # my committed launch records
        committed_score = None
        spare_left = {i: s for i, (s, d) in sp_me.items()}

        def worst_case(c_launches):
            """Worst V over opponent reply sets for my (committed + c)."""
            mine = committed + c_launches
            batch = []
            for (opp, osp, base) in opp_states or [(None, {}, [])]:
                fake = Cand(mine, "joint", -1) if mine else None
                reps = _opponent_replies(world, me, fake,
                                         (opp, osp, base)) \
                    if opp is not None else [[]]
                for rep in reps:
                    ov, fl = _merge_overrides(
                        world, [(me, mine), (opp, rep)], me)
                    batch.append(ctx.leaf(ov, fl))
            if not batch:
                ov, fl = _merge_overrides(world, [(me, mine)], me)
                batch.append(ctx.leaf(ov, fl))
            vals = self.net.batch(batch)
            return float(vals.min())

        committed_score = worst_case([])
        rounds = 0
        while rounds < MAX_WAVES and (time.time() - t0) < TIME_BUDGET:
            rounds += 1
            best = None
            best_score = committed_score
            for c in cands[:12]:
                # affordability vs remaining spares
                need = {}
                ok = True
                for (src, ships, angle, dt, tgt) in c.launches:
                    need[src] = need.get(src, 0) + ships
                for src, k in need.items():
                    if spare_left.get(src, 0) < k:
                        ok = False
                        break
                if not ok:
                    continue
                if (time.time() - t0) > TIME_BUDGET:
                    break
                sc = worst_case(c.launches)
                c.score = sc
                if sc > best_score + EPS_GAIN:
                    best_score = sc
                    best = c
            if best is None:
                break
            committed += best.launches
            committed_score = best_score
            for (src, ships, angle, dt, tgt) in best.launches:
                spare_left[src] = spare_left.get(src, 0) - ships
            cands = [c for c in cands if c is not best
                     and c.tgt != best.tgt]

        moves = []
        send_total = {}
        for (src, ships, angle, dt, tgt) in committed:
            send_total[src] = send_total.get(src, 0) + ships
        for src, tot in send_total.items():
            if tot > world.ships0[src]:
                return []        # accounting bug guard: launch nothing
        for (src, ships, angle, dt, tgt) in committed:
            moves.append([world.pid[src], float(angle), int(ships)])
        return moves


def self_opponents(world, me):
    out = set()
    for o in world.owner0:
        if o >= 0 and o != me:
            out.add(o)
    for f in world.fleets:
        if f[1] >= 0 and f[1] != me:
            out.add(int(f[1]))
    return out


def player_score(world, p):
    g = sum(world.ships0[i] for i in range(world.n_planets)
            if world.owner0[i] == p)
    fl = sum(f[6] for f in world.fleets if f[1] == p)
    return g + fl


def _opp_base_waves(world, opp, osp):
    """The opponent's two most attractive immediate waves (cheap ROI rank)."""
    n = world.n_planets
    out = []
    srcs = [i for i, (s, d) in osp.items() if s >= MIN_SEND]
    scored = []
    for s in srcs:
        sx, sy = world.px[s], world.py[s]
        spare = osp[s][0]
        tgts = sorted((t for t in range(n) if world.owner0[t] != opp),
                      key=lambda t: math.hypot(world.px[t] - sx,
                                               world.py[t] - sy))[:5]
        for t in tgts:
            aim = world.aim_at(s, t, spare)
            if aim is None:
                continue
            req = required_ships(world, t, min(aim[3], world.horizon), opp)
            if req is None or req[0] > spare:
                continue
            roi = (world.prod[t] + 0.3) / (req[0] + 2.0 * aim[3])
            scored.append((roi, s, t, spare))
    scored.sort(reverse=True)
    for (roi, s, t, spare) in scored[:3]:
        rec = _shot(world, s, t, spare)
        if rec is not None:
            out.append((rec[0], rec[1], rec[2], rec[3] + 1, rec[4]))
    return out
