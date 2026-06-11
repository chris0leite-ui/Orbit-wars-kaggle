"""Oracle agent — policy-driven planner with exact-engine sizing and a
value-net portfolio veto.

Per turn:
  1. Exact ledger (engine.World, shared PLAN_HORIZON) — every planet's
     (owner, ships) timeline given all in-flight fleets.
  2. Enumerate the (source -> target) decision surface (shortlist_pairs):
     attacks on nearby + high-value enemy/neutral planets, own->own
     logistics transfers, just-in-time defense of planets the ledger says
     fall, comet evacuation. The same enumeration labels the training data.
  3. The policy net (behavior-cloned from 1500+ rated ladder replays) scores
     every pair: P(a top player fires this now) and a size fraction.
  4. Fire pairs above threshold, sizes snapped to exact-engine requirements
     (capture floor at true arrival tick, full-drain when near-full), all
     launches verified by exact flight simulation.
  5. Value-net veto: if the chosen portfolio's worst-case-vs-reply value
     falls below the null plan's, drop the least confident wave and retest.

The policy decides WHAT a strong player does; the exact engine guarantees
the HOW (sizes that capture, shots that land); the value net only blocks
portfolio-level blunders.
"""

import math
import os
import time

from .engine import (World, PLAN_HORIZON, fleet_speed, safe_launch,
                     required_ships)
from .features import FeatureContext
from .value import ValueNet
from .policy_features import PolicyContext
from .policy import PolicyNet


def _f(name, dflt):
    try:
        return float(os.environ.get(name, dflt))
    except Exception:
        return dflt

TARGETS_PER_SOURCE = int(_f("ORACLE_TPS", 10))
VALUE_TARGETS = int(_f("ORACLE_VALUE_TGTS", 3))
TRANSFERS_PER_SOURCE = int(_f("ORACLE_TRANSFERS", 5))
TRANSFER_MAX_D = _f("ORACLE_TRANSFER_MAX_D", 50.0)
MAX_WAVES = int(_f("ORACLE_MAX_WAVES", 6))
TIME_BUDGET = _f("ORACLE_TIME_BUDGET", 0.55)
FIRE_THETA = _f("ORACLE_FIRE_THETA", 0.50)
CAPTURE_MARGIN = int(_f("ORACLE_CAPTURE_MARGIN", 2))
VETO_DELTA = _f("ORACLE_VETO_DELTA", 0.02)
VETO_ON = os.environ.get("ORACLE_VETO", "1") != "0"
MIN_GARRISON_SRC = 2


def source_states(world, owner):
    """{planet_idx: (garrison, safe_spare, doomed)} for `owner`."""
    out = {}
    for i in range(world.n_planets):
        if world.owner0[i] != owner:
            continue
        g = world.ships0[i]
        if g < 1:
            continue
        safe, doomed = safe_launch(world, i, owner)
        if world.is_comet[i] and world.alive_until[i] <= world.horizon:
            safe = g
        out[i] = (g, safe, doomed)
    return out


def shortlist_pairs(world, src_states):
    """The (src, tgt) decision surface for world.me: [(kind, src, tgt)].

    Mirrored by scripts/oracle_policy_dataset.py so the policy net's train
    and serve distributions match. Kinds: attack, transfer, defend, evac.
    """
    me = world.me
    n = world.n_planets
    pairs = []
    sources = [i for i, (g, s, d) in src_states.items()
               if g >= MIN_GARRISON_SRC]
    if not sources:
        return pairs

    horizon4 = min(4, world.horizon)
    others = [t for t in range(n)
              if world.owner0[t] != me
              and world.post_owner[t][horizon4] != -2]
    own = [t for t in range(n) if world.owner0[t] == me]

    # global value targets: best production per current price
    def value_key(t):
        return -(world.prod[t] + 0.3) / (world.ships0[t] + 5.0)
    val_tgts = sorted((t for t in others if world.prod[t] >= 2),
                      key=value_key)[:VALUE_TARGETS]

    falling = []
    for t in own:
        po = world.post_owner[t]
        for dt in range(1, min(world.horizon, 24) + 1):
            if po[dt] is not None and po[dt] != me and po[dt] != -2:
                falling.append(t)
                break

    seen = set()

    def add(kind, s, t):
        if s != t and (s, t) not in seen:
            seen.add((s, t))
            pairs.append((kind, s, t))

    for s in sources:
        sx, sy = world.px[s], world.py[s]

        def d_to(t):
            return math.hypot(world.px[t] - sx, world.py[t] - sy)

        for t in sorted(others, key=d_to)[:TARGETS_PER_SOURCE]:
            add("attack", s, t)
        for t in val_tgts:
            add("attack", s, t)
        near_own = [t for t in sorted(own, key=d_to)
                    if t != s and d_to(t) <= TRANSFER_MAX_D]
        for t in near_own[:TRANSFERS_PER_SOURCE]:
            add("transfer", s, t)

    for t in falling[:6]:
        tx, ty = world.px[t], world.py[t]
        helpers = sorted((s for s in sources if s != t),
                         key=lambda s: math.hypot(world.px[s] - tx,
                                                  world.py[s] - ty))[:3]
        for s in helpers:
            add("defend", s, t)

    for s in sources:
        if world.is_comet[s] and world.alive_until[s] <= 12:
            sx, sy = world.px[s], world.py[s]
            homes = sorted(
                (t for t in range(n) if t != s
                 and world.post_owner[t][min(world.horizon, 8)] == me),
                key=lambda t: math.hypot(world.px[t] - sx,
                                         world.py[t] - sy))[:2]
            for t in homes:
                add("evac", s, t)
    return pairs


def _shot(world, src, tgt, ships):
    if ships < 1:
        return None
    vs = world.verified_shot(src, tgt, ships)
    if vs is None:
        return None
    angle, dt = vs
    return (src, int(ships), angle, dt, tgt)


def _merge_overrides(launch_sets):
    """[(owner, [launch records])] -> (walk_with overrides, flight records)."""
    deltas, extras, flights = {}, {}, []
    for owner, recs in launch_sets:
        for (src, ships, angle, hit_dt, tgt) in recs:
            deltas[src] = deltas.get(src, 0) - ships
            extras.setdefault(tgt, []).append((hit_dt, owner, ships))
            flights.append((owner, ships, hit_dt, tgt))
    overrides = {i: (deltas.get(i, 0), extras.get(i, []))
                 for i in set(deltas) | set(extras)}
    return overrides, flights


def opponents_of(world, me):
    out = set()
    for o in world.owner0:
        if o >= 0 and o != me:
            out.add(o)
    for f in world.fleets:
        if f[1] >= 0 and f[1] != me:
            out.add(int(f[1]))
    return out


class Planner:
    def __init__(self):
        self.policy = PolicyNet()
        self.value = ValueNet()

    # ----------------------------------------------------------- replies
    def _reply_sets(self, world, me, my_recs):
        """A few plausible opponent reactions for the value veto."""
        reps = [[]]
        opps = sorted(opponents_of(world, me),
                      key=lambda p: -sum(
                          world.ships0[i] for i in range(world.n_planets)
                          if world.owner0[i] == p))
        if not opps or not my_recs:
            return reps, None
        opp = opps[0]
        ost = source_states(world, opp)
        big = max(my_recs, key=lambda r: r[1])
        tgt, hit_dt, src = big[4], big[3], big[0]
        # reinforce the attacked planet before our arrival
        if world.owner0[tgt] == opp:
            tx, ty = world.px[tgt], world.py[tgt]
            for j in sorted((j for j, (g, s, d) in ost.items() if s >= 1),
                            key=lambda j: math.hypot(world.px[j] - tx,
                                                     world.py[j] - ty))[:2]:
                rec = _shot(world, j, tgt, ost[j][1])
                if rec is not None and rec[3] + 1 <= hit_dt:
                    reps.append([(rec[0], rec[1], rec[2],
                                  rec[3] + 1, rec[4])])
                    break
        # counter-snipe our thinned source
        left = world.ships0[src] - sum(r[1] for r in my_recs
                                       if r[0] == src)
        sx, sy = world.px[src], world.py[src]
        for j in sorted((j for j, (g, s, d) in ost.items() if g >= 4),
                        key=lambda j: math.hypot(world.px[j] - sx,
                                                 world.py[j] - sy))[:1]:
            sz = ost[j][0]
            rec = _shot(world, j, src, sz)
            if rec is not None:
                need = left + world.prod[src] * (rec[3] + 1) + 1
                if sz >= need:
                    reps.append([(rec[0], rec[1], rec[2],
                                  rec[3] + 1, rec[4])])
        return reps, opp

    # --------------------------------------------------------------- act
    def act(self, obs):
        t0 = time.time()
        world = World(obs, horizon=PLAN_HORIZON)
        world.build_ledger()
        me = world.me

        src_states = source_states(world, me)
        pairs = shortlist_pairs(world, src_states)
        if not pairs:
            return []

        pctx = PolicyContext(world, src_states)
        feats = [pctx.pair(s, t, *src_states[s]) for (_k, s, t) in pairs]
        p_fire, frac = self.policy.batch(feats)

        order = sorted(range(len(pairs)), key=lambda k: -p_fire[k])
        remaining = {i: g for i, (g, s, d) in src_states.items()}
        chosen = []          # (p, launch record)
        for k in order:
            if len(chosen) >= MAX_WAVES or p_fire[k] < FIRE_THETA:
                break
            if (time.time() - t0) > TIME_BUDGET:
                break
            kind, s, t = pairs[k]
            g_now = remaining.get(s, 0)
            if g_now < 1:
                continue
            size = int(round(float(frac[k]) * src_states[s][0]))
            size = max(1, min(size, g_now))
            # exact-engine snapping for capture attempts
            if kind == "attack":
                aim = world.aim_at(s, t, size)
                if aim is None:
                    continue
                eta = min(aim[3], world.horizon)
                req = required_ships(world, t, eta, me)
                if req is not None:
                    need = req[0] + CAPTURE_MARGIN
                    if size < need <= g_now:
                        size = need
                    elif size < req[0]:
                        # cannot capture: only fire if it reinforces a race
                        slot = world.arrivals[t].get(eta, {})
                        if slot.get(me, 0) <= 0:
                            continue
            if size >= 0.9 * g_now:
                size = g_now
            rec = _shot(world, s, t, size)
            if rec is None:
                continue
            # re-verify capture at the TRUE arrival tick
            if kind == "attack":
                req_at = required_ships(world, t, rec[3], me)
                if req_at is not None and rec[1] < req_at[0]:
                    slot = world.arrivals[t].get(rec[3], {})
                    if slot.get(me, 0) <= 0:
                        continue
            chosen.append((float(p_fire[k]), rec))
            remaining[s] = g_now - rec[1]

        if not chosen:
            return []

        # ---- value-net portfolio veto -------------------------------
        if VETO_ON and self.value is not None:
            ctx = FeatureContext(world)
            for _ in range(3):
                if not chosen:
                    break
                recs = [r for (_p, r) in chosen]
                reps, opp = self._reply_sets(world, me, recs)
                batch = []
                for rep in reps:
                    ov, fl = _merge_overrides([(me, recs), (opp, rep)])
                    batch.append(ctx.leaf(ov, fl))
                for rep in reps:
                    ov, fl = _merge_overrides([(opp, rep)])
                    batch.append(ctx.leaf(ov, fl))
                vals = self.value.batch(batch)
                nrep = len(reps)
                v_plan = float(vals[:nrep].min())
                v_null = float(vals[nrep:].min())
                if v_plan >= v_null - VETO_DELTA:
                    break
                chosen.sort(key=lambda pr: pr[0])
                chosen.pop(0)        # drop least confident wave, retest
                if (time.time() - t0) > TIME_BUDGET:
                    break

        moves = []
        sent = {}
        for (_p, (src, ships, angle, dt, tgt)) in chosen:
            sent[src] = sent.get(src, 0) + ships
        for src, tot in sent.items():
            if tot > world.ships0[src]:
                return []        # accounting bug guard: launch nothing
        for (_p, (src, ships, angle, dt, tgt)) in chosen:
            moves.append([world.pid[src], float(angle), int(ships)])
        return moves
