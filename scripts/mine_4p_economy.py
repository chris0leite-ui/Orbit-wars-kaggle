"""mine_4p_economy.py — when do we fall behind in 4P, and on what axis?

For each 4P episode: per-player total ships (planets+fleets), planet count,
production sum, and cumulative captures (neutral / enemy / from-us) sampled
at checkpoints. Reports our rank trajectory in losses vs wins, and the
eventual winner's trajectory for contrast.

Usage: python scripts/mine_4p_economy.py [corpus_dir]
"""
from __future__ import annotations

import glob
import json
import statistics
import sys

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "audit/live-episodes/53384340"
TEAM = "ChrisLeiteScha"
CHECKPOINTS = [20, 40, 60, 80, 100, 120]


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def analyze(d):
    info = d.get("info", {})
    teams = info.get("TeamNames") or []
    seats = [i for i, t in enumerate(teams) if t == TEAM]
    if len(seats) != 1 or len(d["steps"][0]) != 4:
        return None
    me = seats[0]
    rewards = d.get("rewards") or []
    if not rewards or any(r is None for r in rewards):
        return None
    mx = max(rewards)
    won = rewards[me] == mx and rewards.count(mx) == 1
    winner = rewards.index(mx) if rewards.count(mx) == 1 else None

    steps = d["steps"]
    T = len(steps)

    # cumulative captures per player, split by what was taken
    cum_neutral = [0] * 4
    cum_enemy = [0] * 4
    cum_from_us = [0] * 4
    prev = None
    snap = {}
    for t, s in enumerate(steps):
        obs = s[0].get("observation", {})
        planets = {int(p[0]): (int(p[1]), float(p[5]), float(p[6]))
                   for p in obs.get("planets") or []}
        fleets = obs.get("fleets") or []
        if prev is not None:
            for pid, (o, _s2, _pr) in planets.items():
                if pid in prev and prev[pid][0] != o and 0 <= o < 4:
                    po = prev[pid][0]
                    if po == -1:
                        cum_neutral[o] += 1
                    elif po == me:
                        cum_from_us[o] += 1
                    else:
                        cum_enemy[o] += 1
        prev = planets
        if t in CHECKPOINTS:
            ships = [0.0] * 4
            count = [0] * 4
            prod = [0.0] * 4
            for pid, (o, sh, pr) in planets.items():
                if 0 <= o < 4:
                    ships[o] += sh
                    count[o] += 1
                    prod[o] += pr
            for f in fleets:
                o = int(f[1])
                if 0 <= o < 4:
                    ships[o] += float(f[6])
            snap[t] = {
                "ships": ships[:], "count": count[:], "prod": prod[:],
                "neutral": cum_neutral[:], "enemy": cum_enemy[:],
                "from_us": cum_from_us[:],
            }
    return {"won": won, "me": me, "winner": winner, "snap": snap, "T": T}


def main():
    files = sorted(glob.glob(f"{CORPUS}/episode-*-replay.json"))
    eps = []
    for f in files:
        try:
            r = analyze(json.load(open(f)))
        except Exception:
            continue
        if r:
            eps.append(r)
    losses = [e for e in eps if not e["won"]]
    wins = [e for e in eps if e["won"]]

    def rank_of(vals, i):
        return 1 + sum(1 for j, v in enumerate(vals) if j != i and v > vals[i])

    print(f"4P episodes: {len(eps)}  wins={len(wins)}  losses={len(losses)}")
    for tag, group in (("LOSSES", losses), ("WINS", wins)):
        print(f"\n== {tag} (n={len(group)}) ==")
        hdr = (f"{'step':>5} {'shipRank':>9} {'myShips':>8} {'winShips':>9} "
               f"{'myProd':>7} {'winProd':>8} {'myNeut':>7} {'winNeut':>8} "
               f"{'myEnemy':>8} {'winEnemy':>9}")
        print(hdr)
        for cp in CHECKPOINTS:
            rows = [e for e in group if cp in e["snap"]]
            if not rows:
                continue
            ranks, mys, ws, myp, wp, myn, wn, mye, we = ([] for _ in range(9))
            for e in rows:
                s = e["snap"][cp]
                me = e["me"]
                w = e["winner"]
                ranks.append(rank_of(s["ships"], me))
                mys.append(s["ships"][me])
                myp.append(s["prod"][me])
                myn.append(s["neutral"][me])
                mye.append(s["enemy"][me] + s["from_us"][me])
                if w is not None and w != me:
                    ws.append(s["ships"][w])
                    wp.append(s["prod"][w])
                    wn.append(s["neutral"][w])
                    we.append(s["enemy"][w] + s["from_us"][w])
            print(f"{cp:>5} {median(ranks):>9.1f} {median(mys):>8.0f} "
                  f"{median(ws):>9.0f} {median(myp):>7.1f} {median(wp):>8.1f} "
                  f"{median(myn):>7.1f} {median(wn):>8.1f} {median(mye):>8.1f} "
                  f"{median(we):>9.1f}")


if __name__ == "__main__":
    main()
