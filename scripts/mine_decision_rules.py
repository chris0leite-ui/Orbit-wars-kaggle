"""Mine decision rules from replay corpora: attack sizing, capture stickiness,
defense reactions, and when the winning ship lead is established.

Goes beyond behavior_profile.py's static fingerprint: reconstructs every
fleet's target (a fleet row carries no target — we track each fleet id until
it vanishes and snap its last position to the nearest planet), then classifies
events from the focal player's perspective:

  - ATTACK: focal fleet arriving at an enemy/neutral planet. Records fleet
    size vs target garrison at launch and at arrival, whether the planet
    flipped, and whether the capture STUCK (still focal 20 steps later).
  - DEFENSE: enemy fleet arriving at a focal planet. Records whether the
    garrison grew while the fleet was inbound (reinforced), collapsed to
    near zero just before arrival (evacuated), or sat (absorbed) — and
    whether the planet held.
  - DECISION STEP: last step at which the total-ship lead changes sign —
    after this the eventual outcome is visible in the material count.

Usage:
  python scripts/mine_decision_rules.py audit/top-replays/team-*/ --team "Isaiah"
  python scripts/mine_decision_rules.py <dir> --seat 0   (fixed focal seat)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

LOG1000 = math.log(1000.0)


def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(min(ships, 1000.0)) / LOG1000) ** 1.5


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[i]


def mine_replay(path: str, focal_seat: int):
    d = json.load(open(path))
    steps = d["steps"]
    n_steps = len(steps)
    rewards = d.get("rewards") or [None, None]

    # Per-step planet tables {pid: (owner, x, y, r, ships, prod)} and fleet
    # tables {fid: (owner, x, y, from_id, ships)} from the seat-0 observation
    # (board state is identical across seats).
    planets_by_step, fleets_by_step = [], []
    for s in steps:
        obs = s[0]["observation"]
        planets_by_step.append({int(p[0]): p for p in obs.get("planets", [])})
        fleets_by_step.append({int(f[0]): f for f in obs.get("fleets", [])})

    # Ship totals per player per step (planets + fleets).
    totals = []
    for k in range(n_steps):
        t = [0.0, 0.0]
        for p in planets_by_step[k].values():
            o = int(p[1])
            if 0 <= o <= 1:
                t[o] += float(p[5])
        for f in fleets_by_step[k].values():
            o = int(f[1])
            if 0 <= o <= 1:
                t[o] += float(f[6])
        totals.append(t)

    # Decision step: last sign change of the lead.
    decision = 0
    prev_sign = 0
    for k in range(n_steps):
        lead = totals[k][0] - totals[k][1]
        sign = (lead > 0) - (lead < 0)
        if sign != 0 and sign != prev_sign and prev_sign != 0:
            decision = k
        if sign != 0:
            prev_sign = sign

    # Fleet lifecycle: birth step (first seen) and death step (first absent).
    birth, death, last_seen_row = {}, {}, {}
    for k in range(n_steps):
        for fid, row in fleets_by_step[k].items():
            if fid not in birth:
                birth[fid] = k
            last_seen_row[fid] = (k, row)
    for fid, (k_last, _row) in last_seen_row.items():
        if k_last + 1 < n_steps:
            death[fid] = k_last + 1

    def nearest_planet(x, y, k):
        best, bd = None, float("inf")
        for pid, p in planets_by_step[k].items():
            dx, dy = float(p[2]) - x, float(p[3]) - y
            dist = math.hypot(dx, dy)
            if dist < bd:
                bd, best = dist, pid
        return best, bd

    attacks, defenses = [], []
    for fid, k_death in death.items():
        k_last, row = last_seen_row[fid]
        owner = int(row[1])
        ships = float(row[6])
        x, y = float(row[2]), float(row[3])
        tgt, dist = nearest_planet(x, y, min(k_death, n_steps - 1))
        if tgt is None or dist > float(planets_by_step[k_death][tgt][4]) + fleet_speed(ships) + 2.0:
            continue  # not an arrival we can attribute (comet / sun loss)
        k_birth = birth[fid]
        p_launch = planets_by_step[k_birth].get(tgt)
        p_prearr = planets_by_step[k_last].get(tgt)
        p_arr = planets_by_step[k_death].get(tgt)
        p_stick = planets_by_step[min(k_death + 20, n_steps - 1)].get(tgt)
        if p_launch is None or p_arr is None or p_prearr is None:
            continue
        tgt_owner_launch = int(p_launch[1])
        if owner == focal_seat and tgt_owner_launch != focal_seat:
            attacks.append({
                "step": k_birth,
                "ships": ships,
                "tgt_class": "neutral" if tgt_owner_launch < 0 else "enemy",
                "garrison_launch": float(p_launch[5]),
                "garrison_arrival": float(p_prearr[5]),
                "flipped": int(p_arr[1]) == focal_seat,
                "stuck": p_stick is not None and int(p_stick[1]) == focal_seat,
                "eta": k_death - k_birth,
            })
        elif owner != focal_seat and owner >= 0 and tgt_owner_launch == focal_seat:
            g0 = float(p_launch[5])
            g1 = float(p_prearr[5])
            react = "absorbed"
            if g1 >= g0 + max(4.0, 0.25 * g0):
                react = "reinforced"
            elif g1 <= max(2.0, 0.2 * g0):
                react = "evacuated"
            defenses.append({
                "step": k_death,
                "attacker_ships": ships,
                "garrison_at_launch": g0,
                "garrison_at_arrival": g1,
                "reaction": react,
                "held": p_arr is not None and int(p_arr[1]) == focal_seat,
            })

    return {
        "path": os.path.basename(path),
        "n_steps": n_steps,
        "focal_reward": rewards[focal_seat] if focal_seat < len(rewards) else None,
        "decision_step": decision,
        "attacks": attacks,
        "defenses": defenses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--team", default=None, help="substring of the focal TeamName")
    ap.add_argument("--seat", type=int, default=None, help="fixed focal seat (overrides --team)")
    ap.add_argument("--two-player-only", action="store_true", default=True)
    args = ap.parse_args()

    games = []
    for dd in args.dirs:
        for path in sorted(glob.glob(os.path.join(dd, "episode-*-replay.json"))):
            head = json.load(open(path))
            names = (head.get("info") or {}).get("TeamNames") or []
            if len(names) != 2:
                continue  # 2P analysis only
            if args.seat is not None:
                seat = args.seat
            elif args.team:
                seats = [i for i, n in enumerate(names) if args.team.lower() in (n or "").lower()]
                if len(seats) != 1:
                    continue
                seat = seats[0]
            else:
                seat = 0
            games.append(mine_replay(path, seat))

    if not games:
        print("no 2P games matched")
        return

    wins = [g for g in games if (g["focal_reward"] or 0) > 0]
    atk = [a for g in games for a in g["attacks"]]
    dfn = [d for g in games for d in g["defenses"]]

    print(f"games={len(games)}  focal-wins={len(wins)}  steps p50={pct([g['n_steps'] for g in games], .5)}")
    dec = [g["decision_step"] for g in games]
    decw = [g["decision_step"] for g in wins]
    print(f"decision step (last lead flip): p50={pct(dec, .5)} p75={pct(dec, .75)} p90={pct(dec, .9)}"
          f"  | wins only: p50={pct(decw, .5)} p90={pct(decw, .9)}")

    for cls in ("neutral", "enemy"):
        sub = [a for a in atk if a["tgt_class"] == cls]
        if not sub:
            continue
        ratios = [a["ships"] / max(1.0, a["garrison_launch"]) for a in sub]
        flips = [a for a in sub if a["flipped"]]
        stick = [a for a in flips if a["stuck"]]
        print(f"\nATTACKS on {cls}: n={len(sub)}  size p50={pct([a['ships'] for a in sub], .5):.0f}"
              f"  size/garrison(launch) p25={pct(ratios, .25):.2f} p50={pct(ratios, .5):.2f} p75={pct(ratios, .75):.2f}")
        print(f"  flip rate={len(flips)/len(sub):.2f}  stick|flip={len(stick)/max(1,len(flips)):.2f}"
              f"  eta p50={pct([a['eta'] for a in sub], .5)}")
        early = [a for a in sub if a["step"] <= 60]
        if early:
            er = [a["ships"] / max(1.0, a["garrison_launch"]) for a in early]
            print(f"  early(step<=60): n={len(early)} size/garrison p50={pct(er, .5):.2f}")

    if dfn:
        print(f"\nDEFENSE events (enemy fleet -> focal planet): n={len(dfn)}")
        by = defaultdict(list)
        for d in dfn:
            by[d["reaction"]].append(d)
        for r, sub in sorted(by.items()):
            held = sum(1 for d in sub if d["held"])
            print(f"  {r:11s} n={len(sub):4d} ({len(sub)/len(dfn):.0%})  held={held/len(sub):.2f}"
                  f"  attacker p50={pct([d['attacker_ships'] for d in sub], .5):.0f}")


if __name__ == "__main__":
    main()
