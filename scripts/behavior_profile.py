"""behavior_profile.py — behavioral fingerprint of one team across replays.

For every replay in a directory, find the focal team's seat(s) and extract
per-game behavioral metrics; print medians split by player count. Used to
diff top-ladder agents against ours.

Planet row: [id, owner, x, y, radius, ships, production]
Fleet  row: [id, owner, x, y, angle, from_planet_id, ships]

Usage:
    python scripts/behavior_profile.py audit/top-replays/team-X-sub-Y --team auto
    python scripts/behavior_profile.py audit/live-episodes/53384340 --team ChrisLeiteScha
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def detect_team(files):
    counts = collections.Counter()
    for f in files:
        try:
            teams = json.load(open(f))["info"]["TeamNames"]
        except Exception:
            continue
        for t in set(teams):
            counts[t] += 1
    return counts.most_common(1)[0][0] if counts else None


def profile_episode(d, team):
    teams = (d.get("info", {}).get("TeamNames")) or []
    seats = [i for i, t in enumerate(teams) if t == team]
    if len(seats) != 1:
        return None
    me = seats[0]
    rewards = d.get("rewards") or []
    if not rewards or any(r is None for r in rewards):
        return None
    size = len(teams)
    mx = max(rewards)
    won = rewards[me] == mx and rewards.count(mx) == 1

    steps = d["steps"]
    T = len(steps)
    boards = []          # pid -> (owner, ships, prod)
    new_fleets = []      # per step list of (owner, from_pid, ships)
    seen = set()
    for s in steps:
        obs = s[0].get("observation", {})
        boards.append({int(p[0]): (int(p[1]), float(p[5]), float(p[6]))
                       for p in obs.get("planets") or []})
        nf = []
        for f in obs.get("fleets") or []:
            fid = int(f[0])
            if fid not in seen:
                seen.add(fid)
                nf.append((int(f[1]), int(f[5]), float(f[6])))
        new_fleets.append(nf)

    def my_stats(t):
        cnt = sh = pr = 0.0
        for o, s2, p2 in boards[t].values():
            if o == me:
                cnt += 1; sh += s2; pr += p2
        return cnt, sh, pr

    # cumulative captures by me, split neutral/enemy
    neut = enemy = 0
    first_enemy_cap = None
    caps_by_60 = caps_by_100 = 0
    prev = None
    my_planet_losses = 0
    for t, b in enumerate(boards):
        if prev is not None:
            for pid, (o, _s, _p) in b.items():
                if pid in prev:
                    po = prev[pid][0]
                    if po != o:
                        if o == me:
                            if po == -1:
                                neut += 1
                            else:
                                enemy += 1
                                if first_enemy_cap is None:
                                    first_enemy_cap = t
                            if t <= 60:
                                caps_by_60 += 1
                            if t <= 100:
                                caps_by_100 += 1
                        elif po == me:
                            my_planet_losses += 1
        prev = b

    # launches
    fleet_sizes = []
    launches_per_step = []
    for t, nf in enumerate(new_fleets):
        mine = [x for x in nf if x[0] == me]
        launches_per_step.append(len(mine))
        fleet_sizes.extend(s for _o, _f, s in mine)
    total_launched = sum(fleet_sizes)
    steps_with_launch = sum(1 for n in launches_per_step if n >= 1)
    steps_with_multi = sum(1 for n in launches_per_step if n >= 2)

    # garrison ratio at checkpoints: ships on planets / (planets+fleets ships)
    def garrison_ratio(t):
        if t >= T:
            return None
        _c, sh, _p = my_stats(t)
        fl = 0.0
        for f in steps[t][0].get("observation", {}).get("fleets") or []:
            if int(f[1]) == me:
                fl += float(f[6])
        tot = sh + fl
        return sh / tot if tot > 0 else None

    c20 = my_stats(20) if T > 20 else (None, None, None)
    c40 = my_stats(40) if T > 40 else (None, None, None)
    c80 = my_stats(80) if T > 80 else (None, None, None)

    def q(p):
        if not fleet_sizes:
            return None
        xs = sorted(fleet_sizes)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]

    return {
        "size": size, "won": won, "steps": T,
        "planets20": c20[0], "ships20": c20[1], "prod20": c20[2],
        "planets40": c40[0], "ships40": c40[1], "prod40": c40[2],
        "planets80": c80[0], "ships80": c80[1], "prod80": c80[2],
        "neutral_caps": neut, "enemy_caps": enemy,
        "first_enemy_cap_step": first_enemy_cap,
        "caps_by_60": caps_by_60, "caps_by_100": caps_by_100,
        "planet_losses": my_planet_losses,
        "launch_rate": steps_with_launch / max(T, 1),
        "multi_launch_rate": steps_with_multi / max(T, 1),
        "fleets_per_game": len(fleet_sizes),
        "ships_launched": total_launched,
        "fleet_p25": q(0.25), "fleet_p50": q(0.5),
        "fleet_p75": q(0.75), "fleet_p90": q(0.9),
        "garrison40": garrison_ratio(40), "garrison80": garrison_ratio(80),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--team", default="auto")
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.corpus}/episode-*-replay.json"))
    if not files:
        print("no replays", file=sys.stderr); return 1
    team = args.team if args.team != "auto" else detect_team(files)
    rows = []
    for f in files:
        try:
            r = profile_episode(json.load(open(f)), team)
        except Exception:
            continue
        if r:
            rows.append(r)
    print(f"corpus={args.corpus}  team={team}  episodes={len(rows)}")
    for size in (2, 4):
        g = [r for r in rows if r["size"] == size]
        if not g:
            continue
        wins = sum(r["won"] for r in g)
        print(f"\n== {size}P (n={len(g)}, winrate {wins}/{len(g)} = "
              f"{100*wins/len(g):.0f}%) — medians ==")
        keys = [
            "steps", "planets20", "ships20", "prod20", "planets40", "ships40",
            "prod40", "planets80", "ships80", "prod80", "neutral_caps",
            "enemy_caps", "first_enemy_cap_step", "caps_by_60", "caps_by_100",
            "planet_losses", "launch_rate", "multi_launch_rate",
            "fleets_per_game", "ships_launched", "fleet_p25", "fleet_p50",
            "fleet_p75", "fleet_p90", "garrison40", "garrison80",
        ]
        for k in keys:
            v = med([r[k] for r in g])
            print(f"  {k:22s} {v:10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
