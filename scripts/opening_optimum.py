"""Opening optimality gap: how much production does an agent leave on the table?

The pre-contact opening is a deterministic single-player scheduling problem
(PI thesis, kb/thoughts/2026-06-10-pi-opening-game-solvable.md): neutral
garrisons are STATIC (verified empirically — capture cost is fixed), planet
motion is deterministic (rigid rotation about the board center), and
production compounds. This tool computes a near-optimal capture schedule by
beam search and scores an agent's ACTUAL opening from a replay against it.

Optimum objective: total ships produced by --horizon (default 40), the
compounding currency of the opening. Also reports planets and production
rate at the horizon.

Usage:
  # score a replay corpus (focal team by name):
  python scripts/opening_optimum.py audit/live-episodes/53527125 --team ChrisLeiteScha
  python scripts/opening_optimum.py audit/top-replays/team-15654628* --team Isaiah
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import math
import os
import sys

LOG1000 = math.log(1000.0)
CENTER = 50.0


def fleet_speed(s: float) -> float:
    if s <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(min(s, 1000.0)) / LOG1000) ** 1.5


class Board:
    """Deterministic planet kinematics from a replay's step-0 observation."""

    def __init__(self, obs0: dict):
        self.angvel = float(obs0.get("angular_velocity", 0.0))
        self.planets = {}
        for p in obs0["planets"]:
            pid, owner, x, y, r, ships, prod = p[:7]
            ox, oy = x - CENTER, y - CENTER
            orb_r = math.hypot(ox, oy)
            self.planets[int(pid)] = dict(
                owner=int(owner), r=float(r), ships=float(ships),
                prod=float(prod), orb_r=orb_r, a0=math.atan2(oy, ox),
                orbiting=(orb_r + float(r)) < 50.0,
            )

    def pos(self, pid: int, t: float):
        p = self.planets[pid]
        a = p["a0"] + (self.angvel * t if p["orbiting"] else 0.0)
        return CENTER + p["orb_r"] * math.cos(a), CENTER + p["orb_r"] * math.sin(a)

    def eta(self, src: int, tgt: int, size: float, t: float) -> int:
        """Travel turns from src (launch at t) to moving tgt, fixed-point."""
        sp = fleet_speed(size)
        sx, sy = self.pos(src, t)
        e = 1.0
        for _ in range(4):
            tx, ty = self.pos(tgt, t + e)
            d = math.hypot(tx - sx, ty - sy) - self.planets[tgt]["r"]
            e = max(1.0, d / sp)
        return max(1, math.ceil(e))


def optimal_schedule(board: Board, my_start: list[int], horizon: int,
                     beam_width: int = 192, safe_only_vs: list[int] | None = None):
    """Beam search over capture schedules. Returns (produced, planets_at_H,
    prod_at_H, schedule). State garrison model: per-planet ships grow by
    prod each step; a launch debits garrison+1 of the target from one source."""
    neutrals = [pid for pid, p in board.planets.items() if p["owner"] < 0]
    if safe_only_vs:
        def closer_to_us(n):
            t0 = min(board.eta(s, n, board.planets[n]["ships"] + 1, 0) for s in my_start)
            t1 = min(board.eta(s, n, board.planets[n]["ships"] + 1, 0) for s in safe_only_vs)
            return t0 <= t1
        neutrals = [n for n in neutrals if closer_to_us(n)]

    # State: (t, tuple of (pid, ships) owned, frozenset captured, produced,
    #         in-flight tuple of (arrive_t, tgt, size))
    start = (0, tuple((pid, board.planets[pid]["ships"]) for pid in my_start),
             frozenset(), 0.0, ())
    beam = [start]
    best = (0.0, len(my_start), sum(board.planets[p]["prod"] for p in my_start), ())

    def advance(state, until):
        """Produce + land fleets up to time `until`. Returns new state."""
        t, owned, captured, produced, flights = state
        owned = dict(owned)
        flights = sorted(flights)
        sched = []
        while t < until:
            step_to = until
            if flights and flights[0][0] < step_to:
                step_to = flights[0][0]
            dt = step_to - t
            for pid in owned:
                owned[pid] += board.planets[pid]["prod"] * dt
            produced += sum(board.planets[pid]["prod"] for pid in owned) * dt
            t = step_to
            while flights and flights[0][0] <= t:
                _at, tgt, size = heapq.heappop(flights)
                g = board.planets[tgt]["ships"]
                owned[tgt] = max(1.0, size - g)
                captured = captured | {tgt}
        return (t, tuple(sorted(owned.items())), captured, produced, tuple(flights))

    frontier = beam
    for _depth in range(12):
        nxt = []
        for state in frontier:
            t, owned_t, captured, produced, flights = state
            owned = dict(owned_t)
            # Terminal value of this state if we stop capturing now:
            fin = advance(state, horizon)
            cand_best = (fin[3], len(dict(fin[1])),
                         sum(board.planets[p]["prod"] for p in dict(fin[1])), ())
            if cand_best[0] > best[0]:
                best = cand_best
            for n in neutrals:
                if n in captured or n in owned:
                    continue
                g = board.planets[n]["ships"] + 1.0
                for src in owned:
                    # Earliest launch time: when src garrison reaches g+1
                    # (keep 1 home), given current garrison + prod growth.
                    have = owned[src]
                    prod = board.planets[src]["prod"]
                    need = g + 1.0
                    wait = 0.0 if have >= need else (
                        math.inf if prod <= 0 else math.ceil((need - have) / prod))
                    t_launch = t + wait
                    if t_launch >= horizon:
                        continue
                    e = board.eta(src, n, g, t_launch)
                    if t_launch + e >= horizon + 10:
                        continue
                    s2 = advance(state, t_launch)
                    t2, owned2_t, cap2, prod2, fl2 = s2
                    owned2 = dict(owned2_t)
                    if owned2.get(src, 0.0) < need:
                        continue
                    owned2[src] -= g
                    fl3 = tuple(sorted(fl2 + ((t_launch + e, n, g),)))
                    nxt.append((t2, tuple(sorted(owned2.items())), cap2, prod2, fl3))
        if not nxt:
            break
        # Beam prune: score = produced-so-far + optimistic in-flight prod value
        def h(s):
            t, owned_t, cap, produced, fl = s
            rate = sum(board.planets[p]["prod"] for p, _ in owned_t)
            opt = produced + rate * (horizon - t)
            for at, tgt, _sz in fl:
                if at < horizon:
                    opt += board.planets[tgt]["prod"] * (horizon - at)
            return opt
        nxt.sort(key=h, reverse=True)
        frontier = nxt[:beam_width]
    for state in frontier:
        fin = advance(state, horizon)
        if fin[3] > best[0]:
            best = (fin[3], len(dict(fin[1])),
                    sum(board.planets[p]["prod"] for p in dict(fin[1])), ())
    return best


def actual_opening(replay: dict, seat: int, horizon: int):
    """Produced-by-horizon, planets and prod at horizon, from the replay."""
    steps = replay["steps"]
    produced = 0.0
    for k in range(1, min(horizon + 1, len(steps))):
        obs = steps[k - 1][0]["observation"]
        produced += sum(float(p[6]) for p in obs["planets"] if int(p[1]) == seat)
    kH = min(horizon, len(steps) - 1)
    obsH = steps[kH][0]["observation"]
    mine = [p for p in obsH["planets"] if int(p[1]) == seat]
    return produced, len(mine), sum(float(p[6]) for p in mine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--team", required=True)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--limit", type=int, default=12, help="max replays")
    ap.add_argument("--safe-only", action="store_true",
                    help="optimum restricted to neutrals closer to us than to the opponent")
    args = ap.parse_args()

    rows = []
    n = 0
    for dd in args.dirs:
        for path in sorted(glob.glob(os.path.join(dd, "episode-*-replay.json"))):
            if n >= args.limit:
                break
            d = json.load(open(path))
            names = (d.get("info") or {}).get("TeamNames") or []
            if len(names) != 2:
                continue
            seats = [i for i, nm in enumerate(names) if args.team.lower() in (nm or "").lower()]
            if len(seats) != 1:
                continue
            seat = seats[0]
            obs0 = d["steps"][0][0]["observation"]
            board = Board(obs0)
            my_start = [pid for pid, p in board.planets.items() if p["owner"] == seat]
            opp_start = [pid for pid, p in board.planets.items()
                         if p["owner"] >= 0 and p["owner"] != seat]
            opt = optimal_schedule(board, my_start, args.horizon,
                                   safe_only_vs=opp_start if args.safe_only else None)
            act = actual_opening(d, seat, args.horizon)
            rows.append((os.path.basename(path), opt, act))
            n += 1

    if not rows:
        print("no matching 2P replays")
        return
    print(f"{'replay':36s} {'opt_prod':>8} {'act_prod':>8} {'gap%':>6}  "
          f"{'opt_pl':>6} {'act_pl':>6}  {'opt_rate':>8} {'act_rate':>8}")
    gaps = []
    for name, opt, act in rows:
        gap = 100.0 * (1.0 - act[0] / opt[0]) if opt[0] > 0 else 0.0
        gaps.append(gap)
        print(f"{name:36s} {opt[0]:8.0f} {act[0]:8.0f} {gap:6.1f}  "
              f"{opt[1]:6d} {act[1]:6d}  {opt[2]:8.0f} {act[2]:8.0f}")
    gaps.sort()
    print(f"\nproduction-by-{args.horizon} gap: median {gaps[len(gaps)//2]:.1f}%  "
          f"worst {gaps[-1]:.1f}%  best {gaps[0]:.1f}%  (n={len(gaps)})")


if __name__ == "__main__":
    main()
