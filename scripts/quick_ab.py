"""Tiny A/B: focal-as-P0 only, N seeds, no seat alternation.

Usage:
    python scripts/quick_ab.py focal.py opp.py [--seeds 4] [--workers 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def play_one(args):
    seed, focal_path, opp_path = args
    sys.path.insert(0, str(REPO))
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([focal_path, opp_path])
    final = env.steps[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    n_steps = len(env.steps)
    if r0 > r1:
        outcome = "win"
    elif r1 > r0:
        outcome = "loss"
    else:
        outcome = "draw"
    return (seed, outcome, n_steps)


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("opp")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    focal = str(Path(args.focal).resolve())
    opp = str(Path(args.opp).resolve())
    print(f"== focal={Path(focal).name}  opp={Path(opp).name}  "
          f"seeds={args.seeds}  workers={args.workers} (focal=P0 only) ==")

    tasks = [(s, focal, opp) for s in range(args.seeds)]
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(play_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"   seed={r[0]:>4d}  {r[1]:>4s}  steps={r[2]}")
    wins = sum(1 for r in results if r[1] == "win")
    losses = sum(1 for r in results if r[1] == "loss")
    draws = sum(1 for r in results if r[1] == "draw")
    n = len(results)
    lo, hi = wilson_ci(wins, n)
    elapsed = time.perf_counter() - t0
    print(f"\n   wins={wins}/{n} ({100*wins/n:.1f}%)  losses={losses}  draws={draws}  "
          f"Wilson[{lo:.3f}, {hi:.3f}]  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
