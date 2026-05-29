"""No-swap A/B: focal at P0, opp at P1, archetype-stratified seeds.

Per PI preference (2026-05-29): no seat swapping. Focal as P0 only.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from kaggle_environments import make

from lib.seed_panel import SEED_PANEL_128_INTERLEAVED


def play_one(args):
    seed, p0, p1 = args
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    if r0 is None or r1 is None:
        return (seed, "error")
    if r0 > r1:
        return (seed, "p0_win")
    if r1 > r0:
        return (seed, "p1_win")
    return (seed, "draw")


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("--vs", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-seeds", type=int, default=0,
                    help="skip the first N seeds in the interleaved panel "
                         "(use for distinct re-runs)")
    args = ap.parse_args()

    seeds = SEED_PANEL_128_INTERLEAVED[args.skip_seeds:args.skip_seeds + args.n]
    print(f"== focal-as-P0 A/B: {args.focal} vs {args.vs} ==", flush=True)
    print(f"== n={len(seeds)} archetype-stratified seeds (skip={args.skip_seeds}) "
          f"workers={args.workers} ==", flush=True)
    print(f"== seeds={seeds[:8]}... ==", flush=True)

    tasks = [(s, args.focal, args.vs) for s in seeds]
    t0 = time.perf_counter()
    results: list[tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(play_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            wins = sum(1 for _, o in results if o == "p0_win")
            elapsed = time.perf_counter() - t0
            print(f"  seed={r[0]:5d}  {r[1]:7s}  "
                  f"running={wins}/{len(results)}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    wins = sum(1 for _, o in results if o == "p0_win")
    losses = sum(1 for _, o in results if o == "p1_win")
    draws = sum(1 for _, o in results if o == "draw")
    errors = sum(1 for _, o in results if o == "error")
    n = len(results)
    wr = wins / n if n else 0
    lo, hi = wilson_ci(wins, n)
    print()
    print(f"== RESULT ==", flush=True)
    print(f"  n={n}  wins={wins}  losses={losses}  draws={draws}  errors={errors}",
          flush=True)
    print(f"  focal-as-P0 winrate = {wr:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]",
          flush=True)
    print(f"  elapsed = {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
