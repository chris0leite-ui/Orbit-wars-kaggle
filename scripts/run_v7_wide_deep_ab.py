"""Phase 3c local A/B: v7_wide_deep vs v7_0_drop_one.

32-seed 2P, both seats; reports wins and Wilson 95% lower bound.
Optional 8-seed 4P FFA after the 2P A/B.

Usage:
    python -m scripts.run_v7_wide_deep_ab --seeds 32 --workers 8
    python -m scripts.run_v7_wide_deep_ab --seeds 32 --skip-4p
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from kaggle_environments import make


REPO = Path(__file__).resolve().parents[1]
WIDE = REPO / "submissions" / "v7_wide_deep.py"
V7_0 = REPO / "submissions" / "v7_0_drop_one.py"
V35 = REPO / "submissions" / "v3.5.1.py"


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def _play_one_2p(args):
    seed, p0_path, p1_path = args
    env = make("orbit_wars", configuration={"seed": seed})
    env.run([str(p0_path), str(p1_path)])
    rewards = [s.reward for s in env.state]
    if rewards[0] is None or rewards[1] is None:
        return (seed, "ERROR", 0)
    if rewards[0] > rewards[1]:
        return (seed, "P0_WIN", env.state[0].observation.step)
    if rewards[1] > rewards[0]:
        return (seed, "P1_WIN", env.state[0].observation.step)
    return (seed, "DRAW", env.state[0].observation.step)


def _play_one_4p(args):
    seed, focal_path, bg_path = args
    env = make("orbit_wars", configuration={"seed": seed})
    env.run([str(focal_path), str(bg_path), str(bg_path), str(bg_path)])
    rewards = [s.reward for s in env.state]
    if any(r is None for r in rewards):
        return (seed, "ERROR")
    return (seed, "FOCAL_WIN" if rewards[0] == 1 else "FOCAL_LOSS")


def _run_2p_ab(seeds, workers):
    print(f"\n=== 2P A/B: v7_wide_deep (P0) vs v7_0_drop_one (P1), {len(seeds)} seeds ===")
    pairs_p0 = [(s, WIDE, V7_0) for s in seeds]
    pairs_p1 = [(s, V7_0, WIDE) for s in seeds]  # swap seats
    results_p0 = []
    results_p1 = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs_p0 = {ex.submit(_play_one_2p, p): p[0] for p in pairs_p0}
        futs_p1 = {ex.submit(_play_one_2p, p): p[0] for p in pairs_p1}
        for f in as_completed({**futs_p0, **futs_p1}):
            seed = futs_p0.get(f) or futs_p1.get(f)
            (which, outcome, steps) = (None, *f.result()[1:])
            if f in futs_p0:
                results_p0.append((seed, outcome, steps))
            else:
                results_p1.append((seed, outcome, steps))
    elapsed = time.perf_counter() - t0
    wide_wins_p0 = sum(1 for _, o, _ in results_p0 if o == "P0_WIN")
    wide_wins_p1 = sum(1 for _, o, _ in results_p1 if o == "P1_WIN")
    p0_draws = sum(1 for _, o, _ in results_p0 if o == "DRAW")
    p1_draws = sum(1 for _, o, _ in results_p1 if o == "DRAW")
    total_games = len(results_p0) + len(results_p1)
    wide_wins = wide_wins_p0 + wide_wins_p1
    draws = p0_draws + p1_draws
    losses = total_games - wide_wins - draws
    wr = wide_wins / total_games if total_games else 0
    wilson_lo = _wilson_lower(wide_wins, total_games)
    print(
        f"v7_wide_deep wins: {wide_wins}/{total_games} ({100*wr:.1f}%) "
        f"draws={draws} losses={losses}"
    )
    print(f"Wilson 95% lower: {wilson_lo:.3f}  (pass gate = 0.550)")
    print(f"Per-seat: P0 {wide_wins_p0}/{len(seeds)}, P1 {wide_wins_p1}/{len(seeds)}")
    print(f"Wallclock: {elapsed:.1f} s ({elapsed/total_games:.2f} s/game)")
    return {"wins": wide_wins, "total": total_games, "wilson_lo": wilson_lo}


def _run_4p_ffa(seeds, workers):
    print(f"\n=== 4P FFA: v7_wide_deep focal vs 3×v3.5.1 bg, {len(seeds)} seeds ===")
    pairs = [(s, WIDE, V35) for s in seeds]
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_play_one_4p, p): p[0] for p in pairs}
        for f in as_completed(futs):
            results.append(f.result())
    elapsed = time.perf_counter() - t0
    focal_wins = sum(1 for _, o in results if o == "FOCAL_WIN")
    total = len(results)
    print(
        f"v7_wide_deep wins: {focal_wins}/{total} ({100*focal_wins/total:.1f}%)"
    )
    print(f"Wallclock: {elapsed:.1f} s")
    return {"wins": focal_wins, "total": total}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ffa-seeds", type=int, default=8)
    ap.add_argument("--skip-4p", action="store_true")
    args = ap.parse_args(argv)

    for p in (WIDE, V7_0, V35):
        if not p.is_file():
            print(f"missing bundle: {p}")
            return 1

    seeds_2p = list(range(args.seeds))
    res2 = _run_2p_ab(seeds_2p, args.workers)
    if not args.skip_4p:
        seeds_4p = list(range(args.ffa_seeds))
        res4 = _run_4p_ffa(seeds_4p, args.workers)

    print()
    print("PASS GATE:" if res2["wilson_lo"] >= 0.55 else "FAIL GATE:")
    print(
        f"  2P A/B: {res2['wins']}/{res2['total']} wins, "
        f"Wilson lo = {res2['wilson_lo']:.3f}"
    )
    return 0 if res2["wilson_lo"] >= 0.55 else 2


if __name__ == "__main__":
    sys.exit(main())
