"""Phase 3c ablation runner: A/B each ablation vs v7_0_drop_one.

For each ablation bundle, run N-seed × 2-seat 2P games and report Wilson
95% lower bound. Identifies which lever from v7_wide_deep helped vs
hurt.

Usage:
    python -m scripts.run_ablations --seeds 8
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
V7_0 = REPO / "submissions" / "v7_0_drop_one.py"

ABLATIONS = [
    ("abl_combined",  "WIDER enumerator (combined)"),
    ("abl_K15",       "DEEPER K=15"),
    ("abl_maximin",   "MAXIMIN over Tier 0 + Tier 1"),
    ("abl_value",     "COMPOSITE value_fn"),
    ("abl_lite",      "LITE follow-up policy"),
]


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def _play(args):
    seed, p0_path, p1_path = args
    env = make("orbit_wars", configuration={"seed": seed})
    env.run([str(p0_path), str(p1_path)])
    r = [s.reward for s in env.state]
    if r[0] is None or r[1] is None:
        return ("ERROR", seed)
    if r[0] > r[1]: return ("P0_WIN", seed)
    if r[1] > r[0]: return ("P1_WIN", seed)
    return ("DRAW", seed)


def _run_ab(focal_path: Path, label: str, seeds: list[int], workers: int):
    print(f"\n=== {label} ({focal_path.stem}) ===")
    pairs_p0 = [(s, focal_path, V7_0) for s in seeds]   # focal as P0
    pairs_p1 = [(s, V7_0, focal_path) for s in seeds]   # focal as P1 (swap)
    focal_wins_p0 = 0
    focal_wins_p1 = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs_p0 = [ex.submit(_play, p) for p in pairs_p0]
        futs_p1 = [ex.submit(_play, p) for p in pairs_p1]
        for f in as_completed(futs_p0):
            o, _ = f.result()
            if o == "P0_WIN": focal_wins_p0 += 1
        for f in as_completed(futs_p1):
            o, _ = f.result()
            if o == "P1_WIN": focal_wins_p1 += 1
    elapsed = time.perf_counter() - t0
    wins = focal_wins_p0 + focal_wins_p1
    total = 2 * len(seeds)
    wlo = _wilson_lower(wins, total)
    verdict = "PASS" if wlo >= 0.55 else ("NEUTRAL" if wins >= total / 2 else "FAIL")
    print(
        f"  {focal_path.stem} wins: {wins}/{total} ({100*wins/total:.0f}%)  "
        f"per-seat P0={focal_wins_p0}/{len(seeds)} P1={focal_wins_p1}/{len(seeds)}  "
        f"Wilson_lo={wlo:.3f}  [{verdict}]  {elapsed:.0f}s"
    )
    return {"label": label, "wins": wins, "total": total, "wilson_lo": wlo,
            "verdict": verdict, "elapsed": elapsed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    seeds = list(range(args.seeds))
    print(f"Ablations vs v7_0_drop_one — {args.seeds} seeds × 2 seats = "
          f"{args.seeds * 2} games per ablation, {args.workers} workers")
    print(f"Gate: Wilson 95% lower bound ≥ 0.55 to PASS")

    results = []
    t_total = time.perf_counter()
    for name, label in ABLATIONS:
        bundle = REPO / "submissions" / f"{name}.py"
        if not bundle.is_file():
            print(f"\nSKIP {name} (no bundle)")
            continue
        results.append(_run_ab(bundle, label, seeds, args.workers))
    total_elapsed = time.perf_counter() - t_total

    print(f"\n=== SUMMARY ({total_elapsed:.0f}s total) ===")
    for r in sorted(results, key=lambda x: -x["wilson_lo"]):
        print(
            f"  {r['verdict']:8s} {r['label']:42s}  "
            f"{r['wins']:2d}/{r['total']}  Wilson_lo={r['wilson_lo']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
