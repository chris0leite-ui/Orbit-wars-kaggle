"""A/B harness: analytical vs trajectory baseline — single-tier, fast feedback.

Each seed plays ONE game (focal at P0; no seat swap). One invocation =
one tier of `--n` games. Default `--n 4` is the "did this change help?"
gate (~60 s); if it's negative or inconclusive, ASK the PI before
escalating to `--n 8`. No automatic tier escalation.

Writes a JSON snapshot to `audit/tournaments/analytical-ab-<UTC>.json`.

Usage:
    python -m scripts.analytical_ab               # 4 games (~60 s, default)
    python -m scripts.analytical_ab --n 2         # 2 games (~30 s, fastest)
    python -m scripts.analytical_ab --n 8         # 8 games (~2 min, only after PI ok)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.ab_variants import SEEDS_64  # noqa: E402
from fast import play_one  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402


ANALYTICAL_PATH = str(REPO / "agents" / "analytical" / "main.py")
TRAJECTORY_PATH = str(REPO / "agents" / "baseline" / "main.py")
DEFAULT_N = 4
MAX_N_WITHOUT_PI = 4   # >4 needs PI go-ahead per session protocol.


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _play_seed(args: tuple[int, str, str]) -> tuple[int, str]:
    seed, p0_path, p1_path = args
    r = play_one(seed, p0_path, p1_path, record_timing=False)
    return seed, r.outcome


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"games to play (default {DEFAULT_N}; "
                             f"focal as P0 only — no seat swap)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers for game runs")
    parser.add_argument("--out-dir", default=str(REPO / "audit" / "tournaments"),
                        help="directory for the JSON snapshot")
    parser.add_argument("--analytical", default=ANALYTICAL_PATH,
                        help="path to the candidate agent (default: analytical)")
    parser.add_argument("--baseline", default=TRAJECTORY_PATH,
                        help="path to the anchor agent (default: trajectory baseline)")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="Wilson-lo PASS threshold (default 0.5)")
    args = parser.parse_args(argv)

    n = int(args.n)
    if n > MAX_N_WITHOUT_PI:
        print(f"WARNING: n={n} > {MAX_N_WITHOUT_PI}. Per session protocol, "
              f"escalating beyond {MAX_N_WITHOUT_PI} games needs PI go-ahead.")

    seeds = list(SEEDS_64[:n])
    print(f"=== analytical_ab.py — single tier, n={n} ===")
    print(f"focal:    {args.analytical}")
    print(f"baseline: {args.baseline}")
    print(f"seeds:    {seeds}   workers={args.workers}   "
          f"gate Wlo ≥ {args.gate_threshold}")
    print()

    tasks = [(s, args.analytical, args.baseline) for s in seeds]
    t_start = time.time()
    if args.workers <= 1:
        results = [_play_seed(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_play_seed, t): t[0] for t in tasks}
            results = [f.result() for f in as_completed(futs)]
    total_elapsed = time.time() - t_start

    per_game = [{"seed": s, "outcome": o, "focal_won": o == "p0_win"}
                for s, o in results]
    wins = sum(1 for g in per_game if g["focal_won"])
    n_games = len(per_game)
    lo, hi = _wilson_ci(wins, n_games)
    if lo >= args.gate_threshold:
        verdict = "PASS"
    elif hi < args.gate_threshold:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "wins": wins,
        "losses": n_games - wins,
        "draws": 0,
        "n": n_games,
        "winrate": wins / max(1, n_games),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "verdict": verdict,
        "elapsed_seconds": round(total_elapsed, 1),
        "seeds_played": [g["seed"] for g in per_game],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"analytical-ab-{utc}.json"
    with out_path.open("w") as f:
        json.dump({"summary": summary, "per_game": per_game}, f, indent=2)

    print(f"=== result ===")
    for g in per_game:
        mark = "W" if g["focal_won"] else ("L" if g["outcome"] != "draw" else "D")
        print(f"  seed={g['seed']:<4d} {mark}  ({g['outcome']})")
    print()
    print(f"games:    {summary['n']}  (focal P0 only — no seat swap)")
    print(f"wins:     {summary['wins']}/{summary['n']}  "
          f"({100*summary['winrate']:.1f}%)")
    print(f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"verdict:  {verdict}")
    print(f"elapsed:  {total_elapsed:.1f}s")
    print(f"snapshot: {out_path}")
    if verdict != "PASS" and n_games <= MAX_N_WITHOUT_PI:
        print()
        print(f"NEXT: {verdict} at n={n_games}. ASK PI before escalating to n=8.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
