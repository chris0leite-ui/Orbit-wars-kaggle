"""A/B harness: analytical vs trajectory baseline — fast feedback ladder.

Each seed plays ONE game (focal at P0; no seat swap). Tiers grow as
2 → 4 → 8 so we get a directional signal from 2 games (~30 s), confirm
at 4 (~60 s), settle at 8 (~2 min). Wilson 95% CI is recomputed at every
tier; early-stop on PASS (Wlo ≥ gate) or FAIL (Whi < gate).

Writes a JSON snapshot to `audit/tournaments/analytical-ab-<UTC>.json`.

Usage:
    python -m scripts.analytical_ab            # default ladder 2/4/8
    python -m scripts.analytical_ab --n 8 --workers 4
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
TIER_LADDER = [2, 4, 8]


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
    parser.add_argument("--n", type=int, default=TIER_LADDER[-1],
                        help=f"max games (default {TIER_LADDER[-1]}; "
                             f"ladder {TIER_LADDER}). Focal as P0 only.")
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

    max_n = int(args.n)
    tiers = [t for t in TIER_LADDER if t <= max_n]
    if not tiers or tiers[-1] != max_n:
        tiers.append(max_n)
    tiers = sorted(set(t for t in tiers if t > 0))

    all_seeds = list(SEEDS_64[:max_n])

    print(f"=== analytical_ab.py — fast-feedback ladder ===")
    print(f"focal:    {args.analytical}")
    print(f"baseline: {args.baseline}")
    print(f"ladder:   {tiers}   gate Wlo ≥ {args.gate_threshold}   "
          f"workers={args.workers}")
    print()

    cumulative_wins = 0
    cumulative_n = 0
    last_idx = 0
    verdict = "FAIL"
    per_game: list[dict] = []
    t_start = time.time()

    for tier_n in tiers:
        new_seeds = all_seeds[last_idx:tier_n]
        if not new_seeds:
            continue
        tasks = [(s, args.analytical, args.baseline) for s in new_seeds]
        t0 = time.time()
        if args.workers <= 1:
            results = [_play_seed(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_play_seed, t): t[0] for t in tasks}
                results = [f.result() for f in as_completed(futs)]
        tier_elapsed = time.time() - t0
        tier_wins = sum(1 for _, outcome in results if outcome == "p0_win")
        tier_n_games = len(results)
        cumulative_wins += tier_wins
        cumulative_n += tier_n_games
        last_idx = tier_n
        for s, o in results:
            per_game.append({"seed": s, "outcome": o,
                             "focal_won": o == "p0_win"})

        lo, hi = _wilson_ci(cumulative_wins, cumulative_n)
        wr = cumulative_wins / cumulative_n
        print(f"  tier n={cumulative_n:<2d}  "
              f"wins={cumulative_wins:>2d}/{cumulative_n:<2d}  "
              f"({100*wr:>5.1f}%)  Wlo={lo:.3f}  Whi={hi:.3f}  "
              f"elapsed={tier_elapsed:.1f}s", end="  ")
        if lo >= args.gate_threshold:
            verdict = "PASS"
            print(f"-> STOP  verdict=PASS")
            break
        if hi < args.gate_threshold:
            verdict = "FAIL"
            print(f"-> STOP  verdict=FAIL  (Whi<{args.gate_threshold})")
            break
        if cumulative_n >= max_n:
            verdict = ("PASS" if lo >= args.gate_threshold
                       else ("FAIL" if hi < args.gate_threshold
                             else "INCONCLUSIVE"))
            print(f"-> STOP  verdict={verdict}  (max games)")
            break
        print("-> CONTINUE")

    total_elapsed = time.time() - t_start
    lo, hi = _wilson_ci(cumulative_wins, cumulative_n)

    summary = {
        "wins": cumulative_wins,
        "losses": cumulative_n - cumulative_wins,
        "draws": 0,
        "n": cumulative_n,
        "winrate": cumulative_wins / max(1, cumulative_n),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "verdict": verdict,
        "tiers_used": [t for t in tiers if t <= cumulative_n] or [cumulative_n],
        "elapsed_seconds": round(total_elapsed, 1),
        "seeds_played": [g["seed"] for g in per_game],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"analytical-ab-{utc}.json"
    with out_path.open("w") as f:
        json.dump({"summary": summary, "per_game": per_game}, f, indent=2)

    print()
    print(f"=== result ===")
    print(f"games:    {summary['n']}  (focal P0 only — no seat swap)")
    print(f"wins:     {summary['wins']}/{summary['n']}  "
          f"({100*summary['winrate']:.1f}%)")
    print(f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"verdict:  {verdict}")
    print(f"elapsed:  {total_elapsed:.1f}s")
    print(f"snapshot: {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
