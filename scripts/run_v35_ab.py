"""32-seed 2P A/B: v3.5 vs the v3_snipe baseline (which is now v3.4 in-bundle).

Usage:
    python -m scripts.run_v35_ab [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tournament import run_tournament

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "audit" / "tournaments"
DEFAULT_SEEDS = 32


def wilson_lower(wins: int, n: int) -> float:
    """Lower bound of the 95% Wilson score interval."""
    if n == 0:
        return 0.0
    z = 1.96
    p = wins / n
    denom = 1 + z*z / n
    centre = p + z*z / (2 * n)
    rad = z * ((p * (1 - p) / n) + z*z / (4 * n * n)) ** 0.5
    return (centre - rad) / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "v3.5": str(REPO / "agents" / "v3.5" / "main.py"),
        "v3_snipe": str(REPO / "agents" / "v3_snipe" / "main.py"),
    }
    print(f"A/B: v3.5 vs v3_snipe, {args.seeds} seeds, both sides, workers={args.workers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_DIR
    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        out_dir=out_dir,
        workers=args.workers,
        progress=True,
    )

    # Compute v3.5's overall winrate vs v3_snipe (both sides)
    a = "v3.5"
    b = "v3_snipe"
    # When v3.5 is P0
    p0_stat = result.matrix[a][b]
    # When v3.5 is P1
    p1_stat = result.matrix[b][a]
    # In tournament.py, PairStat.p0_wins is P0's wins; we want v3.5's wins.
    v35_wins_as_p0 = p0_stat.p0_wins
    v35_wins_as_p1 = p1_stat.p1_wins
    n_p0 = p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
    n_p1 = p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws
    total_wins = v35_wins_as_p0 + v35_wins_as_p1
    total_games = n_p0 + n_p1
    wr = total_wins / total_games if total_games else 0.0
    wilson_lo = wilson_lower(total_wins, total_games)

    print()
    print(f"=== A/B result ===")
    print(f"v3.5 as P0 vs v3_snipe as P1: {v35_wins_as_p0}/{n_p0} wins ({p0_stat.draws} draws)")
    print(f"v3.5 as P1 vs v3_snipe as P0: {v35_wins_as_p1}/{n_p1} wins ({p1_stat.draws} draws)")
    print(f"Total: {total_wins}/{total_games} = {wr:.1%}  Wilson lo: {wilson_lo:.1%}")
    print(f"Gate (Wilson lo >= 55%): {'PASS' if wilson_lo >= 0.55 else 'FAIL'}")

    # Save full result
    out_path = out_dir / f"v35-ab-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
