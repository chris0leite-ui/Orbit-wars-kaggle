"""32-seed 2P A/B: aggressive_sizing variant vs v3_snipe baseline.

Confirms the 16-seed 84.4% / Wilson lo 68.2% result at the canonical
32-seed gate. Wilson lo ≥ 55% required for promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tournament import run_tournament

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "audit" / "tournaments"


def wilson_lower(wins: int, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * ((p * (1 - p) / n) + z * z / (4 * n * n)) ** 0.5
    return (centre - rad) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "aggressive_sizing": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing" / "main.py"),
        "v3_snipe": str(REPO / "agents" / "v3_snipe" / "main.py"),
    }
    print(f"32-seed A/B: aggressive_sizing vs v3_snipe (workers={args.workers})")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=False,
    )

    a = "aggressive_sizing"
    b = "v3_snipe"
    p0_stat = result.matrix[a][b]
    p1_stat = result.matrix[b][a]
    a_wins = p0_stat.p0_wins + p1_stat.p1_wins
    a_draws = p0_stat.draws + p1_stat.draws
    n = (p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
         + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws)
    wr = a_wins / n if n else 0.0
    wilson_lo = wilson_lower(a_wins, n)
    verdict = "PASS" if wilson_lo >= 0.55 else ("NEUTRAL" if wilson_lo >= 0.45 else "FAIL")

    print()
    print(f"=== Result ===")
    print(f"aggressive_sizing as P0 vs v3_snipe as P1: {p0_stat.p0_wins}/{p0_stat.p0_wins+p0_stat.p1_wins+p0_stat.draws} wins ({p0_stat.draws} draws)")
    print(f"aggressive_sizing as P1 vs v3_snipe as P0: {p1_stat.p1_wins}/{p1_stat.p0_wins+p1_stat.p1_wins+p1_stat.draws} wins ({p1_stat.draws} draws)")
    print(f"Total: {a_wins}/{n} ({a_draws} draws) = {wr:.1%}  Wilson lo {wilson_lo:.1%}  [{verdict}]")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"aggressive-sizing-32-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
