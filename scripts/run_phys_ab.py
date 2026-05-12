"""A/B: current v3_snipe (with waves 1a+1b) vs v3_snipe_frozen (clean v3.4).

Isolates whether the lib/mechanism.py off-by-one fix + the snipe.py
denominator rebalance regress vs the frozen baseline.
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


def wilson_lower(wins: int, n: int) -> float:
    if n == 0: return 0.0
    z = 1.96
    p = wins / n
    denom = 1 + z*z / n
    centre = p + z*z / (2 * n)
    rad = z * ((p * (1 - p) / n) + z*z / (4 * n * n)) ** 0.5
    return (centre - rad) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "v3_snipe_curr": str(REPO / "agents" / "v3_snipe" / "main.py"),
        "v3_snipe_frozen": str(REPO / "submissions" / "v3_snipe_frozen.py"),
    }
    print(f"Phys A/B: current v3_snipe (waves 1a+1b) vs frozen v3.4. seeds={args.seeds}")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=True,
    )

    a = "v3_snipe_curr"
    b = "v3_snipe_frozen"
    p0_stat = result.matrix[a][b]
    p1_stat = result.matrix[b][a]
    curr_wins = p0_stat.p0_wins + p1_stat.p1_wins
    n = p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws
    wr = curr_wins / n if n else 0.0
    wilson_lo = wilson_lower(curr_wins, n)
    verdict = "PASS" if wilson_lo >= 0.55 else ("NEUTRAL" if wilson_lo >= 0.45 else "FAIL")
    print()
    print(f"=== Phys A/B result ===")
    print(f"v3_snipe_curr (waves 1a+1b) vs v3_snipe_frozen: {curr_wins}/{n} = {wr:.1%}  Wilson lo {wilson_lo:.1%}  [{verdict}]")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"phys-ab-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
