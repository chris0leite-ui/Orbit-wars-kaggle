"""4P FFA validation: aggressive_sizing focal vs frozen weak background.

Uses the standard panel (weakest, enemy_first, baseline) per
audit/2026-05-11-block-e-snipe-mvp.md. Gate: first-place rate
Wilson lo >= 0.90 (prior v3_snipe baseline was 93.8%).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ffa_tournament import run_ffa_tournament

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
    parser.add_argument("--seeds", type=int, default=8)  # 8 × 4 seats = 32 games
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--focal", default="aggressive_sizing")
    args = parser.parse_args()

    if args.focal == "aggressive_sizing":
        focal_path = str(REPO / "agents" / "v35_iter2" / "aggressive_sizing" / "main.py")
    else:
        focal_path = str(REPO / "agents" / args.focal / "main.py")

    background = [
        str(REPO / "agents" / "simple" / "weakest.py"),
        str(REPO / "agents" / "simple" / "enemy_first.py"),
        str(REPO / "data" / "main.py"),  # comp-shipped baseline
    ]

    print(f"4P FFA: focal={args.focal}, background=weakest+enemy_first+baseline")
    print(f"seeds={args.seeds}, workers={args.workers}, {args.seeds * 4} games per focal")

    result = run_ffa_tournament(
        focal=focal_path,
        background=background,
        seeds=list(range(args.seeds)),
        rotate_seats=True,
        workers=args.workers,
        progress=False,
    )

    wins = result.first_place_count
    n = result.n_games
    wr = wins / n if n else 0.0
    wilson_lo = wilson_lower(wins, n)
    verdict = "PASS" if wilson_lo >= 0.90 else ("NEUTRAL" if wilson_lo >= 0.80 else "FAIL")
    print()
    print(f"=== 4P FFA result ===")
    print(f"{args.focal}: {wins}/{n} first-place wins = {wr:.1%}  Wilson lo {wilson_lo:.1%}  [{verdict}]")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"ffa-{args.focal}-{utc}.json"
    out_path.write_text(json.dumps(result.to_json(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
