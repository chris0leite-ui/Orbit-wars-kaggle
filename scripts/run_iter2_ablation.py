"""Run iter-2 ablation panel: 4 surgical variants × v3_snipe baseline.

Variants: aggressive_sizing, endgame_burn, frontier_keep, recapture_tight.
Each tests ONE surgical change vs v3_snipe (current lib, includes
waves 1a+1b from earlier in this session).
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
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "v3_snipe": str(REPO / "agents" / "v3_snipe" / "main.py"),
        "aggressive_sizing": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing" / "main.py"),
        "endgame_burn": str(REPO / "agents" / "v35_iter2" / "endgame_burn" / "main.py"),
        "frontier_keep": str(REPO / "agents" / "v35_iter2" / "frontier_keep" / "main.py"),
        "recapture_tight": str(REPO / "agents" / "v35_iter2" / "recapture_tight" / "main.py"),
    }
    print(f"Iter-2 ablation: 5 agents × {args.seeds} seeds × both-sides, workers={args.workers}")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=False,
    )

    print()
    print(f"=== Iter-2 ablation results: each variant vs v3_snipe baseline ===")
    summary_rows = []
    for variant in ["aggressive_sizing", "endgame_burn", "frontier_keep", "recapture_tight"]:
        p0_stat = result.matrix[variant]["v3_snipe"]
        p1_stat = result.matrix["v3_snipe"][variant]
        v_wins = p0_stat.p0_wins + p1_stat.p1_wins
        v_draws = p0_stat.draws + p1_stat.draws
        n = (p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
             + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws)
        wr = v_wins / n if n else 0.0
        wilson_lo = wilson_lower(v_wins, n)
        verdict = "PASS" if wilson_lo >= 0.55 else ("NEUTRAL" if wilson_lo >= 0.45 else "FAIL")
        print(f"  {variant:<20}: {v_wins:>3}/{n} ({v_draws} draws) = {wr:>5.1%}  Wilson lo {wilson_lo:>5.1%}  [{verdict}]")
        summary_rows.append((variant, v_wins, n, wr, wilson_lo, verdict))

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"iter2-ablation-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
