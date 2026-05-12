"""Run 4 ablation A/Bs in one tournament: each variant vs v3_snipe baseline.

Each ablation isolates ONE wave's contribution. The baseline `v3_snipe`
uses the CURRENT lib/ (so it includes wave 1a + 1b changes); the
variants add ONE additional mission class on top.

Goal: find which wave(s) beat the v3_snipe baseline at Wilson lo >= 55%.
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
    if n == 0:
        return 0.0
    z = 1.96
    p = wins / n
    denom = 1 + z*z / n
    centre = p + z*z / (2 * n)
    rad = z * ((p * (1 - p) / n) + z*z / (4 * n * n)) ** 0.5
    return (centre - rad) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "v3_snipe": str(REPO / "agents" / "v3_snipe" / "main.py"),
        "opening_only": str(REPO / "agents" / "v35_ablations" / "opening_only" / "main.py"),
        "drain_only": str(REPO / "agents" / "v35_ablations" / "drain_only" / "main.py"),
        "gangup_only": str(REPO / "agents" / "v35_ablations" / "gangup_only" / "main.py"),
        "recapture_only": str(REPO / "agents" / "v35_ablations" / "recapture_only" / "main.py"),
    }
    print(f"Ablation panel: 5 agents × {args.seeds} seeds × both-sides, workers={args.workers}")
    print("This is round-robin without self-play.")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=True,
    )

    print()
    print(f"=== Ablation results: each variant vs v3_snipe baseline ===")
    for variant in ["opening_only", "drain_only", "gangup_only", "recapture_only"]:
        # variant as P0
        p0_stat = result.matrix[variant]["v3_snipe"]
        v_wins_as_p0 = p0_stat.p0_wins
        n_p0 = p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
        # variant as P1
        p1_stat = result.matrix["v3_snipe"][variant]
        v_wins_as_p1 = p1_stat.p1_wins
        n_p1 = p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws
        total_wins = v_wins_as_p0 + v_wins_as_p1
        total_games = n_p0 + n_p1
        wr = total_wins / total_games if total_games else 0.0
        wilson_lo = wilson_lower(total_wins, total_games)
        verdict = "PASS" if wilson_lo >= 0.55 else ("NEUTRAL" if wilson_lo >= 0.45 else "FAIL")
        print(f"  {variant:<18}: {total_wins:>3}/{total_games} = {wr:>5.1%}  Wilson lo {wilson_lo:>5.1%}  [{verdict}]")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"v35-ablation-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
