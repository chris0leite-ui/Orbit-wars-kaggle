"""SHIP_FRACTION parameter sweep: 0.6 / 0.7 / 0.8 / 0.9 vs v3_snipe.

Identifies the optimum ship-fraction value. Top-10 empirical implied
fraction is ~0.78; my default 0.7 cleared the gate at Wilson lo 56.6%.
This sweeps the parameter to see if a different value lifts further.
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
        "agg_06": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing_06" / "main.py"),
        "agg_07": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing" / "main.py"),
        "agg_08": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing_08" / "main.py"),
        "agg_09": str(REPO / "agents" / "v35_iter2" / "aggressive_sizing_09" / "main.py"),
    }
    print(f"Sizing sweep: 5 agents × {args.seeds} seeds × both-sides, workers={args.workers}")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=False,
    )

    print()
    print(f"=== Sweep results: each variant vs v3_snipe baseline ===")
    for variant in ["agg_06", "agg_07", "agg_08", "agg_09"]:
        p0_stat = result.matrix[variant]["v3_snipe"]
        p1_stat = result.matrix["v3_snipe"][variant]
        v_wins = p0_stat.p0_wins + p1_stat.p1_wins
        v_draws = p0_stat.draws + p1_stat.draws
        n = (p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
             + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws)
        wr = v_wins / n if n else 0.0
        wilson_lo = wilson_lower(v_wins, n)
        verdict = "PASS" if wilson_lo >= 0.55 else ("NEUTRAL" if wilson_lo >= 0.45 else "FAIL")
        print(f"  {variant:<12}: {v_wins:>3}/{n} ({v_draws} draws) = {wr:>5.1%}  Wilson lo {wilson_lo:>5.1%}  [{verdict}]")

    # Also: head-to-head among the agg_* variants to find the dominant one
    print()
    print(f"=== Head-to-head among agg_* variants ===")
    variants = ["agg_06", "agg_07", "agg_08", "agg_09"]
    for v1 in variants:
        wins = 0
        n_total = 0
        for v2 in variants:
            if v1 == v2:
                continue
            p0_stat = result.matrix[v1][v2]
            p1_stat = result.matrix[v2][v1]
            wins += p0_stat.p0_wins + p1_stat.p1_wins
            n_total += (p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
                         + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws)
        wr = wins / n_total if n_total else 0.0
        print(f"  {v1}: {wins}/{n_total} = {wr:.1%} vs other agg_* variants")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"sizing-sweep-{utc}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
