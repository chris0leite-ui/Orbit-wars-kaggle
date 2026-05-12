"""2P A/B: v7.1_oppaware (focal) vs v7_minimax (frozen baseline).

Tests whether threading opp-action predictions into arrival_size
clears the canonical Wilson lo ≥ 55% promotion gate.

v7_minimax is at LIVE μ=1063.0 — our team peak. v7.1's binding gate
is to NOT regress: Wilson lo ≥ 45% is parity-safe, ≥ 55% is a clear
win signal. Anything below 45% is a regression and v7.1 is shelved.

Usage:
    python -m scripts.run_v71_oppaware_ab --seeds 16          # smoke
    python -m scripts.run_v71_oppaware_ab --seeds 32          # binding
    python -m scripts.run_v71_oppaware_ab --seeds 32 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.tournament import run_tournament

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
    parser.add_argument("--baseline", default="v7_minimax",
                        help="frozen baseline (default: v7_minimax)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = {
        "v7.1_oppaware": str(REPO / "agents" / "v7.1_oppaware" / "main.py"),
        args.baseline: str(REPO / "agents" / args.baseline / "main.py"),
    }
    print(f"{args.seeds}-seed 2P A/B: v7.1_oppaware vs {args.baseline}"
          f" (workers={args.workers})")

    result = run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=False,
        workers=args.workers,
        progress=False,
    )

    a = "v7.1_oppaware"
    b = args.baseline
    p0_stat = result.matrix[a][b]
    p1_stat = result.matrix[b][a]
    a_wins = p0_stat.p0_wins + p1_stat.p1_wins
    a_draws = p0_stat.draws + p1_stat.draws
    n = (p0_stat.p0_wins + p0_stat.p1_wins + p0_stat.draws
         + p1_stat.p0_wins + p1_stat.p1_wins + p1_stat.draws)
    wr = a_wins / n if n else 0.0
    wilson_lo = wilson_lower(a_wins, n)
    # Half-credit for draws (TrueSkill-style); useful when draw rate high.
    adjusted = (a_wins + 0.5 * a_draws) / n if n else 0.0
    if wilson_lo >= 0.55:
        verdict = "PASS"
    elif wilson_lo >= 0.45:
        verdict = "NEUTRAL"
    else:
        verdict = "FAIL"

    print()
    print("=== Result ===")
    print(f"v7.1_oppaware as P0 vs {b} as P1: "
          f"{p0_stat.p0_wins}/{p0_stat.p0_wins+p0_stat.p1_wins+p0_stat.draws}"
          f" wins ({p0_stat.draws} draws)")
    print(f"v7.1_oppaware as P1 vs {b} as P0: "
          f"{p1_stat.p1_wins}/{p1_stat.p0_wins+p1_stat.p1_wins+p1_stat.draws}"
          f" wins ({p1_stat.draws} draws)")
    print(f"Total: {a_wins}/{n}  draws={a_draws}  raw={wr:.1%}"
          f"  adj={adjusted:.1%}  Wilson lo={wilson_lo:.1%}  [{verdict}]")

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"v71-oppaware-vs-{b}-{args.seeds}seed-{utc}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    print(f"\nFull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
