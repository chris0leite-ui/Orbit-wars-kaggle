"""Local A/B harness for v4_planner vs v3.5.1.

Step 6 of /root/.claude/plans/you-are-a-senior-wondrous-trinket.md gate:
- 16 seeds (smoke) → Wilson lo > 40% (not catastrophically broken)
- 32 seeds (gate) → Wilson lo > 50% to continue, > 55% strong candidate
- 64 seeds (confirm) → Wilson lo > 50% required to ship

Invocation:
    python -m scripts.run_v4_ab --seeds 16
    python -m scripts.run_v4_ab --seeds 32 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tournament import run_tournament

V4_PATH = str(REPO / "agents" / "v4_planner" / "main.py")
V351_PATH = str(REPO / "agents" / "v3.5.1" / "main.py")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16,
                        help="number of seeds (each runs both seats)")
    parser.add_argument("--workers", type=int, default=1,
                        help="multiprocessing workers (fork)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="directory for JSON artifact")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "audit" / "tournaments")
    result = run_tournament(
        agents={"v4_planner": V4_PATH, "v3.5.1": V351_PATH},
        seeds=list(range(args.seeds)),
        include_self_play=False,
        out_dir=out_dir,
        progress=True,
        workers=args.workers,
    )
    print()
    print(f"=== v4_planner vs v3.5.1 ({args.seeds} seeds, both seats) ===")
    total_v4_wins = 0
    total_games = 0
    for row in result.matrix:
        for col, stat in result.matrix[row].items():
            print(
                f"  {row} (P0) vs {col} (P1): {stat.p0_wins}/{stat.n} "
                f"Wilson 95% [{stat.wilson_lo:.3f}, {stat.wilson_hi:.3f}]; "
                f"p95 P0={stat.p0_p95_turn_ms:.1f}ms P1={stat.p1_p95_turn_ms:.1f}ms; "
                f"draws={stat.draws}"
            )
            if row == "v4_planner":
                total_v4_wins += stat.p0_wins
                total_games += stat.n
            if col == "v4_planner":
                total_v4_wins += stat.p1_wins
                total_games += stat.n
    if total_games > 0:
        # Pooled Wilson over both seats.
        from scripts.tournament import _wilson_ci
        lo, hi = _wilson_ci(total_v4_wins, total_games)
        print()
        print(
            f"POOLED: v4_planner {total_v4_wins}/{total_games} "
            f"= {total_v4_wins/total_games:.1%}, Wilson 95% [{lo:.3f}, {hi:.3f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
