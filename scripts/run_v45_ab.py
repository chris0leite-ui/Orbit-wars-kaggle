"""A/B harness for v4.5_robust vs v4_planner + diverse-panel gate.

Two modes:

  --mode ab     16-32 seed head-to-head v4.5_robust vs v4_planner.
                Wilson lo gate (≥ 55% to clear).
  --mode panel  16-seed diverse-panel: v4.5_robust vs each of
                {v4_planner, v7_minimax, v3_snipe, v3.5.1}.
                Promoted "diverse-panel" rule: focal must beat
                ≥ 50 % of panel agents at Wilson lo ≥ 55 %
                OR mean panel WR ≥ v4_planner's mean panel WR
                with no cell regressing > 5pp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tournament import run_tournament, _wilson_ci  # noqa: E402

AGENT_PATHS = {
    "v4.5_robust": str(REPO / "agents" / "v4.5_robust" / "main.py"),
    "v4_planner":  str(REPO / "agents" / "v4_planner"  / "main.py"),
    "v7_minimax":  str(REPO / "agents" / "v7_minimax"  / "main.py"),
    "v3_snipe":    str(REPO / "agents" / "v3_snipe"    / "main.py"),
    "v3.5.1":      str(REPO / "agents" / "v3.5.1"      / "main.py"),
}


def _pooled(matrix, focal: str) -> tuple[int, int]:
    """Sum focal wins / total games across both seats."""
    wins = 0
    n = 0
    for row in matrix:
        for col, stat in matrix[row].items():
            if row == focal:
                wins += stat.p0_wins
                n += stat.n
            if col == focal:
                wins += stat.p1_wins
                n += stat.n
    return wins, n


def _ab(seeds: int, workers: int, opponent: str, out_dir: Path):
    result = run_tournament(
        agents={
            "v4.5_robust": AGENT_PATHS["v4.5_robust"],
            opponent:      AGENT_PATHS[opponent],
        },
        seeds=list(range(seeds)),
        include_self_play=False,
        out_dir=out_dir,
        progress=True,
        workers=workers,
    )
    print()
    print(f"=== v4.5_robust vs {opponent} ({seeds} seeds, both seats) ===")
    for row in result.matrix:
        for col, stat in result.matrix[row].items():
            print(
                f"  {row}(P0) vs {col}(P1): {stat.p0_wins}/{stat.n} "
                f"draws={stat.draws}  "
                f"Wilson95% [{stat.wilson_lo:.3f}, {stat.wilson_hi:.3f}]; "
                f"p95 P0={stat.p0_p95_turn_ms:.1f}ms P1={stat.p1_p95_turn_ms:.1f}ms"
            )
    wins, n = _pooled(result.matrix, "v4.5_robust")
    lo, hi = _wilson_ci(wins, n)
    rate = wins / n if n else 0.0
    verdict = "PASS" if lo >= 0.55 else ("NEUTRAL" if lo >= 0.45 else "FAIL")
    print()
    print(
        f"POOLED v4.5_robust: {wins}/{n} = {rate*100:.1f}% "
        f"Wilson95% [{lo*100:.1f}%, {hi*100:.1f}%]  → {verdict} (gate 55% Wilson lo)"
    )


def _panel(seeds: int, workers: int, out_dir: Path):
    opponents = ["v4_planner", "v7_minimax", "v3_snipe", "v3.5.1"]
    rows = {}
    for opp in opponents:
        result = run_tournament(
            agents={
                "v4.5_robust": AGENT_PATHS["v4.5_robust"],
                opp:           AGENT_PATHS[opp],
            },
            seeds=list(range(seeds)),
            include_self_play=False,
            out_dir=out_dir,
            progress=True,
            workers=workers,
        )
        wins, n = _pooled(result.matrix, "v4.5_robust")
        lo, hi = _wilson_ci(wins, n)
        rows[opp] = (wins, n, lo, hi)
    print()
    print("=== v4.5_robust vs diverse panel ({} seeds × 2 seats each) ===".format(seeds))
    pass_cells = 0
    for opp, (wins, n, lo, hi) in rows.items():
        rate = wins / n if n else 0.0
        flag = "PASS" if lo >= 0.55 else ("NEUTRAL" if lo >= 0.45 else "FAIL")
        if lo >= 0.55:
            pass_cells += 1
        print(
            f"  vs {opp:14s} {wins}/{n} = {rate*100:5.1f}% "
            f"Wilson95% [{lo*100:5.1f}%, {hi*100:5.1f}%]  {flag}"
        )
    mean_wr = sum(w / n for (w, n, _, _) in rows.values()) / len(rows)
    print()
    print(f"mean panel WR: {mean_wr*100:.1f}%; cells passing 55% Wilson lo: {pass_cells}/{len(rows)}")
    gate = "PASS" if pass_cells >= 2 else "FAIL"
    print(f"diverse-panel gate (≥ 50% of cells at Wilson lo ≥ 55%): {gate}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ab", "panel"], default="ab")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--opponent", type=str, default="v4_planner",
                        choices=[k for k in AGENT_PATHS if k != "v4.5_robust"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "audit" / "tournaments")

    if args.mode == "ab":
        _ab(args.seeds, args.workers, args.opponent, out_dir)
    else:
        _panel(args.seeds, args.workers, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
