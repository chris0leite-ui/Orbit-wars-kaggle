"""Local A/B: v3_snipe with recapture wire-up vs the same pre-recapture
baseline (`/tmp/v3_snipe_baseline_main.py`). 100 seeds × both sides
(P0/P1) = 200 games. Logs per-side and combined Wilson lower bound.

Gate (plan §Phase 4): Wilson lower bound on combined winrate >= 55%
before any submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tournament import run_tournament  # noqa: E402

CANDIDATE = str(REPO / "agents" / "v3_snipe" / "main.py")
BASELINE = "/tmp/v3_snipe_baseline_main.py"


def main(seeds: list[int], workers: int) -> int:
    result = run_tournament(
        agents={"recapture": CANDIDATE, "baseline": BASELINE},
        seeds=seeds,
        include_self_play=False,
        out_dir=REPO / "audit" / "tournaments",
        progress=False,
        workers=workers,
    )
    # result.matrix[row][col]: stat for games where row plays P0 vs col P1.
    cand_as_p0 = result.matrix["recapture"]["baseline"]   # candidate as P0
    cand_as_p1 = result.matrix["baseline"]["recapture"]   # candidate as P1
    # P0/P1 wins are from row=P0 perspective. Translate.
    cand_p0_wins = cand_as_p0.p0_wins
    cand_p1_wins = cand_as_p1.p1_wins  # candidate plays P1 here
    cand_total_wins = cand_p0_wins + cand_p1_wins
    cand_total_games = cand_as_p0.n + cand_as_p1.n
    # Wilson 95% on combined.
    import math
    n = cand_total_games
    p = cand_total_wins / n if n else 0.0
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    wilson_lo = centre - half
    wilson_hi = centre + half

    print("=== Recapture A/B summary (100 seeds × 2 sides = 200 games) ===")
    print(f"  candidate as P0: {cand_p0_wins}/{cand_as_p0.n}  "
          f"(Wilson lo on side {cand_as_p0.wilson_lo:.3f})")
    print(f"  candidate as P1: {cand_p1_wins}/{cand_as_p1.n}  "
          f"(Wilson lo on side {cand_as_p1.wilson_lo:.3f})")
    print(f"  candidate combined: {cand_total_wins}/{n} = {p:.3f}")
    print(f"  Wilson 95% CI on combined: [{wilson_lo:.3f}, {wilson_hi:.3f}]")
    gate = wilson_lo >= 0.55
    print(f"  GATE (lo >= 0.55): {'PASS' if gate else 'FAIL'}")
    print()
    # P95 turn time for budget check
    print(f"  p95 turn ms candidate (P0): {cand_as_p0.p0_p95_turn_ms:.1f}")
    print(f"  p95 turn ms candidate (P1): {cand_as_p1.p1_p95_turn_ms:.1f}")
    return 0 if gate else 1


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100, help="number of seeds (0..N-1)")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    sys.exit(main(list(range(args.seeds)), args.workers))
