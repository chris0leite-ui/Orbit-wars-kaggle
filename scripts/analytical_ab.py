"""Phase 4 A/B harness: analytical agent vs trajectory baseline.

Runs `scripts.tournament.run_tournament` with the two agents both seats,
counts analytical's symmetric winrate, and reports the Wilson 95% CI
lower bound (Wlo). Writes a JSON snapshot to
`audit/tournaments/analytical-ab-<UTC>.json`.

STOP gate (Phase 4 plan): Wlo ≥ 0.5 → escalate to n=32.
Practical interpretation at n=8 (16 games incl. both seats): ≥10/16
gives Wlo ≈ 0.39; ≥12/16 gives Wlo ≈ 0.52. We treat the gate as
"directional positive" (≥10/16) for the n=8 round; n=32 confirms.

Usage:
    python -m scripts.analytical_ab --n 8 [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Imports below the path injection so the script works whether invoked
# as `python -m scripts.analytical_ab` or `python scripts/analytical_ab.py`.
from scripts.ab_variants import SEEDS_64  # noqa: E402
from scripts import tournament  # noqa: E402


ANALYTICAL_PATH = str(REPO / "agents" / "analytical" / "main.py")
TRAJECTORY_PATH = str(REPO / "agents" / "baseline" / "main.py")


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarise(result, analytical_name: str, baseline_name: str) -> dict:
    """Count analytical's wins/losses/draws across both seat orderings."""
    a, b = analytical_name, baseline_name
    # a-as-P0 vs b-as-P1.
    sp_ap = result.matrix[a].get(b)
    # a-as-P1 vs b-as-P0 (matrix[b][a]).
    sp_bp = result.matrix[b].get(a)

    wins = (sp_ap.p0_wins if sp_ap else 0) + (sp_bp.p1_wins if sp_bp else 0)
    losses = (sp_ap.p1_wins if sp_ap else 0) + (sp_bp.p0_wins if sp_bp else 0)
    draws = (sp_ap.draws if sp_ap else 0) + (sp_bp.draws if sp_bp else 0)
    n = wins + losses + draws
    lo, hi = _wilson_ci(wins, n)
    return {
        "wins": wins, "losses": losses, "draws": draws, "n": n,
        "winrate": wins / n if n else 0.0,
        "wilson_lo": lo, "wilson_hi": hi,
        "as_p0_n": sp_ap.n if sp_ap else 0,
        "as_p0_winrate": (sp_ap.p0_wins / sp_ap.n if sp_ap and sp_ap.n else 0.0),
        "as_p1_n": sp_bp.n if sp_bp else 0,
        "as_p1_winrate": (sp_bp.p1_wins / sp_bp.n if sp_bp and sp_bp.n else 0.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8,
                        help="number of seeds (each plays 2 games — both seat orderings)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers for game runs")
    parser.add_argument("--out-dir", default=str(REPO / "audit" / "tournaments"),
                        help="directory for the JSON snapshot")
    parser.add_argument("--analytical", default=ANALYTICAL_PATH,
                        help="path to the candidate agent (default: analytical)")
    parser.add_argument("--baseline", default=TRAJECTORY_PATH,
                        help="path to the anchor agent (default: trajectory baseline)")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="Wilson-lo threshold for STOP gate (default 0.5)")
    args = parser.parse_args(argv)

    seeds = list(SEEDS_64[: int(args.n)])
    agents = {"analytical": args.analytical, "trajectory": args.baseline}

    print(f"=== analytical_ab.py ===")
    print(f"analytical agent: {args.analytical}")
    print(f"trajectory agent: {args.baseline}")
    print(f"seeds: {len(seeds)} ({seeds[:3]}…) | workers: {args.workers}")
    print()

    t0 = time.time()
    result = tournament.run_tournament(
        agents,
        seeds=seeds,
        include_self_play=False,
        workers=int(args.workers),
        progress=False,
    )
    elapsed = time.time() - t0

    summary = _summarise(result, "analytical", "trajectory")
    summary["seeds"] = seeds
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["n_seats_per_seed"] = 2  # P0 + P1
    summary["games_per_seed"] = 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"analytical-ab-{utc}.json"
    with out_path.open("w") as f:
        json.dump({"summary": summary,
                   "matrix": {
                       a: {b: {"p0_wins": sp.p0_wins, "p1_wins": sp.p1_wins,
                               "draws": sp.draws, "n": sp.n,
                               "p0_winrate": (sp.p0_wins / sp.n if sp.n else 0.0)}
                           for b, sp in row.items()}
                       for a, row in result.matrix.items()
                   }}, f, indent=2)

    print(f"=== result ===")
    print(f"games: {summary['n']} ({summary['games_per_seed']} per seed × {len(seeds)} seeds)")
    print(f"analytical wins:   {summary['wins']:>3} ({100*summary['winrate']:.1f}%)")
    print(f"analytical losses: {summary['losses']:>3}")
    print(f"draws:             {summary['draws']:>3}")
    print(f"Wilson 95% CI:     [{summary['wilson_lo']:.3f}, {summary['wilson_hi']:.3f}]")
    print(f"as P0: {summary['as_p0_n']} games, {100*summary['as_p0_winrate']:.1f}% winrate")
    print(f"as P1: {summary['as_p1_n']} games, {100*summary['as_p1_winrate']:.1f}% winrate")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"snapshot: {out_path}")
    print()

    gate_passes = summary["wilson_lo"] >= float(args.gate_threshold)
    print(f"STOP gate (Wlo ≥ {args.gate_threshold}): {'PASS' if gate_passes else 'FAIL'}")
    if not gate_passes:
        # n=8 has wide CI; treat ≥10/16 as "directional positive."
        directional = summary["wins"] >= max(1, summary["n"] // 2 + 2)
        print(f"  directional-positive (≥{max(1, summary['n']//2 + 2)}/{summary['n']} wins): "
              f"{'YES' if directional else 'NO'} ({summary['wins']}/{summary['n']})")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    sys.exit(main())
