"""Phase C n=8 A/B: bundle (cands=5 default) vs v7_0 and vs agents/baseline.

8 seeds × 2 sides × 2 opponents = 32 games, run in parallel via the
existing scripts.tournament.run_tournament fork-pool. Reports per-pair
Wilson lower bound (Wlo). Submission gate per Phase C plan: Wlo > 0.40
on BOTH opponents triggers a submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.tournament as tournament


SEEDS = [42, 1, 7, 13, 31, 100, 17, 23]  # first 8 of SEEDS_64


def main():
    agents = {
        "bundle": str(REPO / "submissions" / "bundle.py"),
        "v7_0":   str(REPO / "submissions" / "v7_0_drop_one.py"),
        "baseline": str(REPO / "submissions" / "baseline.py"),
    }
    out_dir = REPO / "audit" / "tournaments"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = tournament.run_tournament(
        agents=agents,
        seeds=SEEDS,
        include_self_play=False,
        workers=4,
        out_dir=out_dir,
        progress=True,
    )

    print("\n========== Aggregate (both sides) ==========")
    for a in ("bundle",):
        for b in ("v7_0", "baseline"):
            # Combine: a-as-P0-vs-b + b-as-P0-vs-a (a's wins = a_P0_wins + b_P0_losses for a)
            p_a_p0 = result.matrix[a][b]
            p_b_p0 = result.matrix[b][a]
            a_wins_both = p_a_p0.p0_wins + p_b_p0.p1_wins
            n_both = p_a_p0.n + p_b_p0.n
            winrate = a_wins_both / n_both if n_both else 0.0
            # Wilson 95% CI
            import math
            z = 1.96
            phat = winrate
            denom = 1 + z*z/n_both
            center = (phat + z*z/(2*n_both)) / denom
            margin = (z/denom) * math.sqrt(phat*(1-phat)/n_both + z*z/(4*n_both*n_both))
            wlo = center - margin
            whi = center + margin
            print(f"  {a} vs {b}: {a_wins_both}/{n_both} = {winrate:.3f} "
                  f"Wlo={wlo:.3f} Whi={whi:.3f}")
            # Timing
            for stat in (p_a_p0, p_b_p0):
                print(f"    seat-pair {stat.p0_name}-vs-{stat.p1_name}: "
                      f"p0_p95={stat.p0_p95_turn_ms:.0f}ms "
                      f"p1_p95={stat.p1_p95_turn_ms:.0f}ms")


if __name__ == "__main__":
    main()
