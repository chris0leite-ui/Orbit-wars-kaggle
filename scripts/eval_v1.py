"""Evaluate agent v1 (orbitfix) vs the comp-shipped baseline.

Plan/Step 3 gate (CLAUDE.md::flickering-tinkering-horizon.md):
- v1-vs-baseline ≥ 60% winrate over 20 seeds × both sides;
- v1-vs-v1 P0/P1 winrate split within ±15% of 50/50 (closes ISSUES.md::A.6).

Persists JSON to audit/tournaments/<utc>.json via the existing fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Import the tournament module under its top-level name so its dataclasses
# resolve correctly.
spec = importlib.util.spec_from_file_location("tournament", REPO / "scripts" / "tournament.py")
tournament = importlib.util.module_from_spec(spec)
sys.modules["tournament"] = tournament
spec.loader.exec_module(tournament)


SEEDS_20 = [42, 1, 7, 13, 31, 100, 17, 23, 53, 71, 91, 113, 137, 149, 167, 181, 199, 211, 233, 257]

# 128-seed geometry-stratified panel (built by scripts/build_seed_panel.py).
# Lazy import so a missing data/seed_panel_128.json doesn't break SEEDS_20 callers.
try:
    from lib.seed_panel import SEED_PANEL_128 as SEEDS_128  # noqa: F401
except Exception:  # pragma: no cover
    SEEDS_128 = None


def main() -> int:
    baseline = str(REPO / "data" / "main.py")
    v1 = str(REPO / "agents" / "v1_orbitfix" / "main.py")

    out_dir = REPO / "audit" / "tournaments"
    result = tournament.run_tournament(
        agents={"baseline": baseline, "v1_orbitfix": v1},
        seeds=SEEDS_20,
        include_self_play=True,
        out_dir=out_dir,
        progress=True,
    )

    print()
    print("=== Step 3 gate readout ===")
    for row in ("v1_orbitfix", "baseline"):
        for col in ("baseline", "v1_orbitfix"):
            stat = result.matrix[row][col]
            print(
                f"{row} (P0) vs {col} (P1): {stat.p0_wins}/{stat.n} P0 wins "
                f"(Wilson 95% {stat.wilson_lo:.2f}..{stat.wilson_hi:.2f}); "
                f"p95 turn ms P0={stat.p0_p95_turn_ms:.1f} P1={stat.p1_p95_turn_ms:.1f}; "
                f"mean dShips P0-P1={stat.mean_ship_delta_p0_minus_p1:+.0f}"
            )

    # Aggregate v1 winrate (across both sides).
    v1_as_p0 = result.matrix["v1_orbitfix"]["baseline"]
    v1_as_p1_inverted = result.matrix["baseline"]["v1_orbitfix"]
    v1_total_wins = v1_as_p0.p0_wins + v1_as_p1_inverted.p1_wins
    v1_total_n = v1_as_p0.n + v1_as_p1_inverted.n
    print()
    print(f"v1 vs baseline aggregate: {v1_total_wins}/{v1_total_n} = "
          f"{v1_total_wins / v1_total_n:.1%} (gate: ≥60%)")

    # A.6 self-play split.
    sp = result.matrix["v1_orbitfix"]["v1_orbitfix"]
    print(f"v1 self-play P0/P1 split: {sp.p0_wins}/{sp.n} P0, {sp.p1_wins}/{sp.n} P1, "
          f"{sp.draws} draws (gate: |P0-P1| ≤ ±15% of n)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
