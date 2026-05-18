"""Validate the 128-seed panel: run baseline-vs-baseline tournament and
report per-archetype winrate variance.

Success criteria:
- All 128 games complete (no timeouts / errors).
- Per-archetype winrate variance is non-trivial (≥1 archetype with
  winrate outside [0.40, 0.60]). If every cell is ~50/50 the panel
  isn't exposing geometry-conditional differences and the axes need
  re-thinking.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lib.seed_panel import ARCHETYPE_OF_SEED, SEED_PANEL_128

# Import tournament as a top-level module (matches eval_v1.py's pattern).
spec = importlib.util.spec_from_file_location("tournament", REPO / "scripts" / "tournament.py")
tournament = importlib.util.module_from_spec(spec)
sys.modules["tournament"] = tournament
spec.loader.exec_module(tournament)


def main() -> int:
    baseline = str(REPO / "data" / "main.py")
    out_dir = REPO / "audit" / "tournaments"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"running baseline vs baseline on {len(SEED_PANEL_128)} seeds ...")
    result = tournament.run_tournament(
        agents={"baseline": baseline},
        seeds=SEED_PANEL_128,
        include_self_play=True,
        out_dir=out_dir,
        progress=True,
    )

    pair = result.matrix["baseline"]["baseline"]
    print(f"\ntotal games: {pair.n}  p0 wins: {pair.p0_wins}  p1 wins: {pair.p1_wins}  draws: {pair.draws}")
    assert pair.n == len(SEED_PANEL_128), f"only {pair.n}/{len(SEED_PANEL_128)} games completed"

    # Per-archetype winrate
    per_arch: dict[str, list[int]] = defaultdict(list)
    for game in pair.games:
        arch = ARCHETYPE_OF_SEED.get(game.seed, "UNKNOWN")
        # rewards[0] == 1 means p0 won; -1 means lost; 0 draw.
        p0_win = 1 if game.rewards[0] == 1 else 0
        per_arch[arch].append(p0_win)

    print("\n=== per-archetype P0 winrate (baseline self-play) ===")
    winrates = []
    for arch in sorted(per_arch.keys()):
        wins = sum(per_arch[arch])
        n = len(per_arch[arch])
        wr = wins / n if n else 0
        winrates.append(wr)
        flag = " <-- extreme" if wr < 0.25 or wr > 0.75 else ""
        print(f"  {arch:55s}  {wins}/{n}  {wr:.0%}{flag}")

    n_extreme = sum(1 for wr in winrates if wr < 0.40 or wr > 0.60)
    print(f"\narchetypes with winrate outside [0.40, 0.60]: {n_extreme}/{len(winrates)}")
    print(f"winrate stdev across archetypes: {statistics.pstdev(winrates):.3f}")

    if n_extreme == 0:
        print("\nWARNING: no archetype shows non-trivial P0 winrate skew.")
        print("  This may mean the panel isn't exposing geometry-conditional differences,")
        print("  OR it may mean the baseline-vs-itself is too symmetric to see effects.")
        print("  Re-run with an asymmetric pair (baseline vs v1_orbitfix) to confirm.")
    else:
        print("\nOK: panel exposes geometry-conditional differences.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
