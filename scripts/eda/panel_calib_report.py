"""Quick report on a strategy_panel JSON snapshot.

Pulls Wilson-95 lower bounds on each `A vs B` cell so we can decide
which candidate beats the hardened panel under uncertainty. The
panel-shipped formatter prints point estimates; this script adds
Wilson-lo + matchup-by-matchup confidence intervals.

Usage:
    python -m scripts.eda.panel_calib_report audit/tournaments/<utc>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def wilson_lo(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - rad) / den)


def wilson_hi(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = wins / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + rad) / den)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to tournament JSON")
    args = p.parse_args(argv)
    data = json.load(open(args.path))

    # Schema: data is {(A_name, B_name): {wins_a, wins_b, games, ...}, ...}
    # The on-disk format from scripts/tournament.run_tournament:
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if pairs is None and isinstance(data, dict):
        # Some versions store at top level; flatten.
        pairs = data

    # Build agent list
    names: set[str] = set()
    for key, cell in pairs.items():
        if isinstance(cell, dict):
            a = cell.get("a") or cell.get("agent_a")
            b = cell.get("b") or cell.get("agent_b")
            if a and b:
                names.add(a); names.add(b)
    names = sorted(names)

    # Aggregate both-sides
    agg = {a: {b: [0, 0] for b in names} for a in names}  # wins, games
    for key, cell in pairs.items():
        if not isinstance(cell, dict):
            continue
        a = cell.get("a") or cell.get("agent_a")
        b = cell.get("b") or cell.get("agent_b")
        if not a or not b:
            continue
        wa = cell.get("wins_a", 0)
        wb = cell.get("wins_b", 0)
        n  = cell.get("games", wa + wb)
        agg[a][b][0] += wa; agg[a][b][1] += n
        agg[b][a][0] += wb; agg[b][a][1] += n

    # Table with Wilson-lo
    w = max(len(n) for n in names) + 2
    print(f"\n{'A vs B':<{w}}", end='')
    for b in names:
        print(f"{b[:12]:>14}", end='')
    print()
    print('-' * (w + 14 * len(names)))
    for a in names:
        print(f"{a:<{w}}", end='')
        for b in names:
            wins, n = agg[a][b]
            if a == b or n == 0:
                print(f"{'  --   ':>14}", end='')
            else:
                p_hat = wins / n
                lo = wilson_lo(wins, n)
                print(f"{p_hat:>5.1%} [{lo:>4.0%}] ", end='')
        print()

    # Panel calibration row: mean win-rate vs panel + min Wilson-lo
    print('\n=== calibration: mean vs panel (excl self) + worst-cell Wilson-lo ===')
    for a in names:
        wrs = []
        lo_min = 1.0
        for b in names:
            if a == b:
                continue
            wins, n = agg[a][b]
            if n == 0:
                continue
            wrs.append(wins / n)
            lo_min = min(lo_min, wilson_lo(wins, n))
        mean_wr = sum(wrs) / len(wrs) if wrs else float('nan')
        print(f'  {a:<{w}} mean={mean_wr:>5.1%}  worst-cell Wilson-lo={lo_min:>5.1%}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
