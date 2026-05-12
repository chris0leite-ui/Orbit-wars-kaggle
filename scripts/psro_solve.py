"""Solve mixed Nash equilibrium for a PSRO payoff matrix.

Input: JSON from `scripts/psro_tournament.py` (or similar).
Output: mixed-Nash probability distribution over the pool.

For a symmetric 2-player zero-sum game with antisymmetric payoff matrix
P (where P[i][j] = -P[j][i]), the row-player's Nash mixed strategy is
also a column-player's Nash strategy by symmetry. We solve via
`nashpy`'s support_enumeration for small (n ≤ 6) pools, or
`lemke_howson` for larger pools.

Usage:
  python -m scripts.psro_solve audit/tournaments/psro_payoff_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import nashpy as nash


def solve_nash(P: np.ndarray, method: str = "support_enumeration") -> dict:
    """Return Nash mixed strategy + value.

    For a symmetric zero-sum game, returns p* (row player's strategy).
    By symmetry of the payoff matrix, this is also the column player's.
    """
    game = nash.Game(P, -P.T)  # Zero-sum: row gets P, col gets -P^T

    if method == "support_enumeration":
        eqs = list(game.support_enumeration())
    elif method == "lemke_howson":
        eqs = [game.lemke_howson(initial_dropped_label=0)]
    else:
        raise ValueError(method)

    if not eqs:
        return {"error": "no Nash equilibrium found", "method": method}

    # Pick the first equilibrium; for symmetric zero-sum, all NE have same value.
    row_p, col_p = eqs[0]

    # Compute game value (expected payoff to row player at NE).
    value = float(row_p @ P @ col_p)

    return {
        "method": method,
        "row_strategy": row_p.tolist(),
        "col_strategy": col_p.tolist(),
        "game_value": value,
        "num_equilibria_found": len(eqs),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: psro_solve.py <matrix.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    P = np.array(data["P"], dtype=float)
    pool = data["policies"]

    print(f"Pool ({len(pool)}): {pool}")
    print(f"Payoff matrix (W/D/L-balance, antisymmetric):")
    print("           " + "  ".join(f"{p:>10}" for p in pool))
    for i, name in enumerate(pool):
        row = "  ".join(f"{P[i][j]:>+10.3f}" for j in range(len(pool)))
        print(f"  {name:>10}  {row}")
    print()

    # Try support enumeration first (cleaner for small games)
    try:
        result = solve_nash(P, method="support_enumeration")
    except Exception as e:
        print(f"support_enumeration failed: {e}; trying lemke_howson")
        result = solve_nash(P, method="lemke_howson")

    if "error" in result:
        print(f"NASH FAILED: {result['error']}")
        sys.exit(1)

    print(f"\nNash equilibrium ({result['method']}, "
          f"{result['num_equilibria_found']} found):")
    print(f"  game value: {result['game_value']:+.4f}")
    print(f"  row player mixed strategy:")
    for name, p in zip(pool, result["row_strategy"]):
        bar = "█" * int(p * 40)
        print(f"    {name:>12}  {p:.4f}  {bar}")

    # Save
    out = path.with_suffix(".nash.json")
    out.write_text(json.dumps({
        "pool": pool,
        "P": P.tolist(),
        "nash": result,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
