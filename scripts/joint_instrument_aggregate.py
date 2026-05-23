"""Refined aggregation over the joint-instrument JSONL.

Splits the "uncrackable" count by target owner (own / enemy / neutral)
and by defender size, so we can tell whether the n-source-bundle
candidates are real capture opportunities vs. own-target reinforce
attempts the chooser correctly rejected.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--me", type=int, default=0,
        help="The focal seat id used in the recorded games (default 0).",
    )
    args = parser.parse_args()

    rows = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Total rows: {len(rows)}")
    print(f"Games seen (rough — unique turn=0 rows): "
          f"{sum(1 for r in rows if r['turn'] == 0)}")

    # Filter conditions
    def is_uncrackable(r):
        return r["n_pairs_positive"] == 0 and not r["any_solo_winner"]

    def is_own(r):
        return int(r["tgt_owner"]) == args.me

    def is_enemy(r):
        o = int(r["tgt_owner"])
        return o != -1 and o != args.me

    def is_neutral(r):
        return int(r["tgt_owner"]) == -1

    # By owner
    print()
    print("--- All target observations by owner ---")
    for label, pred in [("own", is_own), ("enemy", is_enemy),
                        ("neutral", is_neutral)]:
        n = sum(1 for r in rows if pred(r))
        n_unc = sum(1 for r in rows if pred(r) and is_uncrackable(r))
        n_unc_3plus = sum(
            1 for r in rows
            if pred(r) and is_uncrackable(r) and r["n_cands"] >= 3
        )
        n_unc_4plus = sum(
            1 for r in rows
            if pred(r) and is_uncrackable(r) and r["n_cands"] >= 4
        )
        print(f"  {label}: total={n}  uncrackable={n_unc} "
              f"({100.0 * n_unc / max(1, n):.1f}%)  "
              f"unc&3+cands={n_unc_3plus}  unc&4+cands={n_unc_4plus}")

    # Real capture opportunities = enemy or neutral, uncrackable, ≥3 cands
    enemy_neutral_targets = [
        r for r in rows
        if (is_enemy(r) or is_neutral(r))
    ]
    enemy_neutral_uncrackable_3plus = [
        r for r in enemy_neutral_targets
        if is_uncrackable(r) and r["n_cands"] >= 3
    ]
    print()
    print("--- Real n-source-extension candidates "
          "(enemy or neutral, uncrackable, ≥3 cands) ---")
    print(f"  count: {len(enemy_neutral_uncrackable_3plus)}")
    print(f"  per total-row-observations: "
          f"{100.0 * len(enemy_neutral_uncrackable_3plus) / max(1, len(rows)):.2f}%")
    if enemy_neutral_uncrackable_3plus:
        garrisons = [r["tgt_ships"]
                     for r in enemy_neutral_uncrackable_3plus]
        prods = [r["tgt_production"]
                 for r in enemy_neutral_uncrackable_3plus]
        n_cands = [r["n_cands"]
                   for r in enemy_neutral_uncrackable_3plus]
        n_pairs = [r["n_pairs_attempted"]
                   for r in enemy_neutral_uncrackable_3plus]
        print(f"  garrison median/p90/max: "
              f"{statistics.median(garrisons):.1f} / "
              f"{sorted(garrisons)[int(0.9*len(garrisons))]:.1f} / "
              f"{max(garrisons):.1f}")
        print(f"  production median/p90/max: "
              f"{statistics.median(prods):.2f} / "
              f"{sorted(prods)[int(0.9*len(prods))]:.2f} / "
              f"{max(prods):.2f}")
        print(f"  n_cands median/p90/max: "
              f"{statistics.median(n_cands):.0f} / "
              f"{sorted(n_cands)[int(0.9*len(n_cands))]:.0f} / "
              f"{max(n_cands)}")
        print(f"  n_pairs_attempted median/p90/max: "
              f"{statistics.median(n_pairs):.0f} / "
              f"{sorted(n_pairs)[int(0.9*len(n_pairs))]:.0f} / "
              f"{max(n_pairs)}")

    # Of those candidates, what fraction had high-production targets
    # (production >= 2)? These are the most valuable to crack.
    n_high_prod = sum(
        1 for r in enemy_neutral_uncrackable_3plus
        if r["tgt_production"] >= 2.0
    )
    n_big_garrison = sum(
        1 for r in enemy_neutral_uncrackable_3plus
        if r["tgt_ships"] >= 30.0
    )
    print()
    print("--- Cracking-difficulty subset ---")
    print(f"  candidates with target_production ≥ 2: "
          f"{n_high_prod} ({100.0 * n_high_prod / max(1, len(enemy_neutral_uncrackable_3plus)):.1f}% of cands)")
    print(f"  candidates with garrison ≥ 30: "
          f"{n_big_garrison} ({100.0 * n_big_garrison / max(1, len(enemy_neutral_uncrackable_3plus)):.1f}% of cands)")

    # Per-turn rate (rough): how many "real candidates" per turn?
    # Use turn buckets across all games.
    by_turn = {}
    for r in enemy_neutral_uncrackable_3plus:
        by_turn[r["turn"]] = by_turn.get(r["turn"], 0) + 1
    # Estimate turns observed: number of distinct (game,turn) pairs is
    # hard to reconstruct without game id. Use max-turn × n_games as a
    # rough estimate.
    n_games_est = sum(1 for r in rows if r["turn"] == 0)
    # Actually rows are not 1 per game-per-turn; many rows can share a
    # turn (one per target). Better: count unique turns by their
    # frequency.
    print()
    print(f"--- Per-turn distribution (rough; rows are per-target) ---")
    if by_turn:
        rates = list(by_turn.values())
        print(f"  candidates per turn median/p90/max: "
              f"{statistics.median(rates):.0f} / "
              f"{sorted(rates)[int(0.9*len(rates))]:.0f} / "
              f"{max(rates)}")
        print(f"  total turns with at least one candidate: {len(by_turn)}")

    # Also: how many candidates per turn AVERAGED per game?
    print()
    print("--- Solo gating (existing pair-enum's restrictive gate) ---")
    n_solo_gated_total = sum(r["n_pairs_solo_gated"] for r in rows)
    n_pairs_attempted_total = sum(r["n_pairs_attempted"] for r in rows)
    print(f"  pair-enum: {n_pairs_attempted_total} attempts, "
          f"{n_solo_gated_total} skipped via solo_gate")
    print(f"  solo-gate rate: {100.0 * n_solo_gated_total / max(1, n_pairs_attempted_total + n_solo_gated_total):.1f}% of would-be-pairs")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
