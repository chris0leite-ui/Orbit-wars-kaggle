"""Post-process a BASELINE_VALIDATOR_TRACE jsonl from one game.

Answers three questions:
  1. How aggressive is the filter? (drop rate per turn, P distribution)
  2. Of dropped shots, how many targeted planets that focal *later captured*?
     (validator was wrong → kept shots succeeded without those drops)
  3. Of turns where validator dropped >=1 shot toward planet T while keeping
     others toward T, did focal capture T within K=20 turns?
     (multi-source coordination loss check)

Usage:
  python -m scripts.diag_validator_drops /tmp/drops.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def main(path: str) -> int:
    turns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns.append(json.loads(line))

    if not turns:
        print("empty trace")
        return 1

    focal_seat = turns[0]["focal_seat"]
    n_turns = len(turns)
    print(f"trace: {n_turns} turns, focal_seat={focal_seat}")
    print()

    # --- Aggregate counts ---------------------------------------------------
    n_emits = 0
    n_dropped = 0
    n_self_reinf = 0
    drop_probs = []
    kept_probs = []
    per_turn_drop_rate = []

    for t in turns:
        emits = t["emits"]
        if not emits:
            per_turn_drop_rate.append(0.0)
            continue
        n_emits += len(emits)
        td, ts = 0, 0
        for e in emits:
            if e["self_reinf"]:
                n_self_reinf += 1
                continue
            if e["dropped"]:
                td += 1
                if e["p"] is not None:
                    drop_probs.append(e["p"])
            else:
                if e["p"] is not None:
                    kept_probs.append(e["p"])
        n_dropped += td
        non_self_reinf = sum(1 for e in emits if not e["self_reinf"])
        per_turn_drop_rate.append(td / non_self_reinf if non_self_reinf else 0.0)

    print("=== aggregate ===")
    print(f"  total emits   : {n_emits}")
    print(f"  self-reinforce: {n_self_reinf} (pass-through)")
    print(f"  scored        : {n_emits - n_self_reinf}")
    print(f"  dropped       : {n_dropped} ({100*n_dropped/max(1,n_emits-n_self_reinf):.1f}% of scored)")
    print()

    # Distribution of drop probabilities (where are we dropping?)
    if drop_probs:
        drop_probs.sort()
        p05 = drop_probs[int(0.05*len(drop_probs))]
        p50 = drop_probs[int(0.50*len(drop_probs))]
        p95 = drop_probs[int(0.95*len(drop_probs))]
        print("=== dropped shot P(success) distribution ===")
        print(f"  n={len(drop_probs)}  p05={p05:.3f}  p50={p50:.3f}  p95={p95:.3f}  max={max(drop_probs):.3f}")
    if kept_probs:
        kept_probs.sort()
        p05 = kept_probs[int(0.05*len(kept_probs))]
        p50 = kept_probs[int(0.50*len(kept_probs))]
        p95 = kept_probs[int(0.95*len(kept_probs))]
        print("=== kept (non-self-reinf) shot P(success) distribution ===")
        print(f"  n={len(kept_probs)}  p05={p05:.3f}  p50={p50:.3f}  p95={p95:.3f}  max={max(kept_probs):.3f}")
    print()

    # Per-turn drop rate distribution
    rates = sorted(per_turn_drop_rate)
    if rates:
        print("=== per-turn drop-rate distribution ===")
        print(f"  p05={rates[int(0.05*len(rates))]*100:.0f}%  "
              f"p50={rates[int(0.50*len(rates))]*100:.0f}%  "
              f"p95={rates[int(0.95*len(rates))]*100:.0f}%  "
              f"max={max(rates)*100:.0f}%")
    print()

    # --- Drop-but-captured analysis ----------------------------------------
    # For each turn, for each dropped shot targeting planet T, check whether
    # T is owned by focal in any of the next K turns. K=20 covers in-flight
    # fleet arrival even from far sources at slow speeds.
    K = 20
    n_drops_total = 0
    n_drops_target_captured_later = 0
    n_drops_target_owned_now = 0  # weird case

    for ti, t in enumerate(turns):
        for e in t["emits"]:
            if e["self_reinf"] or not e["dropped"]:
                continue
            n_drops_total += 1
            tgt = e["tgt"]
            # Check next K turns
            captured = False
            for tj in range(ti + 1, min(ti + 1 + K, n_turns)):
                owners = turns[tj]["owners"]
                if 0 <= tgt < len(owners) and owners[tgt] == focal_seat:
                    captured = True
                    break
            if captured:
                n_drops_target_captured_later += 1
            # Owned now (sanity)
            if 0 <= tgt < len(t["owners"]) and t["owners"][tgt] == focal_seat:
                n_drops_target_owned_now += 1

    print(f"=== drop-but-captured-within-{K}-turns ===")
    if n_drops_total:
        rate = n_drops_target_captured_later / n_drops_total
        print(f"  total non-self-reinf drops : {n_drops_total}")
        print(f"  target captured by focal   : {n_drops_target_captured_later} ({100*rate:.1f}%)")
        print(f"  target already focal (now) : {n_drops_target_owned_now}")
        print(f"  interpretation: dropped shots whose target focal still captured")
        print(f"                  are 'wasted drops' — either the drop")
        print(f"                  was correct (target captured by other shots)")
        print(f"                  OR the drop disrupted a successful plan.")
    print()

    # --- Multi-source coordination loss check -------------------------------
    # Per turn, group emits by target. If validator drops some shots toward T
    # but keeps others, count whether T is captured within K turns. Compare
    # vs all-kept-toward-T case (no drops).
    K2 = 10
    mixed_drop_captured = 0
    mixed_drop_total = 0
    all_kept_captured = 0
    all_kept_total = 0
    all_dropped_captured = 0
    all_dropped_total = 0

    for ti, t in enumerate(turns):
        by_target = defaultdict(list)
        for e in t["emits"]:
            if e["self_reinf"]:
                continue
            by_target[e["tgt"]].append(e)
        for tgt, group in by_target.items():
            drops = sum(1 for e in group if e["dropped"])
            kept = len(group) - drops
            # Did focal capture this target in next K2 turns?
            captured = False
            for tj in range(ti + 1, min(ti + 1 + K2, n_turns)):
                owners = turns[tj]["owners"]
                if 0 <= tgt < len(owners) and owners[tgt] == focal_seat:
                    captured = True
                    break
            if drops > 0 and kept > 0:
                mixed_drop_total += 1
                if captured:
                    mixed_drop_captured += 1
            elif drops == 0 and kept > 0:
                all_kept_total += 1
                if captured:
                    all_kept_captured += 1
            elif drops > 0 and kept == 0:
                all_dropped_total += 1
                if captured:
                    all_dropped_captured += 1

    print(f"=== per-target attempt success (next {K2} turns) ===")
    print(f"  all-kept       : {all_kept_captured}/{all_kept_total}  "
          f"({100*all_kept_captured/max(1,all_kept_total):.1f}%)")
    print(f"  mixed-some-drop: {mixed_drop_captured}/{mixed_drop_total}  "
          f"({100*mixed_drop_captured/max(1,mixed_drop_total):.1f}%)")
    print(f"  all-dropped    : {all_dropped_captured}/{all_dropped_total}  "
          f"({100*all_dropped_captured/max(1,all_dropped_total):.1f}%)")
    print()
    print("  reading:")
    print("    all-kept  > mixed   → validator drops were partially fatal")
    print("                            (kept-some but lost the attempt)")
    print("    mixed   > all-kept  → validator surfaces strong-attempt subset")
    print("    all-dropped low rate → validator correctly killed bad attempts")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/drops.jsonl"))
