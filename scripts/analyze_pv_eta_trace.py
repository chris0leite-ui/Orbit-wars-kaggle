"""Analyze the JSON dump from trace_pv_eta_scoring.py.

Questions the trace data should answer:

  Q1. How often does PV_ETA flip a candidate across the Δ=0 emission
      threshold (positive→non-positive, or vice versa)? These are
      decision-altering flips — the chooser only emits Δ>0.

  Q2. What fraction of TOP-1 winners are wait_N>0 patience candidates
      vs immediate launches? Patience reserves src+tgt + emits NOTHING
      (chooser.py:201-203), so a wait_N>0 winner = a silent turn for
      that src.

  Q3. What's the (wait_N, eta) distribution of winners? If long-eta
      or wait_N>0 candidates dominate winners, PV_ETA is doing real
      work suppressing them; if winners are short-eta wait_N=0, the
      leaf already does the work and PV_ETA is mostly cosmetic.

  Q4. Among the TOP-1 winner per turn, how often does the post-PV
      identity differ from the pre-PV identity? That's the only way
      PV_ETA can change emitted moves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="audit/pv_eta_trace.json")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    captured = data["captured"]

    solo = [c for c in captured if c["kind"] == "solo"
            and c["delta_post"] != float("-inf")
            and c["delta_post"] != -float("inf")
            and c.get("status") not in ("sun", "oob", "timeout", "path_blocked",
                                         "comet_collision", "comet_expired")]

    joint = [c for c in captured if c["kind"] == "joint"]

    print(f"=== INPUT ===")
    print(f"  total captured: {len(captured)}")
    print(f"  solo (scored, not admissibility-killed): {len(solo)}")
    print(f"  joint: {len(joint)}")
    print()

    # ---- Q1: sign flips across Δ=0 ----
    pos_to_nonpos = 0  # would have been emitted (pre>0), suppressed by PV (post<=0)
    nonpos_to_pos = 0  # was non-emit (pre<=0), promoted by PV (post>0) — impossible: PV multiplies by 0<γ^k<1
    for c in solo:
        pre, post = float(c["delta_pre"]), float(c["delta_post"])
        if pre > 0 and post <= 0:
            pos_to_nonpos += 1
        if pre <= 0 and post > 0:
            nonpos_to_pos += 1
    print("=== Q1: SIGN FLIPS ACROSS Δ=0 EMISSION THRESHOLD ===")
    print(f"  Δpre>0  →  Δpost<=0  (PV killed an emit): {pos_to_nonpos} / {len(solo)}")
    print(f"  Δpre<=0 →  Δpost>0   (impossible; PV is γ^k<=1): {nonpos_to_pos}")
    print()

    # ---- group by turn, single-launch only ----
    by_turn: dict[int, list] = {}
    for c in solo:
        by_turn.setdefault(c["turn"], []).append(c)

    # ---- Q4: top-1 identity flip across turns ----
    top1_flipped = 0
    turns_with_pos_emit = 0
    top1_post_wait_n_pos = 0
    top1_pre_wait_n_pos = 0
    top1_winners_eta = Counter()      # bucketed eta of post-PV winner
    top1_pre_winners_eta = Counter()  # bucketed eta of pre-PV winner

    def bucket(eta: int) -> str:
        if eta == 0:  return "0"
        if eta <= 5:  return "1-5"
        if eta <= 10: return "6-10"
        if eta <= 20: return "11-20"
        if eta <= 30: return "21-30"
        return ">30"

    for t in sorted(by_turn.keys()):
        cands = [c for c in by_turn[t] if c["delta_post"] > 0]
        cands_pre = [c for c in by_turn[t] if c["delta_pre"] > 0]
        if not cands and not cands_pre:
            continue
        turns_with_pos_emit += 1
        post_top = max(cands, key=lambda c: c["delta_post"]) if cands else None
        pre_top = max(cands_pre, key=lambda c: c["delta_pre"]) if cands_pre else None

        if post_top and pre_top:
            key_post = (post_top["src"], post_top["tgt"], post_top["wait_N"])
            key_pre = (pre_top["src"], pre_top["tgt"], pre_top["wait_N"])
            if key_post != key_pre:
                top1_flipped += 1
        if post_top and int(post_top["wait_N"]) > 0:
            top1_post_wait_n_pos += 1
        if pre_top and int(pre_top["wait_N"]) > 0:
            top1_pre_wait_n_pos += 1
        if post_top:
            top1_winners_eta[bucket(int(post_top["eta"]))] += 1
        if pre_top:
            top1_pre_winners_eta[bucket(int(pre_top["eta"]))] += 1

    print("=== Q4: TOP-1 IDENTITY FLIPS (the only way PV alters emitted moves) ===")
    print(f"  turns with at least one Δ>0 candidate: {turns_with_pos_emit}")
    print(f"  turns where POST-PV top-1 ≠ PRE-PV top-1: {top1_flipped} "
          f"({100*top1_flipped/max(1,turns_with_pos_emit):.1f}%)")
    print()

    print("=== Q2/Q3: TOP-1 WINNER DISTRIBUTION ===")
    print(f"  POST-PV top-1 is wait_N>0 (PATIENCE — silent for that src):")
    print(f"    {top1_post_wait_n_pos} / {turns_with_pos_emit} "
          f"({100*top1_post_wait_n_pos/max(1,turns_with_pos_emit):.1f}%)")
    print(f"  PRE-PV  top-1 is wait_N>0:")
    print(f"    {top1_pre_wait_n_pos} / {turns_with_pos_emit} "
          f"({100*top1_pre_wait_n_pos/max(1,turns_with_pos_emit):.1f}%)")
    print()
    print(f"  POST-PV top-1 eta distribution: {dict(top1_winners_eta)}")
    print(f"  PRE-PV  top-1 eta distribution: {dict(top1_pre_winners_eta)}")
    print()

    # ---- Q5: examples of the largest PV_ETA-induced rank changes ----
    # For each turn, find the max |delta_pre - delta_post| in top-1 of either side.
    biggest_swings = []
    for t in sorted(by_turn.keys()):
        cands = by_turn[t]
        for c in cands:
            swing_abs = abs(float(c["delta_pre"]) - float(c["delta_post"]))
            swing_rel = (float(c["delta_post"]) / float(c["delta_pre"])
                         if abs(float(c["delta_pre"])) > 1e-9 else 1.0)
            biggest_swings.append((swing_abs, swing_rel, t, c))
    biggest_swings.sort(key=lambda x: -x[0])
    print("=== Q5: BIGGEST PV_ETA DISCOUNT SWINGS (absolute) ===")
    print("  (these are the candidates PV_ETA hits hardest)")
    for swing_abs, swing_rel, t, c in biggest_swings[:10]:
        print(f"    turn {t:3d}: src={c['src']:2d}→tgt={c['tgt']:2d} "
              f"ships={c['ships']:3d} wait={c['wait_N']:2d} eta={c['eta']:3d}  "
              f"Δpre={c['delta_pre']:+8.2f}  Δpost={c['delta_post']:+8.2f}  "
              f"mult=γ^{c['wait_N']+c['eta']}={c['mult']:.3f}")
    print()

    # ---- Q6: PATIENCE candidate prevalence in scoring vs emission ----
    n_wait_n_pos_scored = sum(1 for c in solo if int(c["wait_N"]) > 0)
    n_wait_n_pos_positive_pre = sum(1 for c in solo
                                     if int(c["wait_N"]) > 0 and c["delta_pre"] > 0)
    n_wait_n_pos_positive_post = sum(1 for c in solo
                                      if int(c["wait_N"]) > 0 and c["delta_post"] > 0)
    print("=== Q6: PATIENCE (wait_N>0) CANDIDATE POPULATION ===")
    print(f"  scored: {n_wait_n_pos_scored} / {len(solo)} "
          f"({100*n_wait_n_pos_scored/max(1,len(solo)):.1f}%)")
    print(f"  of those, Δpre > 0:  {n_wait_n_pos_positive_pre} "
          f"({100*n_wait_n_pos_positive_pre/max(1,n_wait_n_pos_scored):.1f}%)")
    print(f"  of those, Δpost > 0: {n_wait_n_pos_positive_post} "
          f"({100*n_wait_n_pos_positive_post/max(1,n_wait_n_pos_scored):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
