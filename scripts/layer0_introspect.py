"""Single-game introspection of the Layer-0 layered chooser.

Wraps `agents.baseline.chooser_layered.layer0_classify` so every turn's
W1/W2/L1/L2 verdicts are logged. Plays one game (layered as P0, opp as
P1) and prints a turn-by-turn trace.

Usage:
    python scripts/layer0_introspect.py [--seed 42] [--opp random]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Force layered before importing anything that reads the env default.
os.environ["BASELINE_CHOOSER"] = "layered"
os.environ["BASELINE_INNER_CHOOSER"] = "trajectory"

import agents.baseline.chooser_layered as cl  # noqa: E402
from agents.baseline.main import agent as layered_agent  # noqa: E402
from agents.baseline.predicates import (  # noqa: E402
    l1_provably_wasted_launch,
    w1_provably_winning_capture,
    w2_provably_held_reinforce,
)
from agents.baseline.predicates import l2_dominance_prune  # noqa: E402
from kaggle_environments import make  # noqa: E402


# ---------------------------------------------------------------------------
# Per-turn statistics, captured by wrapping layer0_classify.
# ---------------------------------------------------------------------------


TURNS: list[dict] = []  # list of per-turn dicts


def _wrap_layer0_classify():
    """Replace `cl.layer0_classify` with a version that records counts."""
    original = cl.layer0_classify

    def wrapped(prerank, world, model, me, step, gamma):
        n_in = len(prerank)
        # Re-classify to count outcomes (cheap: predicates are O(planets×horizon)).
        # We CALL the real function so behaviour is unchanged; the recount is
        # for stats only.
        verdicts: dict[str, int] = Counter()
        per_cand: list[tuple] = []
        for c in prerank:
            cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
            v_l1 = l1_provably_wasted_launch(
                src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
            )
            if v_l1.kind == "discard":
                verdicts["L1"] += 1
                per_cand.append(("L1", int(src.id), int(tgt.id), int(ships), int(wait_N)))
                continue
            v_w1 = w1_provably_winning_capture(
                src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
                gamma=gamma,
            )
            if v_w1.kind == "commit":
                verdicts["W1"] += 1
                per_cand.append(("W1", int(src.id), int(tgt.id), int(ships), int(wait_N), v_w1.lower_bound))
                continue
            v_w2 = w2_provably_held_reinforce(
                src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
            )
            if v_w2.kind == "commit":
                verdicts["W2"] += 1
                per_cand.append(("W2", int(src.id), int(tgt.id), int(ships), int(wait_N)))
                continue
            verdicts["uncertain"] += 1

        # L2 prune impact on the (uncertain) residual.
        uncertain_residual = [
            c for c in prerank
            if (l1_provably_wasted_launch(c[1], c[2], int(c[3]), int(c[7]), int(c[5]),
                                          world, model, int(me)).kind != "discard"
                and w1_provably_winning_capture(c[1], c[2], int(c[3]), int(c[7]), int(c[5]),
                                                world, model, int(me), gamma=gamma).kind != "commit"
                and w2_provably_held_reinforce(c[1], c[2], int(c[3]), int(c[7]), int(c[5]),
                                               world, model, int(me)).kind != "commit")
        ]
        n_after_l2 = len(l2_dominance_prune(uncertain_residual))
        n_l2_pruned = len(uncertain_residual) - n_after_l2

        # Now call the REAL function (unmodified state path).
        commits, residual = original(prerank, world, model, me, step, gamma)

        TURNS.append({
            "step": int(getattr(world, "step", 0) or 0),
            "n_candidates": n_in,
            "n_W1": verdicts["W1"],
            "n_W2": verdicts["W2"],
            "n_L1": verdicts["L1"],
            "n_L2_pruned": n_l2_pruned,
            "n_uncertain": verdicts["uncertain"],
            "n_commits_emitted": len(commits),
            "n_residual_to_inner": len(residual),
            "per_cand": per_cand,
        })
        return commits, residual

    cl.layer0_classify = wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", default="random")
    args = ap.parse_args()

    _wrap_layer0_classify()

    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=True)
    print(f"== one-game introspection: layered (P0) vs {args.opp} (P1), seed={args.seed} ==")
    # Set the seed at env make time isn't directly supported by kaggle_environments
    # — but the env uses its internal seed; we get reproducibility via the
    # interpreter's deterministic path. For our purposes any seed is fine.
    result = env.run([layered_agent, args.opp])
    final = result[-1]
    outcome = [(s.observation.player, s.reward, s.status) for s in final]
    print(f"outcome: {outcome}")
    print(f"n_turns recorded for P0 layered: {len(TURNS)}")
    print()

    # Per-turn summary header.
    print(
        f"{'step':>4}  {'cands':>5}  {'W1':>3}  {'W2':>3}  {'L1':>3}  "
        f"{'L2p':>3}  {'unc':>4}  {'emit_L0':>7}  {'->inner':>7}"
    )
    print("-" * 60)
    for t in TURNS:
        print(
            f"{t['step']:>4}  {t['n_candidates']:>5}  "
            f"{t['n_W1']:>3}  {t['n_W2']:>3}  {t['n_L1']:>3}  "
            f"{t['n_L2_pruned']:>3}  {t['n_uncertain']:>4}  "
            f"{t['n_commits_emitted']:>7}  {t['n_residual_to_inner']:>7}"
        )

    # Aggregates.
    total_cands = sum(t["n_candidates"] for t in TURNS)
    total_w1 = sum(t["n_W1"] for t in TURNS)
    total_w2 = sum(t["n_W2"] for t in TURNS)
    total_l1 = sum(t["n_L1"] for t in TURNS)
    total_l2 = sum(t["n_L2_pruned"] for t in TURNS)
    total_unc = sum(t["n_uncertain"] for t in TURNS)
    total_emit_l0 = sum(t["n_commits_emitted"] for t in TURNS)
    n_turns = max(1, len(TURNS))
    print("-" * 60)
    print(f"totals: cands={total_cands} W1={total_w1} W2={total_w2} "
          f"L1={total_l1} L2p={total_l2} uncertain={total_unc} "
          f"emitted_L0={total_emit_l0}")
    print(f"per-turn avg: cands={total_cands/n_turns:.1f} "
          f"W1={total_w1/n_turns:.2f} W2={total_w2/n_turns:.2f} "
          f"L1={total_l1/n_turns:.2f} L2p={total_l2/n_turns:.2f} "
          f"emit_L0={total_emit_l0/n_turns:.2f}")

    # Sample a few turns where W1/W2 fired, for narrative detail.
    interesting = [t for t in TURNS if t["n_W1"] > 0 or t["n_W2"] > 0]
    print()
    print(f"== first 3 turns with W1 or W2 commits ==")
    for t in interesting[:3]:
        print(f"\nstep {t['step']}: cands={t['n_candidates']} "
              f"W1={t['n_W1']} W2={t['n_W2']} L1={t['n_L1']}")
        for entry in t["per_cand"]:
            tag = entry[0]
            if tag in ("W1",):
                _tag, sid, tid, sh, wn, lb = entry
                print(f"   {tag}: src={sid} -> tgt={tid} ships={sh} wait_N={wn} "
                      f"lower_bound={lb:.2f}")
            elif tag in ("W2", "L1"):
                _tag, sid, tid, sh, wn = entry
                print(f"   {tag}: src={sid} -> tgt={tid} ships={sh} wait_N={wn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
