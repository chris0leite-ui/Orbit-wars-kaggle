"""Single-game introspection of the differential chooser.

Wraps `agents.baseline.chooser_differential.choose_differential` so
every turn's candidate scores and emitted moves are logged. Plays
one game (differential as P0, opp as P1) and prints a turn-by-turn
trace.

Usage:
    python scripts/differential_introspect.py [--seed 42] [--opp submissions/baseline.py]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Force differential before importing anything that reads the env default.
os.environ["BASELINE_CHOOSER"] = "differential"

import agents.baseline.chooser_differential as cd  # noqa: E402
from agents.baseline.main import agent as differential_agent  # noqa: E402
from kaggle_environments import make  # noqa: E402


TURNS: list[dict] = []


def _wrap_choose_differential():
    original = cd.choose_differential

    def wrapped(snap_base, prerank, baseline_favors,
                me, num_seats, wallclock_ms,
                min_horizon, max_horizon, gamma,
                world, model):
        # Compute scores per candidate (use the public score function).
        scored = []
        for c in prerank:
            cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
            try:
                delta = cd.score_candidate_differential(
                    c, world, model, int(me), int(num_seats), float(gamma),
                )
            except Exception:
                delta = float("nan")
            scored.append((
                delta, int(src.id), int(tgt.id),
                int(ships), int(wait_N), int(eta),
            ))
        scored.sort(key=lambda x: -x[0] if x[0] == x[0] else float("inf"))

        moves = original(
            snap_base, prerank, baseline_favors,
            me, num_seats, wallclock_ms,
            min_horizon, max_horizon, gamma,
            world, model,
        )

        TURNS.append({
            "step": int(getattr(world, "step", 0) or 0),
            "n_candidates": len(prerank),
            "scored": scored[:10],  # keep top-10 only for log size
            "n_emit": len(moves),
            "emit_srcs": [int(m[0]) for m in moves],
            "n_positive_delta": sum(
                1 for s in scored if s[0] > 0.0
            ),
        })
        return moves

    cd.choose_differential = wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", default="submissions/baseline.py")
    args = ap.parse_args()

    _wrap_choose_differential()

    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=True)
    print(f"== differential introspect: P0 vs {args.opp} (P1), seed={args.seed} ==")
    result = env.run([differential_agent, args.opp])
    final = result[-1]
    outcome = [(s.observation.player, s.reward, s.status) for s in final]
    print(f"outcome: {outcome}")
    print(f"turns recorded: {len(TURNS)}")
    print()

    print(f"{'step':>4}  {'cands':>5}  {'+Δ':>3}  {'emit':>4}  top-3 (Δ, src→tgt, ships, wait_N, eta)")
    print("-" * 90)
    for t in TURNS:
        top = t["scored"][:3]
        top_str = " | ".join(
            f"({d:+6.1f}, {s}→{tg}, {sh}sh, w{wn}, eta{e})"
            for d, s, tg, sh, wn, e in top
        )
        print(
            f"{t['step']:>4}  {t['n_candidates']:>5}  "
            f"{t['n_positive_delta']:>3}  {t['n_emit']:>4}  {top_str}"
        )

    # Aggregates.
    n_turns = max(1, len(TURNS))
    total_cands = sum(t["n_candidates"] for t in TURNS)
    total_pos = sum(t["n_positive_delta"] for t in TURNS)
    total_emit = sum(t["n_emit"] for t in TURNS)
    print("-" * 90)
    print(f"totals: cands={total_cands} positive-delta={total_pos} emits={total_emit}")
    print(f"per-turn avg: cands={total_cands/n_turns:.1f} "
          f"positive-delta={total_pos/n_turns:.1f} emits={total_emit/n_turns:.2f}")
    # How often emits == positive_delta candidates?
    if total_pos:
        print(f"emit/positive ratio: {total_emit/total_pos:.2f} (1.0 = emit everything positive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
