"""Step-0 diagnostic probe for Reframe A.

Gates the additive-ML-logit-in-chooser implementation. Run BEFORE any
chooser edits to add the ML term: confirms the Booster (a)
differentiates among pv_eta's surviving candidates and (b) carries
information independent of the chooser's `delta`.

Workflow:
  1. Start pv_eta self-play games (source baseline with
     BASELINE_PV_ETA=1 + bundle peak preamble) with
     `BASELINE_TRAJECTORY_TRACE=<path>` set so the chooser dumps
     `(delta, p_success)` per scored candidate to a JSONL file.
  2. Run this script on the JSONL to compute per-turn stats and
     decision gates.

Gate (all three required to proceed with Reframe A):
  (a) median per-turn σ(P_success) >= 0.05  — Booster discriminates
  (b) median per-turn |Spearman ρ(delta, logit P)| < 0.85  — ML carries
      info independent of delta
  (c) median per-turn median(P_success) in [0.2, 0.8] — most candidates
      in the Booster's discriminative range
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Returns 0.0 if degenerate."""
    if len(xs) < 3:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(xs)))
    ry = np.argsort(np.argsort(np.asarray(ys)))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load_trace(path: Path) -> list[dict]:
    """Flatten solo + joint records to per-candidate dicts with
    `{step, delta, p}` and a `kind` tag. Joints contribute the joint
    `delta` paired with each leg's `p` (sum-of-logits would be the
    correct aggregation but for probe purposes per-leg P is the
    discriminative-power question)."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "solo":
                if rec.get("p") is None:
                    continue
                out.append({
                    "step": int(rec["step"]),
                    "delta": float(rec["delta"]),
                    "p": float(rec["p"]),
                    "kind": "solo",
                })
            elif rec.get("kind") == "joint":
                d = float(rec["delta"])
                for leg in rec.get("legs", []):
                    if leg.get("p") is None:
                        continue
                    out.append({
                        "step": int(rec["step"]),
                        "delta": d,
                        "p": float(leg["p"]),
                        "kind": "joint",
                    })
    return out


def per_turn_stats(rows: list[dict]) -> dict:
    """Bucket by step, compute σ(delta), σ(P), σ(logit P), median P,
    Spearman ρ(delta, logit P) per turn. Aggregate across turns."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)

    sigma_delta_pt: list[float] = []
    sigma_p_pt: list[float] = []
    sigma_logit_pt: list[float] = []
    median_p_pt: list[float] = []
    rho_pt: list[float] = []
    n_cand_pt: list[int] = []

    for step, bucket in by_step.items():
        if len(bucket) < 2:
            continue
        deltas = [r["delta"] for r in bucket]
        ps = [r["p"] for r in bucket]
        logits = [_logit(p) for p in ps]
        sigma_delta_pt.append(float(np.std(deltas)))
        sigma_p_pt.append(float(np.std(ps)))
        sigma_logit_pt.append(float(np.std(logits)))
        median_p_pt.append(float(np.median(ps)))
        rho_pt.append(_spearman(deltas, logits))
        n_cand_pt.append(len(bucket))

    def q(xs: list[float], q_: float) -> float:
        if not xs:
            return float("nan")
        return float(np.quantile(np.asarray(xs), q_))

    return {
        "n_turns": len(sigma_delta_pt),
        "median_n_candidates_per_turn": q(n_cand_pt, 0.5) if n_cand_pt else 0,
        "median_sigma_delta": q(sigma_delta_pt, 0.5),
        "median_sigma_p": q(sigma_p_pt, 0.5),
        "median_sigma_logit_p": q(sigma_logit_pt, 0.5),
        "median_median_p": q(median_p_pt, 0.5),
        "median_abs_rho": q([abs(r) for r in rho_pt], 0.5),
        "p25_sigma_p": q(sigma_p_pt, 0.25),
        "p75_sigma_p": q(sigma_p_pt, 0.75),
        "p25_abs_rho": q([abs(r) for r in rho_pt], 0.25),
        "p75_abs_rho": q([abs(r) for r in rho_pt], 0.75),
    }


def p_histogram(rows: list[dict]) -> dict:
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    ps = np.asarray([r["p"] for r in rows])
    hist, _ = np.histogram(ps, bins=bins)
    total = max(1, int(hist.sum()))
    labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins) - 1)]
    return {
        "bins": labels,
        "counts": hist.tolist(),
        "fractions": [round(c / total, 4) for c in hist],
        "n_total": int(total),
        "frac_extreme": round(
            float((ps < 0.1).sum() + (ps > 0.9).sum()) / total, 4,
        ),
    }


def lambda_suggestions(stats: dict) -> list[float]:
    """λ such that ML term magnitude ≈ k × σ(delta) where k ∈ {0.1,
    0.3, 1.0}, given centered-logit form `λ * (logit(P) - logit(0.5))`.
    Use σ(logit P) as the spread of the logit term. Falls back to a
    fixed triple when stats are degenerate."""
    sd = stats.get("median_sigma_delta", 0.0)
    sl = stats.get("median_sigma_logit_p", 0.0)
    if sd <= 0 or sl <= 0:
        return [0.5, 1.5, 5.0]
    return [round(k * sd / sl, 3) for k in (0.1, 0.3, 1.0)]


def evaluate_gates(stats: dict) -> dict:
    sp = stats.get("median_sigma_p", 0.0)
    rho = stats.get("median_abs_rho", 1.0)
    mp = stats.get("median_median_p", 0.5)
    gates = {
        "sigma_p_>=_0.05": sp >= 0.05,
        "abs_rho_<_0.85": rho < 0.85,
        "median_p_in_[0.2, 0.8]": 0.2 <= mp <= 0.8,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def render_report(trace_path: Path, stats: dict, gates: dict,
                  hist: dict, lams: list[float]) -> str:
    lines = []
    lines.append("# Reframe A — Step-0 diagnostic probe report")
    lines.append("")
    lines.append(f"Trace file: `{trace_path}`")
    lines.append(f"Candidates analysed (with valid P): "
                 f"**{hist['n_total']}**")
    lines.append(f"Turns analysed: **{stats['n_turns']}**, "
                 f"median candidates/turn: "
                 f"{stats['median_n_candidates_per_turn']:.1f}")
    lines.append("")
    lines.append("## Per-turn statistics (medians across turns)")
    lines.append("")
    lines.append(f"- σ(delta): **{stats['median_sigma_delta']:.4f}**")
    lines.append(f"- σ(P_success): **{stats['median_sigma_p']:.4f}**  "
                 f"(p25 {stats['p25_sigma_p']:.4f}, "
                 f"p75 {stats['p75_sigma_p']:.4f})")
    lines.append(f"- σ(logit P): **{stats['median_sigma_logit_p']:.4f}**")
    lines.append(f"- median(P_success): "
                 f"**{stats['median_median_p']:.4f}**")
    lines.append(f"- |Spearman ρ(delta, logit P)|: "
                 f"**{stats['median_abs_rho']:.4f}**  "
                 f"(p25 {stats['p25_abs_rho']:.4f}, "
                 f"p75 {stats['p75_abs_rho']:.4f})")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for k, v in gates.items():
        if k == "all_pass":
            continue
        lines.append(f"- `{k}`: **{'PASS' if v else 'FAIL'}**")
    lines.append("")
    verdict = "PASS — proceed to Reframe A" if gates["all_pass"] \
        else "FAIL — abort Reframe A, pivot to Reframe B"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("## P_success distribution")
    lines.append("")
    lines.append("| bin | count | frac |")
    lines.append("|---|---:|---:|")
    for b, c, f in zip(hist["bins"], hist["counts"], hist["fractions"]):
        lines.append(f"| {b} | {c} | {f:.4f} |")
    lines.append("")
    lines.append(f"Tail (P<0.1 or P>0.9) fraction: "
                 f"{hist['frac_extreme']:.4f}")
    lines.append("")
    lines.append("## Suggested λ sweep")
    lines.append("")
    lines.append(f"Targeting ML-logit magnitudes "
                 f"{{0.1, 0.3, 1.0}} × σ(delta):")
    lines.append(f"- λ candidates: **{lams}**")
    lines.append("")
    lines.append("Sweep these via `BASELINE_ML_LAMBDA=<λ>` in Step 7. "
                 "Centered-logit form: "
                 "`λ * (logit(P) - logit(0.5))`.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path,
                        help="JSONL trace from BASELINE_TRAJECTORY_TRACE.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional markdown report destination.")
    args = parser.parse_args()

    if not args.trace.exists():
        print(f"ERROR: trace file not found: {args.trace}", file=sys.stderr)
        return 2

    rows = load_trace(args.trace)
    if not rows:
        print("ERROR: no valid trace rows (all P were None or file empty).",
              file=sys.stderr)
        return 3

    stats = per_turn_stats(rows)
    hist = p_histogram(rows)
    gates = evaluate_gates(stats)
    lams = lambda_suggestions(stats)
    report = render_report(args.trace, stats, gates, hist, lams)

    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nReport written to {args.out}", file=sys.stderr)
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
