"""trace_pv_eta_scoring.py

Run one full game (PV_ETA anchor vs an opponent), capture per-call
scoring of every candidate that reaches `score_candidate_v4` /
`score_candidate_v4_joint`, then reconstruct the pre-multiplier delta
(divide out gamma**(wait_N+eta)) so we can see:

  1. how often the PV_ETA discount changes the top-K ranking, and
  2. what candidate classes get demoted (high-eta, wait_N>0).

Tests the "double-discount" hypothesis: if the leaf already encodes
the eta penalty via its smaller post-arrival ship_term window, the
gamma**(wait_N+eta) multiplier on top is a relative penalty without
clean modeling justification.

Usage:
  python scripts/trace_pv_eta_scoring.py --opp v7_0_drop_one --seed 0 --top-k 5
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_single_file(path: Path, mod_name: str):
    """Import a single-file submission bundle by absolute path."""
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="submissions/baseline_pv_eta_anchor_1163.py")
    ap.add_argument("--opp", default="submissions/v7_0_drop_one.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--out", default="audit/pv_eta_trace.json",
                    help="JSON dump of every scored candidate")
    args = ap.parse_args()

    focal_path = (REPO / args.focal).resolve()
    opp_path = (REPO / args.opp).resolve()

    focal = _load_single_file(focal_path, "_focal_pv_eta")
    opp = _load_single_file(opp_path, "_opp_agent")

    # Per-call capture state. `current_turn` is updated by the focal
    # wrapper from obs.step before delegating to focal.agent.
    captured: list[dict] = []
    current_turn = {"step": 0, "call_idx": 0}

    orig_v4 = focal.score_candidate_v4
    orig_v4_joint = focal.score_candidate_v4_joint

    def wrap_v4(snap_base, src, tgt, ships, angle, me, num_seats, world,
                baseline_favors, favor_fn, gamma, horizon,
                skip_admissibility=False, wait_N=0, eta_hint=0, model=None):
        result = orig_v4(
            snap_base, src, tgt, ships, angle, me, num_seats, world,
            baseline_favors, favor_fn, gamma, horizon,
            skip_admissibility=skip_admissibility,
            wait_N=wait_N, eta_hint=eta_hint, model=model,
        )
        delta_post, status, eta = result
        captured.append({
            "kind": "solo",
            "turn": current_turn["step"],
            "call_idx": current_turn["call_idx"],
            "src": int(src.id), "tgt": int(tgt.id),
            "tgt_owner": int(tgt.owner),
            "ships": int(ships), "wait_N": int(wait_N),
            "eta": int(eta if eta is not None else 0),
            "horizon": int(horizon), "gamma": float(gamma),
            "delta_post": float(delta_post),
            "status": str(status),
        })
        current_turn["call_idx"] += 1
        return result

    def wrap_v4_joint(snap_base, launches, me, num_seats, world,
                      baseline_favors, favor_fn, gamma, horizon,
                      skip_admissibility=False):
        result = orig_v4_joint(
            snap_base, launches, me, num_seats, world,
            baseline_favors, favor_fn, gamma, horizon,
            skip_admissibility=skip_admissibility,
        )
        delta_post, status = result
        # Joint: each leg has its own (src, tgt, ships, angle, wait_N);
        # we log the SUM of ships and MAX of (wait_N + eta-hint-or-0).
        # Joint Δ is unitary so per-leg attribution would be misleading.
        max_arrival = 0
        total_ships = 0
        for src, tgt, ships, angle, wait_N in launches:
            total_ships += int(ships)
            # joint scorer uses leg_etas (0 for wait_N>0 legs in v1);
            # we can't recover it here without re-running predict_fate.
            # Use wait_N as a lower bound — sufficient for ranking analysis.
            max_arrival = max(max_arrival, int(wait_N))
        captured.append({
            "kind": "joint",
            "turn": current_turn["step"],
            "call_idx": current_turn["call_idx"],
            "n_legs": len(launches),
            "ships": total_ships,
            "wait_N": max_arrival,   # lower bound
            "eta": 0,                # unknown without re-running
            "horizon": int(horizon), "gamma": float(gamma),
            "delta_post": float(delta_post),
            "status": str(status),
        })
        current_turn["call_idx"] += 1
        return result

    focal.score_candidate_v4 = wrap_v4
    focal.score_candidate_v4_joint = wrap_v4_joint

    # Build the seat-aware agent callables for kaggle_environments.
    orig_focal_agent = focal.agent

    def focal_agent(obs, config):
        # obs.step is the global tick; track it for the per-turn group key.
        try:
            current_turn["step"] = int(obs.step)
        except Exception:
            current_turn["step"] = int(obs.get("step", current_turn["step"] + 1))
        current_turn["call_idx"] = 0
        return orig_focal_agent(obs, config)

    def opp_agent(obs, config):
        return opp.agent(obs, config)

    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(args.seed)}, debug=False)
    if args.seat == 0:
        env.run([focal_agent, opp_agent])
    else:
        env.run([opp_agent, focal_agent])

    # Score the results: outcome + cumulative stats.
    last_states = env.steps[-1]
    rewards = [s["reward"] for s in last_states]
    print(f"\n=== Game finished. seed={args.seed} seat={args.seat} ===")
    print(f"  rewards: {rewards}  (focal={'P0' if args.seat == 0 else 'P1'})")
    print(f"  total candidates scored: {len(captured)}")

    # Reconstruct delta_pre per call.
    for c in captured:
        wne = int(c["wait_N"]) + int(c["eta"])
        g = float(c["gamma"])
        d = float(c["delta_post"])
        if wne > 0 and 0.0 < g < 1.0:
            c["delta_pre"] = d / (g ** wne)
            c["mult"] = g ** wne
        else:
            c["delta_pre"] = d
            c["mult"] = 1.0

    # Per-turn analysis. For each turn:
    #   - solo top-K by delta_post
    #   - solo top-K by delta_pre
    #   - any ranking changes?
    K = int(args.top_k)
    by_turn: dict[int, list] = {}
    for c in captured:
        by_turn.setdefault(c["turn"], []).append(c)

    summary = {
        "turns": [],
        "n_turns": len(by_turn),
        "n_candidates": len(captured),
        "n_demoted_by_pv_eta": 0,  # candidate that was top-K-pre but not top-K-post
        "n_promoted_by_pv_eta": 0,
        "high_eta_demoted_examples": [],
        "wait_n_demoted_examples": [],
    }

    for t in sorted(by_turn.keys()):
        cands = [c for c in by_turn[t] if c["kind"] == "solo"]
        if not cands:
            continue
        # Drop -inf'd (admissibility failures): they're never picked.
        scored = [c for c in cands if c["delta_post"] != float("-inf")]
        if not scored:
            continue
        top_post = sorted(scored, key=lambda c: -c["delta_post"])[:K]
        top_pre = sorted(scored, key=lambda c: -c["delta_pre"])[:K]
        key_of = lambda c: (c["src"], c["tgt"], c["ships"], c["wait_N"])
        post_keys = {key_of(c) for c in top_post}
        pre_keys = {key_of(c) for c in top_pre}
        demoted = pre_keys - post_keys
        promoted = post_keys - pre_keys
        summary["n_demoted_by_pv_eta"] += len(demoted)
        summary["n_promoted_by_pv_eta"] += len(promoted)
        for c in top_pre:
            if key_of(c) in demoted and c["eta"] >= 15:
                summary["high_eta_demoted_examples"].append(
                    {"turn": t, **{k: c[k] for k in ("src", "tgt", "ships", "wait_N", "eta", "delta_pre", "delta_post")}}
                )
            if key_of(c) in demoted and c["wait_N"] > 0:
                summary["wait_n_demoted_examples"].append(
                    {"turn": t, **{k: c[k] for k in ("src", "tgt", "ships", "wait_N", "eta", "delta_pre", "delta_post")}}
                )
        summary["turns"].append({
            "turn": t,
            "n_scored": len(scored),
            "top_post": [
                {"src": c["src"], "tgt": c["tgt"], "ships": c["ships"],
                 "wait_N": c["wait_N"], "eta": c["eta"],
                 "delta_post": c["delta_post"], "delta_pre": c["delta_pre"]}
                for c in top_post
            ],
            "top_pre": [
                {"src": c["src"], "tgt": c["tgt"], "ships": c["ships"],
                 "wait_N": c["wait_N"], "eta": c["eta"],
                 "delta_post": c["delta_post"], "delta_pre": c["delta_pre"]}
                for c in top_pre
            ],
        })

    out_path = (REPO / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "captured": captured}, f, indent=2)
    print(f"  wrote: {out_path}")

    # Print a compact summary to stdout.
    print()
    print("=== AGGREGATE ===")
    print(f"  turns w/ scoring activity: {len(summary['turns'])}")
    print(f"  top-{K} demotions across all turns (pre→post): "
          f"{summary['n_demoted_by_pv_eta']}")
    print(f"  high-eta (eta>=15) demoted examples: "
          f"{len(summary['high_eta_demoted_examples'])}")
    print(f"  wait_N>0 demoted examples: "
          f"{len(summary['wait_n_demoted_examples'])}")
    print()
    print("=== FIRST 10 TURNS WITH RANKING CHANGE ===")
    n_shown = 0
    for tr in summary["turns"]:
        post_keys = {(c["src"], c["tgt"], c["ships"], c["wait_N"])
                     for c in tr["top_post"]}
        pre_keys = {(c["src"], c["tgt"], c["ships"], c["wait_N"])
                    for c in tr["top_pre"]}
        if pre_keys == post_keys:
            continue
        print(f"\n-- turn {tr['turn']:3d}  (n_scored={tr['n_scored']}) --")
        print("  POST-PV (actual rank):")
        for i, c in enumerate(tr["top_post"]):
            print(f"    #{i+1}: src={c['src']:2d}→tgt={c['tgt']:2d}  "
                  f"ships={c['ships']:3d}  wait={c['wait_N']:2d}  eta={c['eta']:3d}  "
                  f"Δpost={c['delta_post']:+.2f}  Δpre={c['delta_pre']:+.2f}")
        print("  PRE-PV  (no-discount rank):")
        for i, c in enumerate(tr["top_pre"]):
            print(f"    #{i+1}: src={c['src']:2d}→tgt={c['tgt']:2d}  "
                  f"ships={c['ships']:3d}  wait={c['wait_N']:2d}  eta={c['eta']:3d}  "
                  f"Δpost={c['delta_post']:+.2f}  Δpre={c['delta_pre']:+.2f}")
        n_shown += 1
        if n_shown >= 10:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
