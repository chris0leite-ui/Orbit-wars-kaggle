"""scripts/lr_diag_e8_grid.py — E8: pruned LR-meta hyperparameter grid.

Pruned (2026-05-07 PM): l2 only via lbfgs (saga+l1 was 1h+ on tiny C
and prior partial datapoints gave Δ within 1 bp — the rank-no-op
verdict survives without the slow corner). Adds the logits-only
input ablation (P2 from prior plan) while we're here.

Grid: 2 cw × 5 C × 2 input_modes = 20 fits × 5-fold = 100 LR fits.
Each ~3-8s on lbfgs C∈[0.01, 100] → estimated <5 min total.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]
EXTRAS = ["d16_orig_continuous_only", "p1_single_cb_v3_gpu",
          "d17_h1d_yekenot_full"]
ALL_BASES = K21_BASES + EXTRAS


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand_full(P):
    n = len(P)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    rk = np.column_stack([np.argsort(np.argsort(c)) / n for c in P.T])
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _expand_logit_only(P):
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.log(Pc / (1 - Pc))


def _oof(F, y, cw, C):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(
            C=C, class_weight=cw, penalty="l2",
            solver="lbfgs", max_iter=2000
        )
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    P = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in ALL_BASES])

    inputs = {
        "full_72_P_rank_logit": _expand_full(P),
        "logit_only_24": _expand_logit_only(P),
    }

    # Anchor = current production: full input, cw=None, C=1.0
    print("Fitting anchor (full input, cw=None, C=1.0, l2 lbfgs)...")
    anchor_oof, anchor_auc = _oof(inputs["full_72_P_rank_logit"], y, None, 1.0)
    print(f"  anchor AUC: {anchor_auc:.6f}")

    grid = []
    cw_vals = [None, "balanced"]
    C_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
    total = len(inputs) * len(cw_vals) * len(C_vals)
    i = 0
    for input_name, F in inputs.items():
        for cw in cw_vals:
            for C in C_vals:
                i += 1
                print(f"[{i}/{total}] input={input_name} cw={cw} C={C:>7g} ...",
                      end="", flush=True)
                oof, auc = _oof(F, y, cw, C)
                rho, _ = spearmanr(oof, anchor_oof)
                bp = (auc - anchor_auc) * 1e4
                print(f"  AUC={auc:.5f}  Δ={bp:+.2f}bp  ρ={rho:.5f}")
                grid.append({
                    "input": input_name,
                    "class_weight": str(cw),
                    "C": C,
                    "penalty": "l2",
                    "oof_auc": round(auc, 6),
                    "delta_bp_vs_anchor": round(float(bp), 3),
                    "spearman_rho_vs_anchor": round(float(rho), 6),
                })

    out = {
        "anchor_auc": round(anchor_auc, 6),
        "anchor_config": {
            "input": "full_72_P_rank_logit",
            "class_weight": "None",
            "C": 1.0,
            "penalty": "l2",
        },
        "grid": grid,
        "salvaged_l1": [
            {"class_weight": "None", "C": 0.01, "penalty": "l1",
             "delta_bp_vs_anchor": -0.59, "spearman_rho_vs_anchor": 0.99890,
             "note": "from killed run; rank-no-op confirmed"},
            {"class_weight": "None", "C": 0.1, "penalty": "l1",
             "delta_bp_vs_anchor": 0.08, "spearman_rho_vs_anchor": 0.99975,
             "note": "from killed run; rank-no-op confirmed"},
        ],
    }
    json_path = ART / "lr_diag_e8_grid.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== E8 grid summary ===")
    print(f"anchor: AUC={anchor_auc:.6f}\n")
    print(f"{'input':<24s} {'cw':<10s} {'C':>8s} {'AUC':>9s} "
          f"{'Δbp':>7s} {'ρ':>9s}")
    print("-" * 80)
    rows_sorted = sorted(grid, key=lambda r: -r["oof_auc"])
    for r in rows_sorted:
        print(f"{r['input']:<24s} {r['class_weight']:<10s} "
              f"{r['C']:>8g} {r['oof_auc']:>9.5f} "
              f"{r['delta_bp_vs_anchor']:>+7.2f} "
              f"{r['spearman_rho_vs_anchor']:>9.5f}")
    print("-" * 80)
    rhos = [r["spearman_rho_vs_anchor"] for r in grid]
    bps = [r["delta_bp_vs_anchor"] for r in grid]
    print(f"Spearman ρ to anchor: min={min(rhos):.5f}, "
          f"median={float(np.median(rhos)):.5f}, max={max(rhos):.5f}")
    print(f"Δ bp range: [{min(bps):+.2f}, {max(bps):+.2f}]")
    # split by input
    full_bps = [r["delta_bp_vs_anchor"] for r in grid
                if r["input"] == "full_72_P_rank_logit"]
    logit_bps = [r["delta_bp_vs_anchor"] for r in grid
                 if r["input"] == "logit_only_24"]
    print(f"\nfull-input Δ range : [{min(full_bps):+.2f}, {max(full_bps):+.2f}]")
    print(f"logit-only Δ range : [{min(logit_bps):+.2f}, {max(logit_bps):+.2f}]")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
