"""scripts/lr_diag_e6_residual_interactions.py — E6: LR-residual interaction map.

Diagnostic. Answers: which 2-way feature interactions does linear LR
miss? Where are missing interactions concentrated?

Approach:
  1. Fit 5-fold OOF LR (lbfgs L2 C=1, balanced) on standardized
     numerics + categorical dummies.
  2. Compute residuals r = y - p_LR (signed; shows direction).
  3. For each pair of numeric features (i, j) bin into 5×5 quintile
     grid; record mean residual per bin.
  4. Score each pair by max|mean_residual_per_bin| − global mean.
     High score = strong missing-interaction signal.
  5. Compare against PRIMARY OOF residuals: where PRIMARY > LR by
     a lot and the bin agrees with high LR-residual magnitude, that's
     a confirmed interaction the GBDT pool already captures.

Output: scripts/artifacts/lr_diag_e6_residual_interactions.json + console.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"


def main():
    df = pd.read_csv("data/train.csv")
    y = df[TARGET].astype(int).values

    num_cols = [c for c in df.columns
                if c not in ["Driver", "Compound", "Race", TARGET, "id"]
                and pd.api.types.is_numeric_dtype(df[c])]
    X_num = df[num_cols].values.astype(np.float64)
    X_num = StandardScaler().fit_transform(X_num)
    comp_dum = pd.get_dummies(df["Compound"], prefix="Cmp",
                              dtype=np.float64).values
    race_dum = pd.get_dummies(df["Race"], prefix="Race",
                              dtype=np.float64).values
    drv_freq = df["Driver"].map(df["Driver"].value_counts()).values.reshape(-1, 1)
    drv_freq = StandardScaler().fit_transform(drv_freq.astype(np.float64))
    X = np.hstack([X_num, comp_dum, race_dum, drv_freq])
    print(f"Feature matrix: {X.shape}; {len(num_cols)} numeric")

    # 5-fold OOF LR
    print("Fitting 5-fold OOF LR ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_lr = np.zeros(len(y))
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000, class_weight="balanced")
        lr.fit(X[tr], y[tr])
        p_lr[va] = lr.predict_proba(X[va])[:, 1]
        print(f"  fold {fi+1}/5 done", flush=True)
    lr_auc = roc_auc_score(y, p_lr)
    print(f"LR OOF AUC: {lr_auc:.5f}")
    r_lr = y - p_lr  # signed residual

    # PRIMARY for comparison
    prim = np.load(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    if prim.ndim == 2:
        prim = prim[:, 1]
    prim = prim.astype(np.float64)
    prim_auc = roc_auc_score(y, prim)
    r_prim = y - prim
    print(f"PRIMARY OOF AUC: {prim_auc:.5f}")

    # Quintile bins of standardized numerics (5 bins per feature)
    bin_idx = {}
    for i, c in enumerate(num_cols):
        try:
            bin_idx[c] = pd.qcut(X_num[:, i], 5, labels=False,
                                 duplicates="drop").astype(int)
        except Exception:
            continue

    pairs = list(combinations(num_cols, 2))
    print(f"Scoring {len(pairs)} pairs...")
    pair_rows = []
    for a, b in pairs:
        if a not in bin_idx or b not in bin_idx:
            continue
        ba, bb = bin_idx[a], bin_idx[b]
        # 5x5 mean residual grid for LR
        grid_lr = np.zeros((5, 5))
        grid_n = np.zeros((5, 5), dtype=int)
        grid_prim = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                mask = (ba == i) & (bb == j)
                n = int(mask.sum())
                grid_n[i, j] = n
                if n > 100:
                    grid_lr[i, j] = float(r_lr[mask].mean())
                    grid_prim[i, j] = float(r_prim[mask].mean())
        # score: max|cell mean| − global mean|residual|
        valid = grid_n > 100
        if valid.sum() < 4:
            continue
        lr_dev_max = float(np.max(np.abs(grid_lr[valid])))
        prim_dev_max = float(np.max(np.abs(grid_prim[valid])))
        # GBDT-captures-interaction score: cells where LR resid is high
        # but PRIMARY resid is much smaller
        gain = float(np.max(np.abs(grid_lr[valid])) -
                     np.max(np.abs(grid_prim[valid])))
        pair_rows.append({
            "feat_a": a,
            "feat_b": b,
            "lr_max_cell_resid": round(lr_dev_max, 4),
            "primary_max_cell_resid": round(prim_dev_max, 4),
            "gbdt_captures_gap_bp": round(1e4 * gain, 1),
            "n_valid_cells": int(valid.sum()),
        })

    pair_rows.sort(key=lambda r: -r["lr_max_cell_resid"])
    out = {
        "lr_auc": round(lr_auc, 5),
        "primary_auc": round(prim_auc, 5),
        "n_pairs_scored": len(pair_rows),
        "pairs_top": pair_rows[:25],
        "pairs_gbdt_captures_top": sorted(
            pair_rows, key=lambda r: -r["gbdt_captures_gap_bp"])[:15],
    }
    json_path = ART / "lr_diag_e6_residual_interactions.json"
    json_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== E6: LR (AUC {lr_auc:.4f}) vs PRIMARY (AUC {prim_auc:.4f}) ===")
    print(f"\nTop 15 pairs by max|LR cell residual| (where LR breaks):")
    print(f"{'feat_a':<28s} {'feat_b':<28s} {'LR max':>7s} "
          f"{'Prim max':>9s} {'GBDT-gap-bp':>11s}")
    print("-" * 90)
    for r in pair_rows[:15]:
        print(f"{r['feat_a']:<28s} {r['feat_b']:<28s} "
              f"{r['lr_max_cell_resid']:>+7.4f} "
              f"{r['primary_max_cell_resid']:>+9.4f} "
              f"{r['gbdt_captures_gap_bp']:>+11.1f}")
    print(f"\nTop 10 pairs by GBDT-captures-gap (LR fails, PRIMARY rescues):")
    for r in out["pairs_gbdt_captures_top"][:10]:
        print(f"  {r['feat_a']:<26s} × {r['feat_b']:<26s} "
              f"gap={r['gbdt_captures_gap_bp']:+.1f} bp")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
