"""scripts/lr_diag_e4_per_segment.py — E4: per-segment LR AUC.

Diagnostic. Answers: where is the DGP locally linear vs locally
nonlinear? Cell = (Compound, Stint_quintile). Within each cell, fit a
class-weighted LR on raw 11 numeric features (5-fold StratKF inside
the cell, OOF AUC). Compare to the segment-conditional AUC of our
PRIMARY (d17 K=24 d18pool h1d).

Output: scripts/artifacts/lr_diag_e4_per_segment.json + console heatmap.
"""
from __future__ import annotations

import json
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

    # 11 numeric features (drop id + categorical + target)
    cat_cols = ["Driver", "Compound", "Race"]
    num_cols = [c for c in df.columns
                if c not in cat_cols + [TARGET, "id"]
                and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Using {len(num_cols)} numeric features: {num_cols}")
    X = df[num_cols].values.astype(np.float64)

    # Stint quintile (using TyreLife as proxy for stint progression if
    # Stint is unavailable; here we have Stint directly per schema)
    stint = df["Stint"].values if "Stint" in df.columns else None
    if stint is None:
        # fallback: use TyreLife quintile
        tl = df["TyreLife"].values
        stint_q = pd.qcut(tl, 5, labels=False, duplicates="drop")
    else:
        # Stint is integer-valued; bin into quintiles too
        stint_q = pd.qcut(stint.astype(float), 5,
                          labels=False, duplicates="drop")

    compounds = df["Compound"].astype(str).values
    cell_ids = pd.Series(
        [f"{c}|q{q}" for c, q in zip(compounds, stint_q)]
    ).values

    # PRIMARY OOF for comparison
    prim = np.load(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    if prim.ndim == 2:
        prim = prim[:, 1]
    prim = prim.astype(np.float64)

    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for cell in pd.unique(cell_ids):
        mask = cell_ids == cell
        if mask.sum() < 200:
            continue
        Xc, yc, pc = X[mask], y[mask], prim[mask]
        if yc.min() == yc.max():
            continue
        prim_auc = roc_auc_score(yc, pc) if yc.sum() > 0 else float("nan")

        # 5-fold OOF LR within the cell
        try:
            oof = np.zeros(len(yc))
            for tr, va in skf.split(np.zeros(len(yc)), yc):
                if yc[tr].min() == yc[tr].max():
                    continue
                sc = StandardScaler()
                Xt = sc.fit_transform(Xc[tr])
                Xv = sc.transform(Xc[va])
                lr = LogisticRegression(
                    C=1.0, class_weight="balanced",
                    max_iter=2000, solver="lbfgs"
                )
                lr.fit(Xt, yc[tr])
                oof[va] = lr.predict_proba(Xv)[:, 1]
            lr_auc = roc_auc_score(yc, oof)
        except Exception:
            lr_auc = float("nan")

        rows.append({
            "cell": cell,
            "n": int(mask.sum()),
            "pos_rate": round(float(yc.mean()), 4),
            "lr_auc": round(float(lr_auc), 4),
            "primary_auc": round(float(prim_auc), 4),
            "gap_bp": round(1e4 * (prim_auc - lr_auc), 1),
        })

    rows.sort(key=lambda r: -r["gap_bp"])
    out = {"n_cells": len(rows), "rows": rows}
    json_path = ART / "lr_diag_e4_per_segment.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== E4 per-segment LR AUC (Compound × Stint-quintile) ===")
    print(f"{'cell':<18s} {'n':>6s} {'pos%':>6s} {'LR_AUC':>7s} "
          f"{'Prim':>7s} {'gap_bp':>8s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['cell']:<18s} {r['n']:>6d} {100*r['pos_rate']:>5.1f}% "
              f"{r['lr_auc']:>7.4f} {r['primary_auc']:>7.4f} "
              f"{r['gap_bp']:>+8.1f}")
    # summary
    gaps = [r["gap_bp"] for r in rows]
    lr_aucs = [r["lr_auc"] for r in rows]
    print("-" * 60)
    print(f"cells: {len(rows)}")
    print(f"LR AUC range: [{min(lr_aucs):.4f}, {max(lr_aucs):.4f}]")
    print(f"PRIMARY-LR gap median: {np.median(gaps):.1f} bp; "
          f"max: {max(gaps):.1f} bp; min: {min(gaps):.1f} bp")
    print(f"  cells where LR ≥ PRIMARY: "
          f"{sum(1 for g in gaps if g <= 0)}/{len(rows)}")
    print(f"  cells where gap > 100 bp: "
          f"{sum(1 for g in gaps if g > 100)}/{len(rows)}")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
