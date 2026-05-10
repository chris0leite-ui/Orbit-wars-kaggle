"""scripts/lr_diag_e5_bootstrap_coef.py — E5: bootstrap coefficient stability.

Diagnostic. Answers: which features have stable signal vs noise?

50 bootstraps × 3 regularization regimes (L2 C=1, L1 C=0.1, L2
cw='balanced'). Per-feature record: mean coef, std coef, sign-flip
rate, fraction-nonzero (L1 only).

Feature design:
  - 11 raw numeric (standardized)
  - Compound one-hot (5)
  - Race one-hot (26)
  - Driver frequency-encoded (1)
  - 4 cheap pairwise interactions (TyreLife×Stint, LapNumber×Position,
    Cumulative_Degradation×TyreLife, Position×LapNumber)
Total ~47 features.

Output: scripts/artifacts/lr_diag_e5_bootstrap_coef.json + console.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"


def build_features(df):
    num_cols = [c for c in df.columns
                if c not in ["Driver", "Compound", "Race", TARGET, "id"]
                and pd.api.types.is_numeric_dtype(df[c])]
    X_num = df[num_cols].values.astype(np.float64)
    sc = StandardScaler()
    X_num = sc.fit_transform(X_num)

    # Compound one-hot
    comp_dum = pd.get_dummies(df["Compound"], prefix="Cmp",
                              dtype=np.float64).values
    comp_names = [f"Cmp_{c}" for c in
                  pd.get_dummies(df["Compound"], prefix="Cmp").columns
                  if isinstance(c, str)]
    comp_names = list(pd.get_dummies(df["Compound"], prefix="Cmp").columns)

    # Race one-hot (26)
    race_dum = pd.get_dummies(df["Race"], prefix="Race",
                              dtype=np.float64).values
    race_names = list(pd.get_dummies(df["Race"], prefix="Race").columns)

    # Driver frequency
    drv_freq = df["Driver"].map(df["Driver"].value_counts()).values.reshape(-1, 1)
    drv_freq = StandardScaler().fit_transform(drv_freq.astype(np.float64))
    drv_names = ["Driver_freq"]

    # 4 cheap interactions on standardized numerics
    nm_idx = {n: i for i, n in enumerate(num_cols)}
    int_pairs = [
        ("TyreLife", "Stint"),
        ("LapNumber", "Position"),
        ("Cumulative_Degradation", "TyreLife"),
        ("Position", "LapNumber"),
    ]
    ints = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]]
        for a, b in int_pairs if a in nm_idx and b in nm_idx
    ])
    int_names = [f"int_{a}_x_{b}" for a, b in int_pairs
                 if a in nm_idx and b in nm_idx]

    X = np.hstack([X_num, comp_dum, race_dum, drv_freq, ints])
    names = (list(num_cols) + comp_names + race_names + drv_names + int_names)
    return X, names, num_cols


def run_regime(X, y, regime, n_boot=50, seed=42):
    """Return coefficient matrix (n_boot, n_features) and per-boot OOB AUC."""
    rng = np.random.default_rng(seed)
    K = X.shape[1]
    coefs = np.zeros((n_boot, K))
    aucs = np.zeros(n_boot)
    n = len(y)
    for b in range(n_boot):
        idx = rng.choice(n, size=n // 2, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))[:50000]  # cap OOB eval
        if regime == "l2":
            lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                    max_iter=1000)
        elif regime == "l1":
            lr = LogisticRegression(C=0.1, penalty="l1", solver="saga",
                                    max_iter=2000)
        elif regime == "l2_balanced":
            lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                    max_iter=1000, class_weight="balanced")
        else:
            raise ValueError(regime)
        lr.fit(X[idx], y[idx])
        coefs[b] = lr.coef_[0]
        if len(oob) > 1000:
            p = lr.predict_proba(X[oob])[:, 1]
            try:
                aucs[b] = roc_auc_score(y[oob], p)
            except ValueError:
                aucs[b] = float("nan")
        if (b + 1) % 10 == 0:
            print(f"  {regime}: {b+1}/{n_boot} done", flush=True)
    return coefs, aucs


def summarize(coefs, names):
    K = coefs.shape[1]
    rows = []
    for k in range(K):
        c = coefs[:, k]
        sign_flip = float(np.mean(np.sign(c) != np.sign(np.median(c))))
        nonzero = float(np.mean(np.abs(c) > 1e-6))
        rows.append({
            "feature": names[k],
            "coef_mean": round(float(np.mean(c)), 4),
            "coef_std": round(float(np.std(c)), 4),
            "coef_median": round(float(np.median(c)), 4),
            "abs_mean": round(float(np.mean(np.abs(c))), 4),
            "sign_flip_rate": round(sign_flip, 3),
            "frac_nonzero": round(nonzero, 3),
            "snr": round(float(abs(np.mean(c)) / (np.std(c) + 1e-9)), 2),
        })
    return rows


def main():
    df = pd.read_csv("data/train.csv")
    y = df[TARGET].astype(int).values
    X, names, num_cols = build_features(df)
    print(f"Feature matrix: {X.shape}; {len(num_cols)} numeric, "
          f"{X.shape[1] - len(num_cols)} categorical/interaction")

    out = {"feature_count": X.shape[1], "n_boot": 50, "regimes": {}}
    # l1 saga dropped (>>1 min/fit at C=0.1; E8 already settled L1 is
    # rank-no-op; l2 + l2_balanced sufficient for signal-vs-noise)
    for regime in ["l2", "l2_balanced"]:
        print(f"\n=== regime: {regime} ===")
        coefs, aucs = run_regime(X, y, regime, n_boot=50)
        rows = summarize(coefs, names)
        out["regimes"][regime] = {
            "rows": rows,
            "auc_oob_mean": round(float(np.nanmean(aucs)), 5),
            "auc_oob_std": round(float(np.nanstd(aucs)), 5),
        }

    json_path = ART / "lr_diag_e5_bootstrap_coef.json"
    json_path.write_text(json.dumps(out, indent=2))

    # Console synthesis: top SNR features per regime (real signal),
    # high-flip features (noise), and L1 sparsity
    for regime in ["l2", "l2_balanced"]:
        rows = out["regimes"][regime]["rows"]
        rows_sorted = sorted(rows, key=lambda r: -r["snr"])
        print(f"\n=== {regime}: AUC OOB = "
              f"{out['regimes'][regime]['auc_oob_mean']:.5f} "
              f"± {out['regimes'][regime]['auc_oob_std']:.5f} ===")
        print(f"{'feature':<30s} {'mean':>9s} {'std':>9s} "
              f"{'SNR':>5s} {'flip':>5s} {'NZ':>5s}")
        print("-" * 70)
        for r in rows_sorted[:10]:
            print(f"{r['feature']:<30s} {r['coef_mean']:>+9.4f} "
                  f"{r['coef_std']:>9.4f} {r['snr']:>5.1f} "
                  f"{r['sign_flip_rate']:>5.2f} {r['frac_nonzero']:>5.2f}")
        # high-flip / noise features
        flips = [r for r in rows if r["sign_flip_rate"] > 0.2]
        flips.sort(key=lambda r: -r["sign_flip_rate"])
        print(f"\n  high-sign-flip (noise) features: {len(flips)}")
        for r in flips[:5]:
            print(f"    {r['feature']:<28s} flip={r['sign_flip_rate']:.2f} "
                  f"snr={r['snr']:.1f}")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
