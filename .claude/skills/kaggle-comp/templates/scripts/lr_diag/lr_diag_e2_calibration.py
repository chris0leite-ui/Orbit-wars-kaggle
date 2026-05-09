"""scripts/lr_diag_e2_calibration.py — E2: per-base LR calibration map.

Diagnostic. Answers: which of K=24 bases are mis-calibrated, and is
the LR-meta doing implicit re-calibration that a per-base Platt fix
would short-circuit?

For each base b, fit per-fold LR(logit_b → y) and record:
  - slope, intercept (averaged over 5 folds)
  - calibrated AUC (= base AUC since LR is monotone in 1D)
  - Brier score before / after calibration
  - log-loss before / after calibration

A well-calibrated base has slope ≈ 1, intercept ≈ 0, Brier-after ≈
Brier-before. Mis-calibrated bases (slope far from 1 or intercept != 0)
indicate the LR-meta is spending coefficients on re-calibration that a
1-D Platt scaler would handle for free.

Output: scripts/artifacts/lr_diag_e2_calibration.json + console.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score)
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


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y)), y))

    rows = []
    for name in ALL_BASES:
        p = _pos(ART / f"oof_{name}_strat.npy")
        l = _logit(p)
        # base metrics
        auc = roc_auc_score(y, p)
        brier = brier_score_loss(y, p)
        ll = log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))
        # per-fold LR(logit -> y); take mean slope/intercept; OOF cal preds
        cal = np.zeros_like(p)
        slopes, intercepts = [], []
        for tr, va in folds:
            lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
            lr.fit(l[tr].reshape(-1, 1), y[tr])
            slopes.append(float(lr.coef_[0, 0]))
            intercepts.append(float(lr.intercept_[0]))
            cal[va] = lr.predict_proba(l[va].reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 1e-9, 1 - 1e-9)
        cal_auc = roc_auc_score(y, cal)
        cal_brier = brier_score_loss(y, cal)
        cal_ll = log_loss(y, cal)

        rows.append({
            "base": name,
            "auc": round(auc, 5),
            "brier_raw": round(brier, 5),
            "logloss_raw": round(ll, 5),
            "platt_slope": round(float(np.mean(slopes)), 4),
            "platt_intercept": round(float(np.mean(intercepts)), 4),
            "brier_calibrated": round(cal_brier, 5),
            "logloss_calibrated": round(cal_ll, 5),
            "brier_delta_bp": round(1e4 * (brier - cal_brier), 2),
            "logloss_delta_bp": round(1e4 * (ll - cal_ll), 2),
            "auc_calibrated": round(cal_auc, 5),
        })

    out = {"n_bases": len(rows), "rows": rows}
    json_path = ART / "lr_diag_e2_calibration.json"
    json_path.write_text(json.dumps(out, indent=2))

    # Sort by absolute slope deviation from 1 (mis-calibration severity)
    rows_sorted = sorted(rows, key=lambda r: -abs(r["platt_slope"] - 1.0))

    print("\n=== E2 per-base LR calibration map (K=24) ===")
    print(f"{'base':<32s} {'AUC':>7s} {'slope':>7s} {'intcpt':>7s} "
          f"{'Brier-Δbp':>10s} {'LL-Δbp':>9s}")
    print("-" * 80)
    for r in rows_sorted:
        print(f"{r['base']:<32s} {r['auc']:>7.5f} "
              f"{r['platt_slope']:>+7.3f} {r['platt_intercept']:>+7.3f} "
              f"{r['brier_delta_bp']:>+10.2f} {r['logloss_delta_bp']:>+9.2f}")

    # Summary stats
    slopes = [r["platt_slope"] for r in rows]
    deltas = [r["brier_delta_bp"] for r in rows]
    print("-" * 80)
    print(f"slope median: {np.median(slopes):.3f}  mean: {np.mean(slopes):.3f}")
    print(f"slope range : [{min(slopes):.3f}, {max(slopes):.3f}]")
    print(f"  bases with |slope-1| > 0.1 : "
          f"{sum(1 for s in slopes if abs(s-1) > 0.1)}/24")
    print(f"  bases with |slope-1| > 0.3 : "
          f"{sum(1 for s in slopes if abs(s-1) > 0.3)}/24")
    print(f"Brier-improvement-from-Platt median: {np.median(deltas):.2f} bp")
    print(f"  bases with Brier-Δ > 5 bp : "
          f"{sum(1 for d in deltas if d > 5)}/24 (these are mis-calibrated)")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
