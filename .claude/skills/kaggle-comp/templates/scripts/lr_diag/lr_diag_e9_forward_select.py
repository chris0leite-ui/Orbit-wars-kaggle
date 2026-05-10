"""scripts/lr_diag_e9_forward_select.py — E9: forward selection trace.

Diagnostic. Answers: in what order would a greedy CV-optimal LR-meta
add bases? When does it plateau? Is K=24 really the right pool size?

Procedure:
  1. Start with empty selected set S.
  2. At each step, for every candidate c not in S, fit 5-fold OOF
     LR-meta on S∪{c} and record OOF AUC.
  3. Add the c that maximizes OOF AUC; record (step, base, AUC).
  4. Continue until all 24 bases added (so we see the full trace).
  5. Output pick order, marginal AUC contribution per pick, and
     plateau-point detection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def _expand(P):
    n = len(P)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    rk = np.column_stack([np.argsort(np.argsort(c)) / n for c in P.T])
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _meta_oof(F, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000)
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    y_full = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    P_full_all = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in ALL_BASES])
    # Subsample to 110k for forward selection trace (1500 LR fits would
    # be 3h+ on full 439k; stratified subsample preserves AUC measurement
    # resolution to ~0.001).
    rng = np.random.default_rng(42)
    idx0 = np.where(y_full == 0)[0]
    idx1 = np.where(y_full == 1)[0]
    n0_keep = min(len(idx0), 88000)
    n1_keep = min(len(idx1), 22000)
    idx_sub = np.concatenate([
        rng.choice(idx0, n0_keep, replace=False),
        rng.choice(idx1, n1_keep, replace=False),
    ])
    rng.shuffle(idx_sub)
    y = y_full[idx_sub]
    P_full = P_full_all[idx_sub]
    print(f"Subsampled: n={len(y)} (full {len(y_full)}); "
          f"pos rate {y.mean():.3f}")

    selected = []
    remaining = list(range(len(ALL_BASES)))
    trace = []
    prev_auc = 0.5
    early_stop_window = []

    print(f"Forward selection over K={len(ALL_BASES)} bases ...")
    while remaining:
        best_c = None
        best_auc = -np.inf
        for c in remaining:
            cols = selected + [c]
            P = P_full[:, cols]
            F = _expand(P)
            auc = _meta_oof(F, y)
            if auc > best_auc:
                best_auc = auc
                best_c = c
        # record step
        step_n = len(selected) + 1
        delta = best_auc - prev_auc
        trace.append({
            "step": step_n,
            "added_base": ALL_BASES[best_c],
            "oof_auc": round(best_auc, 6),
            "delta_bp": round(1e4 * delta, 3),
            "k_total": step_n,
        })
        print(f"  [{step_n:>2d}/{len(ALL_BASES)}] add {ALL_BASES[best_c]:<32s}  "
              f"AUC={best_auc:.5f}  Δ={1e4*delta:+.2f}bp", flush=True)
        selected.append(best_c)
        remaining.remove(best_c)
        prev_auc = best_auc
        # early stop: 3 consecutive Δ < 0.05 bp AND step >= 8
        early_stop_window.append(1e4 * delta)
        if len(early_stop_window) > 3:
            early_stop_window.pop(0)
        if step_n >= 8 and all(d < 0.05 for d in early_stop_window) \
                and len(early_stop_window) == 3:
            print(f"  early stop: 3 consecutive Δ < 0.05 bp at step {step_n}")
            break

    # Plateau detection: first step where delta < 0.05 bp
    plateau_step = next(
        (t["step"] for t in trace[3:] if t["delta_bp"] < 0.05), None
    )
    # Best AUC step
    best_step = max(trace, key=lambda t: t["oof_auc"])

    out = {
        "n_bases": len(ALL_BASES),
        "trace": trace,
        "plateau_step_first_under_0.05bp": plateau_step,
        "best_step": best_step,
        "production_K24_auc": trace[-1]["oof_auc"],
    }
    json_path = ART / "lr_diag_e9_forward_select.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== E9 forward-selection trace summary ===")
    print(f"Best AUC step: K={best_step['step']}, AUC={best_step['oof_auc']:.5f}")
    print(f"Production K=24 AUC: {trace[-1]['oof_auc']:.5f}")
    print(f"Plateau (Δ < 0.05 bp) first at step: {plateau_step}")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
