"""scripts/lr_diag_a4_per_segment.py — A4: per-Compound LR specialists.

Arc C, A4. Cross-confirmation of A2's finding via a structurally
distinct construction.

Design: one LR per Compound (5 specialists). Each fit only on rows
where Compound=c, predicts only those val rows; predictions
concatenated into a single OOF column.

Features: rich (vanilla numeric + Compound-conditional Race dummies +
the 9 E6 Stint-cross interactions). class_weight='balanced'.

Per-Compound row counts (from E4):
  HARD   ~170k    SOFT   ~38k    MEDIUM   ~210k
  INTERMEDIATE ~17k    WET ~1.3k

Tests whether per-segment specialization captures Compound-conditional
patterns the global model can't. If A2_rich was meta-null because of
(Stint × *) saturation in the GBDT pool, then A4 should also be
meta-null — but tests via different mechanism (segment specialization
vs interaction engineering).
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
from sklearn.preprocessing import StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED = 42

K10_BASES = [
    "d17_h1d_yekenot_full", "p1_single_cb_v3_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000)
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def build_features(df_train, df_test):
    num_cols = [c for c in df_train.columns
                if c not in ["Driver", "Compound", "Race", TARGET, "id"]
                and pd.api.types.is_numeric_dtype(df_train[c])]
    df = pd.concat([df_train.assign(__split="tr"),
                    df_test.assign(__split="te")], ignore_index=True)
    X_num = df[num_cols].values.astype(np.float64)
    race = pd.get_dummies(df["Race"], prefix="Race", dtype=np.float64).values
    drv_counts = df_train["Driver"].value_counts()
    drv_freq = df["Driver"].map(drv_counts).fillna(0).values.reshape(-1, 1)
    drv_freq = drv_freq.astype(np.float64)
    drv_freq = (drv_freq - drv_freq.mean()) / (drv_freq.std() + 1e-12)
    nm_idx = {n: i for i, n in enumerate(num_cols)}
    rich_pairs = [
        ("Stint", "RaceProgress"), ("Stint", "Year"),
        ("Stint", "LapNumber"), ("Stint", "TyreLife"),
        ("Stint", "LapTime (s)"), ("Stint", "LapTime_Delta"),
        ("Stint", "Cumulative_Degradation"), ("Stint", "Position"),
        ("LapTime (s)", "LapTime_Delta"),
    ]
    rich = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]]
        for a, b in rich_pairs if a in nm_idx and b in nm_idx
    ])
    cheap_pairs = [
        ("TyreLife", "Stint"), ("LapNumber", "Position"),
        ("Cumulative_Degradation", "TyreLife"), ("Position", "LapNumber"),
    ]
    cheap = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]]
        for a, b in cheap_pairs if a in nm_idx and b in nm_idx
    ])
    X_rich = np.hstack([X_num, race, drv_freq, cheap, rich])

    sp = df["__split"].values
    return X_rich[sp == "tr"], X_rich[sp == "te"]


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    compound = train["Compound"].astype(str).values
    compound_te = test["Compound"].astype(str).values

    X_train, X_test = build_features(train, test)
    print(f"Features: {X_train.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(X_test.shape[0])
    fold_aucs = []
    per_compound_stats = []

    compounds = list(np.unique(compound))
    print(f"Compounds: {compounds}")

    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        # per-Compound LR
        cmp_va_pred = np.zeros(len(va))
        cmp_te_pred = np.zeros(X_test.shape[0])
        for c in compounds:
            tr_c_mask = (compound[tr] == c)
            va_c_mask = (compound[va] == c)
            te_c_mask = (compound_te == c)
            if tr_c_mask.sum() < 200 or va_c_mask.sum() < 50:
                continue
            X_tr_c = X_train[tr][tr_c_mask]
            y_tr_c = y[tr][tr_c_mask]
            if y_tr_c.min() == y_tr_c.max():
                continue
            X_va_c = X_train[va][va_c_mask]
            X_te_c = X_test[te_c_mask]
            sc = StandardScaler()
            Xt = sc.fit_transform(X_tr_c)
            Xv = sc.transform(X_va_c)
            Xte = sc.transform(X_te_c) if te_c_mask.any() else None
            lr = LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs",
                max_iter=2000, class_weight="balanced",
            )
            lr.fit(Xt, y_tr_c)
            cmp_va_pred[va_c_mask] = lr.predict_proba(Xv)[:, 1]
            if Xte is not None:
                cmp_te_pred[te_c_mask] = lr.predict_proba(Xte)[:, 1]
            if fi == 0:
                # log per-Compound stats
                a_va = roc_auc_score(y[va][va_c_mask],
                                     cmp_va_pred[va_c_mask]) \
                    if (y[va][va_c_mask].min() < y[va][va_c_mask].max()) else float("nan")
                per_compound_stats.append({
                    "compound": c,
                    "n_train": int(tr_c_mask.sum()),
                    "n_val": int(va_c_mask.sum()),
                    "fold0_va_auc": round(float(a_va), 4),
                })
        oof[va] = cmp_va_pred
        test_pred += cmp_te_pred / 5
        fa = roc_auc_score(y[va], cmp_va_pred)
        fold_aucs.append(round(float(fa), 5))
        print(f"  fold {fi+1}/5 AUC={fa:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"\nA4 per-Compound OOF AUC: {auc:.5f}; folds: {fold_aucs}")
    print("\nPer-Compound stats (fold 0):")
    for s in per_compound_stats:
        print(f"  {s['compound']:<14s} n_tr={s['n_train']:>6d} "
              f"n_va={s['n_val']:>5d} va_AUC={s['fold0_va_auc']:.4f}")

    # Save artifacts
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_a4_per_compound_strat.npy", oof2)
    np.save(ART / "test_a4_per_compound_strat.npy", test2)

    # Compare standalone diversity
    primary_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    rho_prim, _ = spearmanr(oof, primary_oof)
    print(f"\nρ(A4_per_compound, PRIMARY) = {rho_prim:.5f}")

    # Min-meta gate vs K=10
    print("\nMin-meta gate vs K=10 ...")
    P_k10 = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K10_BASES])
    F_base = _expand(P_k10)
    _, auc_k10 = _meta_oof(F_base, y)
    P_with = np.column_stack([P_k10, oof.reshape(-1, 1)])
    F_with = _expand(P_with)
    _, auc_with = _meta_oof(F_with, y)
    delta_bp = (auc_with - auc_k10) * 1e4
    print(f"  K=10 baseline:        {auc_k10:.5f}")
    print(f"  K=10 + A4_per_compound: {auc_with:.5f}  Δ={delta_bp:+.2f} bp")

    # Also: combined K=10 + A2_rich + A4
    a2_r = _pos(ART / "oof_a2_rich_strat.npy")
    P_with2 = np.column_stack([P_k10, a2_r.reshape(-1, 1), oof.reshape(-1, 1)])
    F_with2 = _expand(P_with2)
    _, auc_with2 = _meta_oof(F_with2, y)
    delta_bp2 = (auc_with2 - auc_k10) * 1e4
    print(f"  K=10 + A2_rich + A4:  {auc_with2:.5f}  Δ={delta_bp2:+.2f} bp")

    out = {
        "n_features": int(X_train.shape[1]),
        "oof_auc": round(float(auc), 5),
        "fold_aucs": fold_aucs,
        "per_compound_fold0": per_compound_stats,
        "rho_vs_primary": round(float(rho_prim), 5),
        "k10_baseline": round(float(auc_k10), 6),
        "k10_plus_a4_auc": round(float(auc_with), 6),
        "k10_plus_a4_delta_bp": round(float(delta_bp), 3),
        "k10_plus_a2rich_a4_auc": round(float(auc_with2), 6),
        "k10_plus_a2rich_a4_delta_bp": round(float(delta_bp2), 3),
    }
    json_path = ART / "lr_diag_a4_per_segment.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
