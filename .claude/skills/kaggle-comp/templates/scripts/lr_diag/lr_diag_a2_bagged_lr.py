"""scripts/lr_diag_a2_bagged_lr.py — A2: Bagged-LR base, vanilla + rich.

Arc C, A2. New LR-population base for the K=10 core.

Two variants:
  - vanilla : E5-style features (11 numeric + Compound + Race
              dummies + Driver-freq + 4 cheap interactions)
  - rich    : vanilla + the 8 top-E6 Stint-cross interactions
              (Stint × {RaceProgress, Year, LapNumber, TyreLife,
              LapTime, LapTime_Delta, Cum_Degradation, Position}
              + LapTime × LapTime_Delta).

Sharp test: does adding E6's identified interactions push LR from
0.85 to GBDT range? Quantifies "linear-but-with-interactions".

Pipeline:
  - 5-fold StratKF; within each fold, fit 20 LR-bags on 50%
    bootstraps; avg predicted probas → OOF column.
  - All LRs: lbfgs L2 C=1.0, class_weight='balanced'.
  - StandardScaler fit on train rows of fold.

Output:
  - oof_a2_vanilla_strat.npy  + test_*
  - oof_a2_rich_strat.npy     + test_*
  - lr_diag_a2_bagged_lr.json (OOF AUC, fold AUCs,
    coef-distribution per feature)
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
N_BAGS = 20
SEED = 42


def build_features_basic(df_train, df_test):
    """Vanilla feature set ~ E5."""
    num_cols = [c for c in df_train.columns
                if c not in ["Driver", "Compound", "Race", TARGET, "id"]
                and pd.api.types.is_numeric_dtype(df_train[c])]
    df = pd.concat([df_train.assign(__split="tr"),
                    df_test.assign(__split="te")], ignore_index=True)

    X_num = df[num_cols].values.astype(np.float64)
    # Compound one-hot
    comp = pd.get_dummies(df["Compound"], prefix="Cmp", dtype=np.float64).values
    # Race one-hot
    race = pd.get_dummies(df["Race"], prefix="Race", dtype=np.float64).values
    # Driver-freq: based on TRAIN frequencies only (avoid test contamination
    # of frequency table; train and test should share Driver levels but
    # frequencies should reflect train distribution)
    drv_counts = df_train["Driver"].value_counts()
    drv_freq = df["Driver"].map(drv_counts).fillna(0).values.reshape(-1, 1)
    drv_freq = drv_freq.astype(np.float64)
    drv_freq = (drv_freq - drv_freq.mean()) / (drv_freq.std() + 1e-12)

    X_basic = np.hstack([X_num, comp, race, drv_freq])
    names_basic = (list(num_cols) +
                   list(pd.get_dummies(df["Compound"], prefix="Cmp").columns) +
                   list(pd.get_dummies(df["Race"], prefix="Race").columns) +
                   ["Driver_freq"])

    # Cheap interactions (4): used in both vanilla and rich
    nm_idx = {n: i for i, n in enumerate(num_cols)}
    cheap_pairs = [
        ("TyreLife", "Stint"),
        ("LapNumber", "Position"),
        ("Cumulative_Degradation", "TyreLife"),
        ("Position", "LapNumber"),
    ]
    chea = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]]
        for a, b in cheap_pairs if a in nm_idx and b in nm_idx
    ])
    cheap_names = [f"int_{a}_x_{b}" for a, b in cheap_pairs
                   if a in nm_idx and b in nm_idx]

    X_vanilla = np.hstack([X_basic, chea])
    names_vanilla = names_basic + cheap_names

    # Rich: add E6's top Stint-cross + LapTime×LapTime_Delta
    rich_pairs = [
        ("Stint", "RaceProgress"),
        ("Stint", "Year"),
        ("Stint", "LapNumber"),
        ("Stint", "TyreLife"),
        ("Stint", "LapTime (s)"),
        ("Stint", "LapTime_Delta"),
        ("Stint", "Cumulative_Degradation"),
        ("Stint", "Position"),
        ("LapTime (s)", "LapTime_Delta"),
    ]
    rich = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]]
        for a, b in rich_pairs if a in nm_idx and b in nm_idx
    ])
    rich_names = [f"int_{a}_x_{b}" for a, b in rich_pairs
                  if a in nm_idx and b in nm_idx]
    X_rich = np.hstack([X_vanilla, rich])
    names_rich = names_vanilla + rich_names

    sp = df["__split"].values
    return (X_vanilla[sp == "tr"], X_vanilla[sp == "te"],
            X_rich[sp == "tr"], X_rich[sp == "te"],
            names_vanilla, names_rich)


def fit_bagged_lr(X_tr, y_tr, X_va, X_te, n_bags=N_BAGS, seed=SEED):
    """Fit n_bags LRs on 50% bootstraps; return mean val + test probs;
    plus list of coefficient vectors (one per bag)."""
    rng = np.random.default_rng(seed)
    n = len(y_tr)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)
    p_va = np.zeros(X_va.shape[0])
    p_te = np.zeros(X_te.shape[0])
    coefs = []
    for b in range(n_bags):
        idx = rng.choice(n, size=n // 2, replace=True)
        lr = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced",
        )
        lr.fit(X_tr_s[idx], y_tr[idx])
        p_va += lr.predict_proba(X_va_s)[:, 1]
        p_te += lr.predict_proba(X_te_s)[:, 1]
        coefs.append(lr.coef_[0].astype(np.float64))
    return p_va / n_bags, p_te / n_bags, np.vstack(coefs)


def run_variant(name, X_train, X_test, y, df_test_n, names):
    print(f"\n=== A2 variant: {name} | features: {X_train.shape[1]} ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    p_te_acc = np.zeros(X_test.shape[0])
    fold_aucs = []
    coefs_all = []
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        p_va, p_te, coefs = fit_bagged_lr(
            X_train[tr], y[tr], X_train[va], X_test, seed=SEED + fi
        )
        oof[va] = p_va
        p_te_acc += p_te / 5
        a = roc_auc_score(y[va], p_va)
        fold_aucs.append(round(float(a), 5))
        coefs_all.append(coefs)
        print(f"  fold {fi+1}/5 AUC={a:.5f}", flush=True)
    auc = roc_auc_score(y, oof)
    coefs_all = np.vstack(coefs_all)  # (5*N_BAGS, K)
    # per-feature stability
    stab = []
    for k in range(coefs_all.shape[1]):
        c = coefs_all[:, k]
        stab.append({
            "feature": names[k],
            "coef_mean": round(float(np.mean(c)), 4),
            "coef_std": round(float(np.std(c)), 4),
            "snr": round(float(abs(np.mean(c)) / (np.std(c) + 1e-9)), 2),
            "sign_flip_rate": round(float(np.mean(
                np.sign(c) != np.sign(np.median(c))
            )), 3),
        })
    stab.sort(key=lambda r: -r["snr"])
    print(f"  OOF AUC: {auc:.5f}; folds: {fold_aucs}")
    # save artifacts
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - p_te_acc, p_te_acc])
    np.save(ART / f"oof_a2_{name}_strat.npy", oof2)
    np.save(ART / f"test_a2_{name}_strat.npy", test2)
    return {
        "name": name,
        "n_features": int(X_train.shape[1]),
        "oof_auc": round(float(auc), 5),
        "fold_aucs": fold_aucs,
        "stability_top10": stab[:10],
        "stability_full": stab,
    }


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    Xv_tr, Xv_te, Xr_tr, Xr_te, names_v, names_r = build_features_basic(
        train, test
    )
    print(f"Train: {Xv_tr.shape}; Test: {Xv_te.shape}")
    print(f"Vanilla features: {len(names_v)}; Rich features: {len(names_r)}")

    out = {"variants": {}}
    for name, Xtr, Xte, names in [
        ("vanilla", Xv_tr, Xv_te, names_v),
        ("rich", Xr_tr, Xr_te, names_r),
    ]:
        out["variants"][name] = run_variant(name, Xtr, Xte, y, len(test), names)

    # interaction-feature stability summary (rich only)
    int_features_top = [r for r in out["variants"]["rich"]["stability_full"]
                        if r["feature"].startswith("int_Stint_x")
                        or r["feature"] == "int_LapTime (s)_x_LapTime_Delta"]
    out["rich_interaction_stability"] = sorted(int_features_top,
                                                key=lambda r: -r["snr"])

    json_path = ART / "lr_diag_a2_bagged_lr.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== A2 summary ===")
    print(f"vanilla OOF AUC: {out['variants']['vanilla']['oof_auc']}")
    print(f"rich    OOF AUC: {out['variants']['rich']['oof_auc']}")
    delta_vs_v = (out['variants']['rich']['oof_auc']
                  - out['variants']['vanilla']['oof_auc']) * 1e4
    print(f"rich − vanilla:  {delta_vs_v:+.1f} bp (= effect of E6's 9 "
          "Stint-cross interactions)")
    print(f"\nE6-derived interaction stability in rich variant:")
    for r in out["rich_interaction_stability"]:
        print(f"  {r['feature']:<40s} mean={r['coef_mean']:+.4f} "
              f"std={r['coef_std']:.4f} SNR={r['snr']:.1f} "
              f"flip={r['sign_flip_rate']:.2f}")
    print(f"\n→ JSON saved: {json_path}")
    print(f"→ OOF/test artifacts saved as oof_a2_{{vanilla,rich}}_strat.npy")


if __name__ == "__main__":
    main()
