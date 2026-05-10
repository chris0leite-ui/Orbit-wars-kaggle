"""scripts/lr_diag_a2_gate.py — min-meta gate of A2 (vanilla, rich) on K=10 core.

Per Arc B / E9, the true effective pool is K=10:
  d17_h1d_yekenot_full, p1_single_cb_v3_gpu, f1_hgbc_deep,
  d16_orig_continuous_only, b_lapsuntilpit, baseline_two_anchor,
  d9_R6_next_compound, cb_year-cat, e5_optuna_lgbm, d9f_FM_A.

Probe: does A2_vanilla or A2_rich add when stacked into K=10?
Reports OOF AUC of K=10 baseline, K=10 + A2_vanilla, K=10 + A2_rich,
plus K=10 + both. Spearman ρ to PRIMARY = d17_K24_d18pool_h1d.
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000)
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    P_k10 = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K10_BASES])

    a2_v = _pos(ART / "oof_a2_vanilla_strat.npy")
    a2_r = _pos(ART / "oof_a2_rich_strat.npy")

    primary_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    primary_auc = roc_auc_score(y, primary_oof)
    print(f"PRIMARY (d17_K24_d18pool_h1d) OOF AUC: {primary_auc:.5f}")

    # Standalone A2 AUCs and ρ
    a2_v_auc = roc_auc_score(y, a2_v)
    a2_r_auc = roc_auc_score(y, a2_r)
    rho_v_prim, _ = spearmanr(a2_v, primary_oof)
    rho_r_prim, _ = spearmanr(a2_r, primary_oof)
    rho_vr, _ = spearmanr(a2_v, a2_r)
    print(f"\nA2 standalone:")
    print(f"  vanilla OOF AUC: {a2_v_auc:.5f}  ρ vs PRIMARY: {rho_v_prim:.5f}")
    print(f"  rich    OOF AUC: {a2_r_auc:.5f}  ρ vs PRIMARY: {rho_r_prim:.5f}")
    print(f"  ρ(vanilla, rich): {rho_vr:.5f}")

    # K=10 baseline
    F_base = _expand(P_k10)
    oof_base, auc_base = _meta_oof(F_base, y)
    print(f"\nK=10 LR-meta baseline OOF AUC: {auc_base:.5f}")

    results = {
        "k10_auc": round(float(auc_base), 6),
        "primary_auc": round(float(primary_auc), 6),
        "a2_vanilla_standalone": {
            "auc": round(float(a2_v_auc), 6),
            "rho_vs_primary": round(float(rho_v_prim), 6),
        },
        "a2_rich_standalone": {
            "auc": round(float(a2_r_auc), 6),
            "rho_vs_primary": round(float(rho_r_prim), 6),
            "rho_vs_vanilla": round(float(rho_vr), 6),
        },
        "configs": [],
    }

    configs = [
        ("k10_plus_vanilla", [a2_v]),
        ("k10_plus_rich", [a2_r]),
        ("k10_plus_both", [a2_v, a2_r]),
    ]
    for name, extra in configs:
        P_with = np.column_stack([P_k10] + [c.reshape(-1, 1) for c in extra])
        F_with = _expand(P_with)
        oof_w, auc_w = _meta_oof(F_with, y)
        rho_p, _ = spearmanr(oof_w, primary_oof)
        delta_bp = (auc_w - auc_base) * 1e4
        results["configs"].append({
            "name": name,
            "k": int(P_with.shape[1]),
            "oof_auc": round(float(auc_w), 6),
            "delta_vs_k10_bp": round(float(delta_bp), 3),
            "rho_vs_primary": round(float(rho_p), 6),
        })
        print(f"  {name}: OOF={auc_w:.5f}  Δ={delta_bp:+.2f}bp  "
              f"ρ_PRIM={rho_p:.5f}")

    # Also: A2 rich INSTEAD OF d9f_FM_A (FM_A was the K=10 weakest base)
    # K=10 - FM_A + A2_rich
    P_swap = np.column_stack([
        P_k10[:, :-1],  # drop d9f_FM_A (last in K10_BASES)
        a2_r.reshape(-1, 1),
    ])
    F_swap = _expand(P_swap)
    oof_s, auc_s = _meta_oof(F_swap, y)
    delta_bp_s = (auc_s - auc_base) * 1e4
    results["k9_swap_FMA_for_a2_rich"] = {
        "oof_auc": round(float(auc_s), 6),
        "delta_vs_k10_bp": round(float(delta_bp_s), 3),
    }
    print(f"  k10_swap_FMA→a2_rich: OOF={auc_s:.5f}  Δ={delta_bp_s:+.2f}bp")

    json_path = ART / "lr_diag_a2_gate.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
