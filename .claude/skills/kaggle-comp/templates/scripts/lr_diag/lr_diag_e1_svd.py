"""scripts/lr_diag_e1_svd.py — E1: effective rank of K=24 OOF matrix.

Diagnostic. Answers: is the K=22+ saturation a redundancy artifact?

Computes for both probability and logit representations:
  - singular value spectrum
  - effective rank via entropy: exp(H(p)) where p_i = σ_i² / Σ σ_j²
  - cumulative-variance ranks at {90, 95, 99, 99.9}%
  - stable rank ||A||_F² / ||A||_2²
  - condition number

Also outputs Spearman-rho top-pairs (which bases are most redundant)
and the residualized rank after removing the PRIMARY's component
(d13e Compound×Stint τ=20k baseline + h1d).

Output: scripts/artifacts/lr_diag_e1_svd.json + console summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
EXTRAS = [
    "d16_orig_continuous_only",
    "p1_single_cb_v3_gpu",
    "d17_h1d_yekenot_full",
]
ALL_BASES = K21_BASES + EXTRAS


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _eff_rank_entropy(s: np.ndarray) -> float:
    s2 = s ** 2
    p = s2 / s2.sum()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def _cumvar_rank(s: np.ndarray, frac: float) -> int:
    s2 = s ** 2
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, frac) + 1)


def _stable_rank(s: np.ndarray) -> float:
    return float((s ** 2).sum() / (s[0] ** 2))


def _spectrum(M: np.ndarray, label: str, names: list[str]) -> dict:
    # standardize columns to remove scale effects
    Mc = M - M.mean(axis=0)
    Mc = Mc / (Mc.std(axis=0) + 1e-12)
    s = np.linalg.svd(Mc, compute_uv=False)
    out = {
        "label": label,
        "shape": list(Mc.shape),
        "singular_values": [round(float(x), 4) for x in s],
        "eff_rank_entropy": round(_eff_rank_entropy(s), 3),
        "rank_90pct": _cumvar_rank(s, 0.90),
        "rank_95pct": _cumvar_rank(s, 0.95),
        "rank_99pct": _cumvar_rank(s, 0.99),
        "rank_999pct": _cumvar_rank(s, 0.999),
        "stable_rank": round(_stable_rank(s), 3),
        "condition_number": round(float(s[0] / max(s[-1], 1e-12)), 1),
        "var_top_5_pct": round(100 * (s[:5] ** 2).sum() / (s ** 2).sum(), 2),
        "var_top_10_pct": round(100 * (s[:10] ** 2).sum() / (s ** 2).sum(), 2),
    }
    return out


def _top_redundant_pairs(M: np.ndarray, names: list[str], k: int = 10) -> list[dict]:
    K = M.shape[1]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            r, _ = spearmanr(M[:, i], M[:, j])
            pairs.append((float(r), names[i], names[j]))
    pairs.sort(key=lambda x: -abs(x[0]))
    return [{"rho": round(r, 4), "a": a, "b": b} for r, a, b in pairs[:k]]


def main():
    # K=24 base OOFs
    mats_p = [_pos(ART / f"oof_{b}_strat.npy") for b in ALL_BASES]
    P = np.column_stack(mats_p)  # (N, 24) probabilities
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    L = np.log(P_clip / (1 - P_clip))  # logits

    # Spectrum analyses
    out = {
        "n_rows": P.shape[0],
        "n_bases": P.shape[1],
        "bases": ALL_BASES,
        "spec_prob": _spectrum(P, "probability", ALL_BASES),
        "spec_logit": _spectrum(L, "logit", ALL_BASES),
    }

    # Top redundant pairs (Spearman)
    out["top_redundant_pairs_logit"] = _top_redundant_pairs(L, ALL_BASES, k=15)

    # Residualized rank: regress each base on PRIMARY logit, then SVD residuals
    primary = _pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")  # for sanity (test space)
    # use d17_K24_d18pool_h1d as PRIMARY OOF anchor
    prim_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    prim_oof = np.clip(prim_oof, 1e-9, 1 - 1e-9)
    prim_log = np.log(prim_oof / (1 - prim_oof))
    R = np.zeros_like(L)
    a = prim_log - prim_log.mean()
    aa = float((a * a).sum())
    for j in range(L.shape[1]):
        b = L[:, j] - L[:, j].mean()
        beta = float((a * b).sum() / aa)
        R[:, j] = b - beta * a
    out["spec_residualized_logit"] = _spectrum(R, "residualized_logit_after_PRIMARY", ALL_BASES)

    # Reference: explained-variance interpretation
    s = out["spec_logit"]["singular_values"]
    out["interpretation"] = (
        f"24 bases; logit effective rank (entropy) = "
        f"{out['spec_logit']['eff_rank_entropy']}; "
        f"rank at 95% var = {out['spec_logit']['rank_95pct']}; "
        f"rank at 99% var = {out['spec_logit']['rank_99pct']}; "
        f"top-5 singular values explain "
        f"{out['spec_logit']['var_top_5_pct']}% of variance."
    )

    json_path = ART / "lr_diag_e1_svd.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== E1 SVD effective rank — K=24 OOF matrix ===")
    print(f"shape: {out['n_rows']} x {out['n_bases']}")
    print(f"\nLogit space (LR-meta sees this):")
    sl = out["spec_logit"]
    print(f"  eff_rank (entropy) : {sl['eff_rank_entropy']}")
    print(f"  rank @ 90% var     : {sl['rank_90pct']}")
    print(f"  rank @ 95% var     : {sl['rank_95pct']}")
    print(f"  rank @ 99% var     : {sl['rank_99pct']}")
    print(f"  rank @ 99.9% var   : {sl['rank_999pct']}")
    print(f"  stable rank        : {sl['stable_rank']}")
    print(f"  cond number        : {sl['condition_number']}")
    print(f"  top-5 var %        : {sl['var_top_5_pct']}")
    print(f"  top-10 var %       : {sl['var_top_10_pct']}")
    print(f"\nProbability space (raw OOF):")
    sp = out["spec_prob"]
    print(f"  eff_rank (entropy) : {sp['eff_rank_entropy']}")
    print(f"  rank @ 95% var     : {sp['rank_95pct']}")
    print(f"  rank @ 99% var     : {sp['rank_99pct']}")
    print(f"\nResidualized after PRIMARY (d17_K24_d18pool_h1d):")
    sr = out["spec_residualized_logit"]
    print(f"  eff_rank (entropy) : {sr['eff_rank_entropy']}")
    print(f"  rank @ 95% var     : {sr['rank_95pct']}")
    print(f"  rank @ 99% var     : {sr['rank_99pct']}")
    print(f"\nTop-15 most redundant pairs (Spearman ρ on logits):")
    for p in out["top_redundant_pairs_logit"]:
        print(f"  ρ={p['rho']:+.4f}  {p['a']:<32s}  {p['b']}")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
