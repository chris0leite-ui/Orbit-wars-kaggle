# FE recipe: Simple Logistic Regression — fast baseline + ceiling probe

> Origin: s6e5 (playground-series-s6e5, F1 PitNextLap binary AUC).
> Branch: `claude/logistic-regression-ensemble-0PNkA`.
> Verified s6e5 numbers: **`lr_kbins20_ohe` 0.92038 OOF in 22s** (the
> 30-second baseline) and **`lr_mega` 0.92776 OOF in 7m25s** (the LR
> ceiling). Closes 87% of the GBDT-vs-`lr_raw` gap.

## When to apply

- **First hour of any new tabular comp.** Before fitting GBDT, run the
  30-second LR baseline. The lift from `lr_raw_std` (raw numerics
  only) to `lr_kbins20_ohe` (KBins+OHE everything) tells you what
  fraction of the comp's signal is binnable univariate structure
  vs. higher-order interactions.
- **As a fast iteration vehicle.** When testing FE ideas, fit `lr_kbins20_ohe`-
  style baseline; LR converges in tens of seconds vs minutes for GBDT.
  The relative ranking of FE recipes transfers (mostly) to GBDT.
- **As a ceiling probe.** Run the mega recipe (~8 min CPU). If it lands
  >100 bp below your single-GBDT, **stacking is necessary** for this
  comp; LR is at best a diversity contributor. If it lands within 30
  bp, GBDT lift is mostly axis-aligned bin signal that LR also captures.
- **NOT as a primary model** (unless GBDT is unavailable or the comp
  rules forbid it). LR cannot recover non-linear interaction structure.

## The 30-second baseline (canonical LR recipe)

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse

NUM_COLS = [...]   # all your continuous + low-precision-int features
CAT_COLS = [...]   # all your categorical features
TARGET, SEED, N_FOLDS = "y", 42, 5

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = train[TARGET].astype(int).values

# 1. KBins(20, quantile, onehot) on every numeric — fit on combined
#    train+test (Rule 25-safe iff AV-AUC ~0.5; verify with adversarial-val
#    classifier first if unsure).
num_tr = train[NUM_COLS].fillna(0).values.astype(np.float32)
num_te = test[NUM_COLS].fillna(0).values.astype(np.float32)
kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile",
                      subsample=None)
kb.fit(np.vstack([num_tr, num_te]))
Xb_tr, Xb_te = kb.transform(num_tr), kb.transform(num_te)

# 2. OneHot on cats
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                    dtype=np.float32)
enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]]))
Oc_tr, Oc_te = enc.transform(train[CAT_COLS]), enc.transform(test[CAT_COLS])

# 3. Stack sparse, 5-fold CV LR with liblinear
Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
Xte = sparse.hstack([Xb_te, Oc_te], format="csr")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
for tr, va in skf.split(np.zeros(len(y)), y):
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
    lr.fit(Xtr[tr], y[tr])
    oof[va] = lr.predict_proba(Xtr[va])[:, 1]
    test_pred += lr.predict_proba(Xte)[:, 1] / N_FOLDS

print(f"OOF AUC {roc_auc_score(y, oof):.5f}")
```

**Why it works.** KBins-OHE turns each continuous feature into a piecewise-
constant function that LR can fit linearly. Functionally equivalent to
"every feature gets its own decision stump"; LR learns one weight per
bucket. This is "tree splits in a linear-model wrapper" — exactly the
structure GBDTs build via splits.

**Verified s6e5: 0.92038 OOF in 22 seconds.** Closes 88% of the
gap from `lr_raw_std` (0.825) to GBDT pool meta (0.954) on this comp.

## The mega recipe (LR ceiling)

When you need the absolute LR ceiling — typically as a "what's left
on the table?" diagnostic before committing to stacking:

1. **Static FE** (label-independent, fit once):
   - Tyre/algebra: x², log, sqrt, ratios (e.g. `Cumulative_Degradation /
     (TyreLife + 1)`).
   - Race-progress family: `est_total_laps`, `laps_remaining`,
     `is_pit_window` (domain thresholds), `position_pressure`, `urgency_score`.
   - Lag/rolling within group: `delta_lag1/2`, `roll{3,5,7,10,15}_lt`,
     `lap_vs_r{3,7,15}`, `roll3_std`.
   - Historical priors (when external data available): driver-historical
     pit-lap mean/std, circuit-historical pit-lap mean/std.
   - Combo cat factorize: `(Race, Compound)`, `(Race, Year)`, `(Driver, Compound)`.
2. **Per-fold label-conditional aggregates (`fit_fs_a`):** mean/max
   aggregates conditioned on `y=1` rows of the FOLD-TRAIN ONLY.
   See `scripts/p1_features.py::fit_fs_a` for the canonical pattern.
   **Rule 24 critical:** never fit on full-train-with-labels.
3. **6 CV target encoders** at varied keys + smoothing, fit per-fold:
   `(Driver, Race, Year)` α=20, `(Driver, Race)` α=30, `(Race, Compound)`
   α=25, `(Driver, Compound)` α=25, `(Race, Year)` α=20, `(Driver, Race,
   Compound)` α=15.
4. **3-way TE sweep** (4 keys × 4 smoothings = 16 features): same keys
   as above, smoothings ∈ {1, 5, 20, 100}.
5. **DGP rule lookups** (4 rule-keys × 4 alphas = 16 features):
   Bayesian-smoothed `P(y=1 | rule_key)` at α ∈ {5, 20, 100, 500} for
   key combos like `(Compound, Stint)`, `(Driver, Compound)`,
   `(Year, Race)`, `(Compound, TyreLife_decile)`.
6. **KBins(20, quantile) + OHE** on every numeric (the 30-second
   baseline component).
7. **Cat OHE** on raw categorical columns.
8. **Densify** (~1200 cols × N_train rows; ~1.7 GB at float32 for 350k
   rows — fits in 16 GB RAM).
9. **StandardScaler** the dense block.
10. **LR** with `C=1.0, solver='lbfgs', max_iter=2000`. lbfgs is
    BLAS-multi-threaded; on dense it's much faster than liblinear/saga.

**Verified s6e5: 0.92776 OOF in 7m25s CPU.** That's the simple-LR
ceiling on this comp.

## Per-fold mechanics (fold-safe template)

Crucial pattern to keep all label-conditional FE leak-free:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_list = list(skf.split(np.zeros(len(y)), y))

# Compute label-free static features ONCE
train_S, test_S, state = make_features_static(train), ...

# Pre-compute CV TE features (fold_list-aware, leak-free)
te_oof, te_test = {}, {}
for cols, smooth, name in TE_CONFIGS:
    te_oof[name], te_test[name] = cv_target_encode(
        train, test, cols, train[TARGET], fold_list, smooth)

oof = np.zeros(len(y), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
for k, (tr, va) in enumerate(fold_list):
    # Per-fold LABEL-CONDITIONAL aggregates (FS_A): tr-rows ONLY
    fs_a = fit_fs_a(train.iloc[tr])
    train_A = apply_fs_a(train_S, fs_a)
    test_A = apply_fs_a(test_S, fs_a)

    # Per-fold DGP rule lookups: tr-rows ONLY
    rule_tr, rule_va, rule_te = build_dgp_rule_features(
        train, test, y, tr, va)

    # Stack: static + 6-CV-TE + 3-way-TE + rule + KBins + cat-OHE
    feat_tr = np.hstack([train_A.iloc[tr][feat_cols],
                         te_train_arr[tr], threeway_arr[tr], rule_tr])
    feat_va = np.hstack([train_A.iloc[va][feat_cols],
                         te_train_arr[va], threeway_arr[va], rule_va])
    feat_te = np.hstack([test_A[feat_cols],
                         te_test_arr, threeway_test_arr, rule_te])

    # Standardize per-fold (on tr-rows only); densify KBins+OHE
    sc = StandardScaler().fit(feat_tr)
    feat_tr, feat_va, feat_te = sc.transform(feat_tr), sc.transform(feat_va), sc.transform(feat_te)
    Xtr_tr = np.hstack([feat_tr, Bk_tr[tr].toarray(), Oc_tr[tr].toarray()])
    Xtr_va = np.hstack([feat_va, Bk_tr[va].toarray(), Oc_tr[va].toarray()])
    Xte_s  = np.hstack([feat_te, Bk_te.toarray(),     Oc_te.toarray()])

    # Fit LR
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    lr.fit(Xtr_tr, y[tr])
    oof[va] = lr.predict_proba(Xtr_va)[:, 1]
    test_pred += lr.predict_proba(Xte_s)[:, 1] / N_FOLDS
```

## Mechanism map — when each FE family helps which model

| FE family | LR | GBDT | NN |
|---|---|---|---|
| Raw numerics + StandardScaler | ✓ floor | ✓ baseline | ✓ baseline |
| Polynomial expansion (deg 2) | ✓ +50 bp over raw | ≈0 (trees split already) | ≈0 (activations) |
| **KBins(15-25, quantile) + OHE on every numeric** | **★ best (+95 bp)** | small | small |
| KBins on only some numerics (yekenot) | ✗ leaves nonlinear gaps | +20 bp on CB v4 (s6e5 d17) | designed for NN ✓ |
| OneHot on cats | ✓ +10 bp | ≈0 (handled natively) | ✓ via embeddings |
| TE single key + smoothing | ✓ +20 bp | small lift | small lift |
| **3-way CV target encoding** | ✓ ~+5 bp | comp's documented +200 bp on LGBM | ≈ |
| Tree-engineered (rolling means, lag deltas, position pressure) | ✗ −62 bp on LR (!) | ✓ designed for | ≈0 |
| DGP rule lookups (Bayesian smoothed) | ✓ ~+80 bp over raw | ≈0 (trees rediscover in splits) | ≈ |
| Splines (B-splines, 5 knots, quantile) | ✓ +94 bp over raw | small | small |

**Reading.** *A recipe that wins for one model class can hurt another.*
For LR, **bin every continuous feature**. For GBDT, **engineer
ratios and TE**. For NN, **scale + add embeddings + carefully chosen
ratios**.

## Anti-patterns and pitfalls (s6e5 lessons)

- **Don't run LR variants of the same recipe family.** L1 vs L2,
  `class_weight='balanced'` vs default, C-sweep — these are rank-no-ops
  on AUC (ρ ≥ 0.997 across them). Pick one variant per FE family;
  use the rest of the budget on different families. (s6e5 friction
  `class-weight-balanced-rank-no-op-on-binary-auc`.)
- **Don't pass tree-engineered FE to LR.** 50 Rozen-style features
  (rolling means, lap-in-stint, position pressure, etc.) gave LR
  AUC 0.857 — *lower* than the 14-feature raw+TE baseline 0.845.
  Tree-friendly FE actively hurts LR; binning is the LR-friendly way.
  (s6e5 friction `tree-friendly-fe-hurts-lr`.)
- **Don't apply yekenot's KBins-on-2-of-11 strategy to LR.** That's
  NN-targeted (NN learns the rest from raw scaling). LR drops to 0.859
  vs full KBins 0.920. (s6e5 friction `yekenot-recipe-is-NN-specific-for-LR`.)
- **Don't expect rich-FE LR to add diversity to a GBDT pool.** Adding
  the 0.928 mega LR to a GBDT-K24 pool: combined eff_rank stays at
  3.33 (zero gain). Rich FE makes LR *stronger as a single model* but
  the lift projects onto direction-1 the simple LRs already covered.
  (s6e5 friction `lr-meta-rank-lock-strong-anchor`, 6× cross-confirmed.)
- **Don't forget Rule 25 (transductive features need AV check).**
  Combined-set transforms (KBins, hash, factorize maps fit on
  train+test) are only safe if adversarial-validation AUC ≈ 0.5. On
  s6e5 AV-AUC=0.502, so combined-fit was safe. **Run AV first if
  unsure.**

## Calibration ladder (s6e5, verified)

| Recipe | OOF AUC | Time | n_feats |
|---|---:|---:|---:|
| `lr_raw_std` | 0.82467 | 3 s | 11 |
| `lr_raw_te` (raw + 4 TE features, fold-safe) | 0.84528 | 5 s | 14 |
| `lr_raw_ohe` (raw + cat OHE) | 0.85407 | 19 s | 929 |
| `lr_rozen_full` (Rozen LGBM-style FE — **anti-recipe**) | 0.85735 | 46 s | 93 |
| `lr_kbins_yekenot` (yekenot KBins on 2 of 11 — **anti-recipe for LR**) | 0.85948 | 148 s | 1122 |
| `lr_poly2_std` (degree-2 polynomial of 11 numerics) | 0.88244 | 119 s | 77 |
| `lr_te_3way_sweep` (3-way TE × 4 smoothings + raw + cat OHE) | 0.90403 | 89 s | ~150 |
| `lr_dgp_rules` (4-key × 4-α DGP lookups + cat OHE) | 0.90714 | 106 s | 945 |
| `lr_kbins5_ohe` | 0.91082 | 22 s | 964 |
| `lr_yekenot_full_recipe` | 0.91860 | 46 s | **45** |
| **`lr_kbins20_ohe`** ★ 30-second LR baseline | **0.92038** | 22 s | 1077 |
| **`lr_mega`** ★ LR ceiling | **0.92776** | 7m25s | 1202 |
| GBDT single (CatBoost v4) | 0.95200 | 30 min GPU | – |
| GBDT pool LR-meta (K=24) | 0.95385 | 50 s | – |

LR closes 87% of the GBDT-vs-`lr_raw` gap with mega. The remaining
26 bp is non-linear interaction structure.

## Friction tags this recipe interacts with

- `kbins-ohe-emulates-tree-splits-in-linear-models` — the structural
  reason KBins+OHE is the strongest LR recipe.
- `tree-friendly-fe-hurts-lr` — Rozen-style FE designed for LGBM
  *actively hurts* LR (s6e5 round 2 evidence).
- `lr-meta-rank-lock-strong-anchor` — rich-FE LR lifts the single LR
  but does NOT add new directions to a saturated GBDT pool.
- `bank-diversity-needs-new-model-class-not-new-FE` — refined: holds
  for *diversity*, not for single-model strength.
- `target-construction-layer-leakage` (Rule 24) — fold-safe label-
  conditional FE is mandatory; per-fold FS_A and TE.
- `transductive-features-need-AV-check` (Rule 25) — combined-set
  KBins/OHE/hash require AV-AUC ≈ 0.5 to be leak-free.
- `random-subspace-LR-reduces-eff-rank` — bagging LR over random feature
  subsets *destroys* diversity. 10 bags of 30%-subset on s6e5 mega gave
  bag eff_rank 1.67 (vs 2.19 for 20 hand-designed FE variants). **Skip
  this; do not budget time for it.**
- `per-segment-LR-on-DGP-partition-finds-new-signal` — *highest-EV LR
  diversity move on s6e5*. Per-(Compound × Year) mega LR gave +60.8 bp
  over global mega (0.92776 → 0.93385). Pattern: 2023-cohort cells lift
  most (+725 to +1081 bp), small cells regress. Mechanism: pooled-coef
  LR can't represent cohort-conditional DGP shifts.
- `mega-oof-as-gbdt-feature-causes-overfit` — adding the mega LR's
  fold-safe OOF as a single feature to a fresh LGBM on raw caused
  −36.7 bp regression. Mega is meta-level useful, not feature-level.
- `gbdt-meta-not-better-than-lr-meta-on-saturated-pool` — non-linear
  meta does NOT unlock the residual-after-PRIMARY directions. On s6e5
  K=24+mega: LR-meta 0.95387, LightGBM-meta 0.95378 (Δ −0.89 bp).
  Rank-lock is structural to the OOF correlation pattern, not solver-
  specific.

## Diagnostic on the LR-bank's effective rank

| Pool | n cols | eff_rank (entropy) |
|---|---:|---:|
| Round 1 LR-bank (15 FE-variant LRs) | 15 | 2.00 |
| Round 2 LR-bank (20 incl. rich FE + mega) | 20 | 2.19 |
| GBDT-K24 pool (Arc-A E1) | 24 | 2.882 |
| **GBDT+LR combined** | 39-44 | **3.33** (no change between rounds) |
| GBDT+LR residualized after PRIMARY removal | 39-44 | 13.56 → 14.05 (+0.5) |

20 distinct LR FE recipes collapse to ~2 effective signal directions.
Adding rich FE to LR boosts single-base AUC by 73 bp but does NOT
increase combined eff_rank with GBDT. **LR is concentrated, not
orthogonal, to GBDT.**
