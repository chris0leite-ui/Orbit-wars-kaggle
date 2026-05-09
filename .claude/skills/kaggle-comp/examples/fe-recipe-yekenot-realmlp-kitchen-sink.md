# FE recipe: yekenot RealMLP-class kitchen-sink (CV-TE + floor-cat + count + KBins + orig)

> Origin: s6e5 (playground-series-s6e5, F1 PitNextLap binary AUC).
> External author: kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit.
> Verified: standalone fold-0 AUC 0.95366 vs default-config RealMLP 0.94454
> on s6e5 (+91 bp same fold). Closed the +69 bp recipe gap diagnosed
> after H1 v1/v2/v3 misdiagnosis.

## When to apply

- Synthetic-tabular Playground series with AUC metric.
- A RealMLP-class NN base in your pool that is plateau'd at standalone
  OOF well below the public LB top.
- The published "single-NN" or "single-LGBM" OOFs from public notebooks
  are 50-100 bp above your equivalent base — strong signal that the
  recipe is incomplete, not the architecture.
- Engineered-cat target encoding (TE) is not yet in your pool, and the
  comp has clear 2-way interaction structure (e.g. `Race × Compound`,
  `Race × Year`).

## Six load-bearing items (do all six; missing any one degrades by ~10-50 bp)

### 1. Arithmetic ratio features

```python
df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
```

Two carefully-chosen ratios. NN sees these as continuous; GBDT can derive
them but NN cannot. Generalise: pick 2-3 domain-meaningful ratios per comp.

### 2. Floor-based numeric → categorical

```python
for col in num_cols + ratio_names:
    cat_name = f"{col[1:]}_cat_" if col.startswith("_") else f"{col}_cat_"
    if fit:
        codes, uniques = np.floor(df[col]).factorize()
        category_map[col] = uniques
    else:
        uniques = category_map[col]
        codes = np.floor(df[col]).map({c: i for i, c in enumerate(uniques)}).fillna(-1).astype("int32")
    df[cat_name] = codes.astype(str)
```

Discretizes every numeric (and ratio) by integer floor + factorize. Gives
the NN coarse-grained position information without losing the original
continuous channel. Categorical view is one cheap, ordinal-encoded
column per numeric.

### 3. Count encoding on every categorical

```python
for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
    count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
    if fit:
        category_map[count_name] = df[col].value_counts()
    df[count_name] = df[col].map(category_map[count_name]).fillna(0).astype("int32")
```

Frequency feature — how rare is this driver/race/year? NN feature column.

### 4. KBins discretization on a few key numerics

```python
bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
for col, bins_list in bin_config.items():
    for n_bins in bins_list:
        bin_name = f"{col}_{n_bins}_quantile_bin_"
        if fit:
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                   strategy="quantile", subsample=None)
            kb.fit(df[[col]])
            category_map[bin_name] = kb
        df[bin_name] = category_map[bin_name].transform(df[[col]]).ravel().astype(str)
```

Yekenot used n_bins=200 on `RaceProgress` (fine grain) and n_bins=7 on
`LapTime (s)` (coarse grain). Pick by domain — high-density continuous
features get fine bins, low-precision continuous get coarse.

### 5. Combo cats (interaction categories)

```python
important_combos = [("Race", "Compound"), ("Race", "Year")]
combo_names = []
for cols in important_combos:
    name = "_".join(cols) + "_"
    series = df[cols[0]].astype(str)
    for c in cols[1:]:
        series = series + "_" + df[c].astype(str)
    if fit:
        codes, uniques = pd.factorize(series, sort=False)
        category_map[name] = uniques
    df[name] = codes.astype(str)
    combo_names.append(name)
```

Pick 2-3 high-variance 2-way combos (NOT 3-way; Rozen's `(Driver, Race,
Year)` 3-way is the classic single-LGBM trick but yekenot's 2-way is
better for NN+TE).

### 6. CV TargetEncoder on combo cats — INSIDE each fold loop ⭐ LOAD-BEARING

```python
for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig)), 1):
    X_tr = pd.concat([X.iloc[tr_idx], orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
    y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
    X_val = X.iloc[val_idx]; y_val = y.iloc[val_idx]
    X_tst = X_test.copy()

    TE = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True, random_state=SEED)
    tr_enc  = TE.fit_transform(X_tr[combo_names], y_tr)
    val_enc = TE.transform(X_val[combo_names])
    tst_enc = TE.transform(X_tst[combo_names])
    te_names = [f"_{c}TE" for c in combo_names]
    X_tr[te_names] = tr_enc; X_val[te_names] = val_enc; X_tst[te_names] = tst_enc

    model = RealMLP_TD_Classifier(**params)
    model.fit(X_tr, y_tr, X_val, y_val)
```

**Critical fold-safety detail (Rule 24):** `TargetEncoder(cv=N_FOLDS,
shuffle=True)` does internal cross-fitting on `X_tr` to produce
target-encoded features for the rows IN `X_tr` (train+orig union for the
fold) without each row seeing its own label. The transform on `X_val`
and `X_tst` uses the FULL-X_tr-fit map (no val-row labels). This is the
canonical sklearn CV-TE pattern; it does NOT leak.

Per-fold orig concat (and orig is also 5-fold split — using only
`orig_tr_fold` for training preserves stratified-orig + train fold
parity, but the ratio is roughly 4/5 of orig per outer fold).

## RealMLP_TD hyperparameters (yekenot's exact, cell 8 of his notebook)

```python
params = dict(
    random_state=42, verbosity=1,
    val_metric_name="1-auc_ovr",
    n_ens=24, n_epochs=6, batch_size=256,
    use_early_stopping=False,
    lr=0.03, wd=0.018, sq_mom=0.98,
    lr_sched="lin_cos_log_15",
    first_layer_lr_factor=0.25,
    embedding_size=6, max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128],
    act="silu", p_drop=0.05, p_drop_sched="expm4t",
    plr_hidden_1=16, plr_hidden_2=8,
    plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos",
    add_front_scale=False,
    bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale",
          "smooth_clip", "embedding", "l2_normalize"],
)
```

## Compute scaling guidance

| n_ens | n_epochs | wall (5-fold StratKF, 70k val/350k train per fold + 100k orig) | s6e5 verified |
|---:|---:|---|---|
| 1 | 2 | ~2 min/fold (smoke only, AUC well below) | n/a |
| 3 | 6 | ~4 min/fold (~17 min total no orig) | 0.94516 OOF (no FE / no TE) |
| 4 | 6 | ~7 min/fold (~35 min total with orig + full FE + TE) | **0.95366 fold-0 with full recipe** |
| 8 | 6 | ~14 min/fold (~70 min total) | unverified, projected ~0.953-0.954 |
| 24 | 6 | ~42 min/fold (~3.5 h total CPU; yekenot ran on Kaggle GPU) | 0.95273 yekenot pub |

The FE pipeline (items 1-6) is far more load-bearing than n_ens. Plan
`n_ens=4` as the floor, scale up if EV justifies.

## Anti-patterns and pitfalls (s6e5 lessons)

- **Don't replicate hyperparameters alone.** "Recipe gap" when a
  published author scores +50-70 bp above your equivalent arch is
  almost always a multi-item FE pipeline gap, NOT just hyperparameters.
  Read the FULL notebook source before BOTE; otherwise you'll
  misdiagnose (s6e5 friction
  `recipe-gap-misdiagnosis-when-public-author-FE-not-fully-replicated`).
- **Don't merge orig into train without the FE pipeline.** Without TE
  + count encoding + floor-cat, orig-merge alone HURTS by 30-40 bp/fold
  (s6e5 H1 V3 evidence). The full FE pipeline is what makes orig
  augmentation work.
- **`torch.set_num_interop_threads`** can be called only once per
  process — set at script init, not per fold (s6e5 H1d v1 crash at
  fold 2).
- **Don't trust 3-way TE alone.** Rozen's `(Driver, Race, Year)` 3-way
  TE is the famous +200 bp single-LGBM trick but is leak-prone (s6e5
  Day-17 P1 thesis falsified). Yekenot's 2-way `(Race, Compound)` +
  `(Race, Year)` is more robust at sklearn TargetEncoder cv=5.

## Friction tags this recipe interacts with

- `target-construction-layer-leakage` — Rule 24 strict-OOF audit.
  Yekenot's pipeline IS strict-OOF (TE inside fold, orig is independent
  samples not target-aggregated). Pass.
- `synthetic-augmented-driver-codes-cap-external-data-coverage` — orig
  has 100% real-driver TLAs; train.csv is 60% synthetic D### codes.
  Joining orig as additional supervised samples (this recipe) ≠ joining
  external telemetry on Driver key. The FE pipeline + orig-merge bypass
  the 1.4% match-rate cap H2 hit.

## Calibration outcome (s6e5) — VALIDATED

- PI sealed prediction full-recipe LB Δ: **+10 bp**
- Agent BOTE full-recipe LB Δ: **+5 to +8 bp**
- **Verified 5-fold StratKF OOF AUC: 0.95257** (vs yekenot pub 0.95273,
  gap −1.6 bp = within fold variance; vs default-config realmlp
  baseline 0.94582 = **+675 bp standalone**).
- Per-fold AUCs: 0.95366, 0.95153, 0.95232, 0.95189, 0.95375.
- **K=22 ADD (canonical LR-meta) OOF 0.95355** = +28.16 bp vs K=21
  baseline (0.95073), +23.4 bp vs current d16 PRIMARY OOF (0.951208).
- ρ_test vs d16 PRIMARY: **0.97180** (single base) / **0.98728** (K=22
  stack) — first base to break the 5-month ρ < 0.99 plateau.
- Predicted LB at K=22 ADD via probe.py band: **+18.4 bp** over
  PRIMARY 0.95089 → projected **0.95273** (gap to top-5% closes from
  −25.6 bp to −7.2 bp).
- Yekenot's notebook is now flagged as VALIDATED at
  `external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md`.
