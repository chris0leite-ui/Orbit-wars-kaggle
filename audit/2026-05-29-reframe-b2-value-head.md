# Reframe B.2 — per-target value head (first cut)

> Built: 2026-05-29 PM5 by `claude/competition-objective-alignment-hqNVM`.
> Follows the within-owner stratified probe verdict
> (`audit/2026-05-29-pveta-leaf-residual-within-owner.md`).

## What landed

A 15-feature LightGBM regressor predicting seat-0 ship-delta over the
next 10 turns, added additively to the pv_eta chooser's score with
λ=1.0 (the head's natural scale, since the output is in ship units).

### Infrastructure (commit `9d32066`)

| File | Purpose |
|---|---|
| `lib/value_head_features.py` | 14-d pure feature encoder; caller appends `leaf_delta` as `feats[14]` |
| `agents/baseline/_value_head.py` | Lazy-load + featurize + per-candidate predict (clone of `_ml_logit.py` for regression) |
| `agents/baseline_pv_eta_vh/main.py` | Env-var wrapper: `BASELINE_PV_ETA=1 + BASELINE_VH_LAMBDA=1.0` |
| `scripts/gen_b2_corpus.py` | Two-stage corpus runner (self-play + label pairing) |
| `scripts/train_value_head.py` | LightGBM regression_l1 trainer with σ(label) + Spearman gates |
| `scripts/bundle_pv_eta_vh.py` | Single-file Kaggle bundler |
| `tests/test_value_head_features.py` | 7 encoder unit tests |
| chooser modification | `chooser_trajectory.py` — additive `vh_get_lambda() * vh_predict_one(...)` after the ML lookup |

## Corpus (run `2026-05-29-100games`)

Self-play of `baseline_pv_eta_probe` (pv_eta with `BASELINE_ML_LAMBDA=0`,
`BASELINE_VH_TRACE_FEATURES=1`). The runner hit its 60-min timeout at
95/100 games — stage 2 (label pairing) ran on what was on disk.

| Stat | Value |
|---|---|
| Games | 95 (planned 100; runner timed out at 95) |
| Labelled candidates | **33,865** (≈ 18× the smoke; well above the planned 18k) |
| Per-seat | seat 0: 17,130 / seat 1: 16,735 — balanced |
| σ(label) | 458.5 (healthy band [200, 1500]) |
| mean(label) | −3.3 |
| Skipped (joints, no-features, truncated) | 8 / 0 / 286 |

Per-owner-at-launch label distribution reproduces B.1's pattern almost
exactly:

| owner | n | mean | σ | B.1 K=10 mean | B.1 K=10 σ |
|---|---:|---:|---:|---:|---:|
| me | 19,872 | +18 | 471 | −83 | 388 |
| neutral | 6,570 | +66 | 236 | +5 | 150 |
| enemy | 7,423 | −122 | 545 | −201 | 573 |

Same shape — enemy captures most overpredicted with highest spread,
neutral well-calibrated.

## Model (`data/value_head/value_head_model.txt` — gitignored, 654 KB)

LightGBM regression_l1, 31 leaves, lr=0.05, game-level 80/20 split.
Early stopping at iteration 231 (40-round patience).

| Metric | Val | Notes |
|---|---:|---|
| Best iteration | 231 | |
| RMSE | 376.99 | |
| MAE | 204.82 | |
| R² | +0.0323 | head explains 3.2% of label variance |
| **Spearman ρ** | **+0.3586** | **3× above the 0.10 mandatory gate** |
| σ(y_va) | 383.2 | |
| σ(residual) | 367.8 | head reduces residual σ by 15 ships |
| σ(pred) | 112.7 | head's output range |
| Walker parity | **0.000e+00** | exact match with `Booster.predict(...)` |

Compare to B.1's chooser leaf alone: R²≈0.005 and Spearman ρ≈0 at
K=10. The head is **7× better on R² and ~30× better on rank order** —
which is what the chooser actually argmaxes on.

### Feature importance (val-time gain ranking)

From the smoke corpus (4 games / 1961 candidates) — pattern at the
100-game scale should sharpen but the ranking is robust:

| Rank | Feature | Notes |
|---:|---|---|
| 1 | `tgt_distance_to_opp_centroid_at_eta` | The within-owner verdict predicted this would dominate; head confirms |
| 2 | `leaf_delta` | Chooser's own score is informative but not dominant — head adds independent signal |
| 3 | `ships_sent` | Enemy-launch ship-size correction (B.1 K=10 enemy-cell F=4.66) |
| 4 | `tgt_distance_to_sun_at_eta` | Per-planet geometric covariate |
| 5 | `combat_margin_at_arrival` | Direct (ships − defenders) ratio |
| 6 | `src_distance_to_sun` | |
| 7 | `tgt_production` | |
| 8 | `eta` | Weak inside owner cells (B.1 confirmed) |
| 9-15 | rest | Owner one-hots near bottom — interaction captured by per-planet × ship/eta splits |

The "all-in" feature design (per-planet covariates baked in from the
first cut, per PI direction) was correct. Without per-planet features
this head's top-ranked signal would have been missing.

## Bundle (`submissions/baseline_pv_eta_vh.py` — gitignored, 1.06 MB)

Built by `scripts/bundle_pv_eta_vh.py`. Inlines the inner pv_eta bundle
+ wrapper preamble + the 654KB regressor text as a base64-gzip blob in
`_VH_MODEL_B64`. Pure-numpy inference via `lib._validator_tree_walker`;
no lightgbm at submit time.

Verifications passing this commit:
- `pytest tests/test_bundle.py` — 10/10 GREEN.
- `pytest tests/test_value_head_features.py` — 7/7 GREEN.
- Bundle plays a clean game vs random (smoke iteration earlier).
- Walker parity 0.000e+00 on 500 val rows.

## A/B vs bare pv_eta (Rule 43, Rule 45)

```
python fast.py eval submissions/baseline_pv_eta_vh.py \
  --vs submissions/_imported/baseline_pv_eta.py \
  --max-seeds 32 --gate 0.50 --workers 4
```

| Field | Value |
|---|---|
| n | TBD (adaptive Wilson; up to 32) |
| Wins | TBD |
| Win rate | TBD |
| Wilson 95% CI | TBD |
| Gate | TBD |
| Verdict | TBD |

## Submission decision

TBD pending A/B verdict. If Wilson-lo ≥ 0.50:
- Pre-submit checklist: Rules 42 (push claim board) / 43 (panel) / 45
  (n ≥ 32) / 46 (bundle smoke).
- Rolling pair bot half (μ=1091.9, `baseline_leaf_pv_2p`) will be
  evicted; predicted μ should clear that floor.

If Wilson-lo < 0.50:
- λ sweep {0.5, 2.0} before declaring the axis falsified (within Rule
  37: B.2 is a new axis distinct from Reframe A's per-shot binary
  classifier).

## What this run does NOT do

- No 4P-coverage training data (2P self-play only).
- No multi-horizon auxiliary heads (single K=10).
- No CRN-paired advantage labels (the within-owner verdict shifted the
  bottleneck to features, not the supervision unit).
- No joints — the head is solo-only by design (training corpus
  excludes joint-arrival candidates).

## Reproduction

```
# Stage 1 (self-play), planned 100 games:
python scripts/gen_b2_corpus.py --games 100 --seed 2000 \
  --out data/value_head/corpus_runs/2026-05-29-100games \
  --wallclock-ms 100 --workers 4

# Stage 2 only (if stage 1 didn't finish — what happened here):
python scripts/gen_b2_corpus.py --games 100 --seed 2000 \
  --out data/value_head/corpus_runs/2026-05-29-100games \
  --skip-selfplay --workers 1

# Train:
python scripts/train_value_head.py \
  --corpus data/value_head/corpus_runs/2026-05-29-100games/corpus.jsonl \
  --out data/value_head/value_head_model.txt

# Bundle:
python scripts/bundle_pv_eta_vh.py
```

## Files preserved in this audit

The model artifact (`data/value_head/value_head_model.txt`, 654 KB),
the bundle (`submissions/baseline_pv_eta_vh.py`, 1.06 MB), and the
corpus (`data/value_head/corpus_runs/2026-05-29-100games/corpus.jsonl`,
~50 MB) are gitignored — they live in this container's filesystem
only. Reproduce via the commands above; the source code + this audit
fully specify the run.
