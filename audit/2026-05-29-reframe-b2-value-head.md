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

## A/B vs bare pv_eta (Rule 43, Rule 45) — CATASTROPHIC FAIL

| λ | Wins | n | Win rate | Wilson 95% CI | Verdict |
|---:|---:|---:|---:|---|---|
| 1.0 (default) | 0 | 32 | 0.0 % | [0.000, 0.107] | **FAIL** |
| 0.1 (sweep) | 0 | 32 | 0.0 % | [0.000, 0.107] | **FAIL** |

Single-game sanity (seed=99): VH at λ=1.0 as P0 lost to bare pv_eta
P1 over a clean 500-turn game (both DONE, no crashes). Latency healthy:
λ=1.0 p50=238ms / p95=691ms; λ=0.1 p50=423ms / p95=714ms (head's
per-candidate predict adds ~200ms p50). All within the 1000ms env cap.

Both seats lost universally — `fast.py eval` uses balanced focal
rotation so VH played P0 half the seeds and P1 half. 0/32 isn't a
seat-asymmetry artifact.

## Diagnosis — selection bias on observational labels

The Spearman ρ = +0.359 gate at training time was a false positive.
Mechanically:

- The training trace `trace_accepted` only emits features for
  candidates the chooser **accepted**. The training distribution is
  `{candidates pv_eta picks}`.
- Labels are observational K=10 ship-delta — "what happened in the
  game where this candidate was picked", not "what would have happened
  if a different candidate had been picked."
- At inference, the chooser uses `head_out` to re-rank **all** prerank
  candidates, including ones pv_eta would have rejected. The head's
  predictions on rejected candidates are unconstrained LightGBM
  extrapolation.
- Even at λ=0.1 (head perturbation = ~10 ships per candidate), those
  out-of-distribution predictions flip the chooser's argmax to
  systematically losing actions.

This is the **PM3 distillation-collapse failure mode** in operational
form (see `knowledge-base/thoughts/2026-05-28-pm-distillation-action-
rank-collapse.md`). High val Spearman ρ within the training
distribution does NOT imply rank-order preservation on the deployment
distribution.

## What's closed (Rule 37 axis cap)

The **observational-label additive-term head on pv_eta's chooser** axis
is falsified. Both verdict directions (λ=1.0 high, λ=0.1 low) hit the
same 0/32 floor — the bottleneck is the **label semantics**, not the
model's coefficient. Do not re-run with:

- Adjusted λ (any value)
- Different K horizon (5, 20) with the same observational target
- Different feature subsets
- A bigger corpus of the same data shape

All four would reproduce 0/32 by the same selection-bias mechanism.

## Submission decision

**No push.** Both rolling-pair slots stay (sub 53131296 at μ=1097 and
sub 53117942 at μ=1092). Predicted-μ for the VH bundle would be
catastrophically below either — evicting a rolling-pair slot here
would be a Rule 42 violation.

## Next session — Reframe B.3

See `audit/2026-05-30-reframe-b3-crn-advantage-plan.md` for the
full plan. Headline: replace observational labels with **CRN-paired
advantage** `A(s, a) = focal_margin(action) − focal_margin(idle)`
over K=10 steps. The full B.2 infrastructure (encoder, value-head
loader, wrapper, bundler, A/B harness) is reused unchanged — only the
labels are wrong.

Critical-path: verify `lib/fast_sim.rollout` works with pv_eta as a
policy callback BEFORE building stage 2 corpus infrastructure. If
fast_sim works, B.3 stage 2 fits in ~7.5 h CPU on a 4-worker box.
If not, fallback to env.clone+step blows the cost up ~20×.

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
