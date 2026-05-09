# lr_diag — drop-in LR-diagnostics for any tabular comp

Origin: s6e5 LR-diagnostics expedition, 2026-05-07. See parent skill
doc `../../../lr-diagnostics.md` for full description, decision
rules, and three durable lessons.

## What you must change before running on a new comp

Each script has these s6e5-isms hard-coded near the top. Edit them:

```python
TARGET = "PitNextLap"           # → your binary target column
ART = Path("scripts/artifacts") # → keep as-is unless your project differs

# E1, E8, E9, a2_gate, a4_per_segment
K21_BASES = [...]               # → your pool of base OOF names
                                #   (each must have oof_<NAME>_strat.npy)
EXTRAS = [...]                  # → bases added beyond the original pool

# a2_gate.py, a4_per_segment.py
K10_BASES = [...]               # → your forward-selected core (from E9)
```

In `e4_per_segment.py`, `e5_bootstrap_coef.py`, `e6_residual_interactions.py`,
`a2_bagged_lr.py`, `a4_per_segment.py`:

```python
cat_cols = ["Driver", "Compound", "Race"]   # → your categorical columns
# Per-segment cell key (E4, A4):
df["Compound"]                              # → your segmentation column
# Stint-cross interactions (E6, A2):
("Stint", "RaceProgress"), ...              # → your dominant-feature ×
                                            #   other-features pairs
```

## Recommended sequence

```bash
# Arc A: pool/meta diagnostics
python scripts/lr_diag_e1_svd.py
python scripts/lr_diag_e2_calibration.py
python scripts/lr_diag_e8_grid.py
python scripts/lr_diag_e4_per_segment.py

# Arc B: DGP archaeology
python scripts/lr_diag_e5_bootstrap_coef.py
python scripts/lr_diag_e6_residual_interactions.py
python scripts/lr_diag_e9_forward_select.py
# → use E9 output to define K_N_BASES (the true effective pool)

# Arc C: new-base injection (only if Arc A/B point to LR-class diversity)
python scripts/lr_diag_a2_bagged_lr.py
python scripts/lr_diag_a2_gate.py
python scripts/lr_diag_a4_per_segment.py
```

## Cost guide (1M-row binary classification, lbfgs L2)

| Script | Wall time |
|---|---:|
| e1_svd | <1 min |
| e2_calibration | 5-30 min (24 bases × 5-fold; depends on row count) |
| e4_per_segment | 30-60 min |
| e5_bootstrap_coef | 5-15 min (50 boots × 2 regimes; **DROP saga L1** at low C — saga at C ≤ 0.01 takes >>1 min/fit) |
| e6_residual_interactions | 5 min |
| e8_grid | 5-10 min (10 lbfgs L2 grid; saga L1 takes 1h+ — pruned by default) |
| e9_forward_select | 10-30 min (subsampled to ~110k rows) |
| a2_bagged_lr | 5-15 min |
| a2_gate | 1-5 min |
| a4_per_segment | 5 min |

## Known pitfalls

- **saga L1 + small C is intractable** on 400k+ row × 50+ feature
  matrices. The committed scripts use lbfgs L2 by default; one-line
  swap to saga L1 will deadlock.
- **`max_iter` warnings**: lbfgs sometimes converges past 2000 iters
  for very low C; bump to 4000 if you see ConvergenceWarning.
- **sklearn 1.8+ deprecates `penalty=` kwarg** in favor of `l1_ratio`.
  Scripts work but emit warnings; harmless.
- **5-fold StratKF random_state=42** is hard-coded for cross-script
  consistency. If your comp uses GroupKFold (group-leakage risk),
  swap the fold object in `_meta_oof()` and `run_variant()` etc.

## Adapting to non-binary metric

The scripts assume binary AUC. For multinomial / balanced-accuracy:
- E8: use `class_weight='balanced'` + `solver='lbfgs'` with
  `multi_class='multinomial'` (sklearn ≤1.7) or `l1_ratio`-form (1.8+).
  Per s6e4 12th place: logits-as-features matters more for balanced
  accuracy than for AUC.
- E1/E9: re-define `_pos()` to extract per-class probs.
- A2/A4: regression (multi-output LR) instead of binary; metrics
  swap to `balanced_accuracy_score` or `log_loss`.

## See also

- `audit/2026-05-07-lr-diagnostics-arc{A,B,C}.md` (origin)
- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` (motivation)
- s6e4 writeups stash: `/tmp/s6e4-writeups/` (from comp ps-s6e4
  via `kaggle competition_list_topic_messages`)
