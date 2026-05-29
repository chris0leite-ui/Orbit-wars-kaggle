# Reframe B.1 — pv_eta leaf-residual diagnostic probe

Data dir: `audit/2026-05-29-pveta-probe-data`
Games analysed: **16**  Seat-0 accepted candidates: **3002**

## Verdict: **GREENLIT Reframe B.2 (per-target value head)**

Triggers (K → residual_ratio, max F):
- K=5: ratio=1.242, max F=43.20
- K=10: ratio=1.126, max F=35.94
- K=20: ratio=1.070, max F=43.71

Gate rule: any K with σ(residual)/σ(actual) > 0.5 AND any stratification ANOVA F > 4.

## Per-K stats

| K | n | σ(actual) | σ(pred) | σ(residual) | ratio | R² | ρ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2995 | 327.24 | 224.87 | 406.41 | 1.242 | 0.003 | 0.007 |
| 10 | 2976 | 501.86 | 225.51 | 565.26 | 1.126 | 0.006 | -0.036 |
| 20 | 2895 | 699.53 | 228.12 | 748.57 | 1.070 | 0.004 | -0.023 |

## ANOVA F-stats by stratification axis

| K | ship_quintile | eta_bucket | owner_at_launch | top-5 tgt |
|---:|---:|---:|---:|---:|
| 5 | 6.42 (g=5) | 10.93 (g=4) | 43.20 (g=3) | 1.62 (g=6) |
| 10 | 7.13 (g=5) | 11.46 (g=4) | 35.94 (g=3) | 3.28 (g=6) |
| 20 | 8.00 (g=5) | 14.47 (g=4) | 43.71 (g=3) | 4.06 (g=6) |

## Residual by ship_quintile (K=5)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| Q3 | 603 | -85.78 | 374.59 |
| Q2 | 601 | -33.01 | 242.18 |
| Q5 | 601 | -150.72 | 529.34 |
| Q4 | 599 | -82.44 | 313.66 |
| Q1 | 591 | -92.00 | 492.39 |

## Residual by eta_bucket (K=5)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| [4-8] | 1448 | -102.59 | 389.75 |
| [1-3] | 631 | -121.71 | 465.36 |
| [9+] | 515 | -91.15 | 505.47 |
| [0] | 401 | +15.96 | 53.19 |

## Residual by owner_at_launch (K=5)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| me | 1704 | -82.75 | 387.96 |
| neutral | 652 | +5.40 | 149.95 |
| enemy | 639 | -200.95 | 572.81 |

## Residual by target_id_top5 (K=5)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| other | 1943 | -95.09 | 445.45 |
| tgt_4 | 224 | -126.34 | 446.03 |
| tgt_8 | 220 | -53.49 | 266.78 |
| tgt_1 | 210 | -36.91 | 200.19 |
| tgt_12 | 208 | -98.61 | 316.07 |
| tgt_16 | 190 | -67.42 | 311.06 |

## Interpretation

σ(residual)/σ(actual) measures the fraction of future ship-delta variance that the chooser's leaf-Δ does NOT explain. Ratio → 0 means the leaf already predicts the outcome perfectly; ratio → 1 means the leaf is noise against the ground truth.

ANOVA F > 4 on an axis means residuals systematically differ across that axis's buckets. A non-flat residual structure is the headroom a per-target value head can exploit; if all F-stats are small, the leaf's errors are unstructured noise that no per-target head can fix.
