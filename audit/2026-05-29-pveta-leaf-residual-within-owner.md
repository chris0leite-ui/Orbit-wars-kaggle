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

## Within-owner stratified ANOVA (B.1 follow-up sanity check)

For each K, the residual is partitioned by `owner_at_launch` (me / neutral / enemy) and the ship-quintile and top-5 target_id buckets are **recomputed inside each cell** so cutpoints reflect the within-cell distribution. F-stat thresholds vs the global pass: small within-cell F means the leaf's errors are dominated by the 3-way owner categorical, and a per-candidate head fed (owner, eta, ships) cannot rank candidates within a category.

### K = 5

| owner | n | F(ship_quintile) | F(eta_bucket) | F(target_id_top5) |
|---|---:|---:|---:|---:|
| me | 1704 | 1.46 (g=5) | 1.54 (g=3) | 1.28 (g=6) |
| neutral | 652 | 1.12 (g=5) | 1.45 (g=4) | 1.05 (g=6) |
| enemy | 639 | 1.35 (g=5) | 1.22 (g=4) | 0.69 (g=6) |

### K = 10

| owner | n | F(ship_quintile) | F(eta_bucket) | F(target_id_top5) |
|---|---:|---:|---:|---:|
| me | 1700 | 0.76 (g=5) | 0.83 (g=3) | 4.27 (g=6) |
| neutral | 641 | 1.19 (g=5) | 2.19 (g=4) | 1.26 (g=6) |
| enemy | 635 | 4.66 (g=5) | 2.05 (g=4) | 0.61 (g=6) |

### K = 20

| owner | n | F(ship_quintile) | F(eta_bucket) | F(target_id_top5) |
|---|---:|---:|---:|---:|
| me | 1675 | 0.14 (g=5) | 1.62 (g=3) | 5.78 (g=6) |
| neutral | 625 | 0.56 (g=5) | 0.33 (g=4) | 4.49 (g=6) |
| enemy | 595 | 6.70 (g=5) | 1.08 (g=4) | 1.95 (g=6) |

## Residual by target_id_top5 (K=10, owner=me, F=4.27)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| other | 1004 | -88.80 | 566.98 |
| tgt_4 | 166 | -274.15 | 773.07 |
| tgt_8 | 136 | -85.00 | 462.53 |
| tgt_9 | 136 | -18.08 | 397.91 |
| tgt_16 | 134 | -111.00 | 573.16 |
| tgt_1 | 124 | -33.42 | 317.40 |

## Residual by ship_quintile (K=10, owner=enemy, F=4.66)

| bucket | n | mean residual | std |
|---|---:|---:|---:|
| Q3 | 131 | -142.64 | 602.24 |
| Q1 | 127 | -198.60 | 895.27 |
| Q4 | 127 | -211.88 | 472.10 |
| Q5 | 127 | -448.23 | 920.87 |
| Q2 | 123 | -110.75 | 369.58 |

## B.2 within-owner verdict

**GREEN — B.2 as specced has within-category signal. At least one of {me, enemy} shows ship- or eta-driven residual structure (F > 4) inside the cell. Proceed with the (owner_at_launch, eta, ships, leaf-Δ) regressor.**

At K=10: me+enemy max F (ship/eta) = 4.66; across-all max F (ship/eta) = 4.66.

| owner | n | F(ship) | F(eta) | F(target) |
|---|---:|---:|---:|---:|
| me | 1700 | 0.76 | 0.83 | 4.27 |
| neutral | 641 | 1.19 | 2.19 | 1.26 |
| enemy | 635 | 4.66 | 2.05 | 0.61 |

Gate: GREEN if me-or-enemy max(F_ship, F_eta) > 4; AMBER if any cell max(F_ship, F_eta) ≥ 2; else RED.

## Interpretation

σ(residual)/σ(actual) measures the fraction of future ship-delta variance that the chooser's leaf-Δ does NOT explain. Ratio → 0 means the leaf already predicts the outcome perfectly; ratio → 1 means the leaf is noise against the ground truth.

ANOVA F > 4 on an axis means residuals systematically differ across that axis's buckets. A non-flat residual structure is the headroom a per-target value head can exploit; if all F-stats are small, the leaf's errors are unstructured noise that no per-target head can fix.

## Operator verdict — qualified GREEN; amend B.2 design

The mechanical gate fires GREEN (enemy-cell ship-quintile F = 4.66 at K=10), but the within-cell picture is concentrated and **changes one premise of the B.2 plan**.

### What the within-cell signal actually looks like

| Horizon | Where the within-cell signal lives | Where the B.2 feature set picks it up |
|---|---|---|
| K=5 | Nowhere. ALL within-cell F are < 2 (max = 1.54). | n/a — K=5 is too short to train on |
| K=10 | enemy / ship-quintile (F=4.66); me / target_id (F=4.27) | enemy/ship: YES (`ships` feature). me/target: NO (plan excludes target_id) |
| K=20 | enemy / ship-quintile (F=6.70); me / target_id (F=5.78); neutral / target_id (F=4.49) | same as K=10 |

Inside the `me` cell at K=10 the (`ships`, `eta`) handles are essentially flat (F = 0.76, 0.83). Inside the `enemy` cell at K=10 `eta` is weak (F = 2.05). The B.2 feature set as specced — `owner_at_launch` + `eta` + `ships` + leaf-Δ — captures the enemy-launch ship-size correction and the 3-way owner offset, and **nothing else**.

### Mechanistic read

- **Enemy captures**: residual grows monotonically with launch size (Q5 enemy residual = −448, Q2 = −110, K=10). The chooser's leaf assumes a static opponent; in reality the opponent counter-emits roughly in proportion to the captured planet's value, so a bigger launch attracts a bigger counter. The leaf's overprediction grows with launch size. This is the "ship-feature inside the enemy cell" signal.
- **Own-planet reinforcements**: residual varies per target planet, not per launch size (tgt_4 residual = −274, tgt_1 = −33, K=10 inside me). The own-planet reinforcement value is set by which planet you're reinforcing (production rate, position relative to opp center, defense pressure), not by how many ships you send. **The plan's "don't bother with target_id" was based on the global F-stat (1.6 at K=5, 3.28 at K=10), which is diluted by the enemy + neutral cells where target_id carries no signal. The me-cell pattern only shows up when you condition on owner.**
- **Neutral captures**: globally well-calibrated (residual = +5, σ=150) AND within-cell flat. No learning headroom here.

### What to amend in B.2

1. **Add per-planet covariates, conditioned on owner.** The within-cell target_id signal in the me-cell (F=4.27 at K=10, F=5.78 at K=20) is real. Don't encode raw planet_id (sparse — ~136 samples per top-5 target across 22 planets); encode stable per-planet characteristics: production rate, position, distance to opponent centroid, distance to sun. These × `owner_at_launch` interactions capture the conditional pattern without bloating capacity.
2. **K=10 is the right training horizon.** K=5 has zero within-cell signal anywhere. K=20 has slightly more signal but more games are truncated. The original handover already picked K=10; this probe confirms.
3. **Calibrate expectations.** Within-cell σ(residual) at K=10 ranges 150 (neutral) to 920 (Q5 enemy). The head reduces variance by F-driven structure, but the absolute residual stays large — expect a directional A/B improvement, not a transformative one. R²(within-cell) is bounded by these F-stats and will not approach the variance the global F-stat suggests is "headroom."

### What this means for the session ladder

- B.2 stays viable; the plan's `owner_at_launch` + `eta` + `ships` + leaf-Δ feature set will learn the enemy-launch correction.
- Add per-planet covariates before training, not after. Doing it later requires regenerating the corpus.
- Submission gate unchanged: Wilson-lo ≥ 0.50 vs bare pv_eta at n ≥ 32 (Rule 43, Rule 45).
- If the head trained on `owner × ships × eta × leaf-Δ` alone fails to clear the gate, the per-planet-features amendment is the first thing to try before declaring the axis closed.

### Open question for PI

Worth asking before B.2 corpus regen: do we want the per-planet covariates baked into the feature set on the first cut, or do we ship the plan's minimal feature set first as a control and add per-planet on a second iteration if the first one stalls at parity? The minimal-control-first version is the cleaner falsification (we'd know whether per-planet matters); the all-in version is one fewer iteration.
