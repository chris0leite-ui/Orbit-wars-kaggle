# Archetype × planet-class rollup audit

Per-archetype, per-planet-class behaviour comparison: top-10 winning play vs our submission 52710995 across the first 100 turns of every 2P game.

Class label = `{prod}_{kin}_{prox}`: production above/below per-board median, rotating vs static (`is_orbiting`), inner vs outer (orbital radius above/below per-board median). 8 classes total.

Headline metric is **`target_intensity_delta`** (top-10 minus ours):
- Strongly positive → top-10 prizes this class, we ignore it.
- Strongly negative → we waste shots on this class, top-10 ignores it.
- Near zero with low end-owned-rates → true filler planets.

Caveat: the prior aggregate audit established top-10 launches ~2x as often in absolute terms, so most classes show a uniformly positive intensity delta. **`target_share_delta`** (per-class % of total launches; top-10 share minus ours share) normalises out the universal aggression deficit and surfaces class-conditional allocation differences.

## Headline findings

Ranked by |mean share-delta| across all 16 informative cells. Positive = top-10 over-allocates relative to ours; negative = we over-allocate.

- **`low_prod_rotating_inner`**: mean Δshare +0.100 — top-10 prizes this class (8 cells positive / 2 negative at |Δ| ≥ 0.05).
- **`high_prod_static_outer`**: mean Δshare -0.049 — we over-allocate to this class (3 cells positive / 7 negative at |Δ| ≥ 0.05).
- **`high_prod_static_inner`**: mean Δshare -0.047 — we over-allocate to this class (1 cells positive / 9 negative at |Δ| ≥ 0.05).
- **`high_prod_rotating_inner`**: mean Δshare -0.023 — we over-allocate to this class (3 cells positive / 5 negative at |Δ| ≥ 0.05).
- **`low_prod_static_inner`**: mean Δshare +0.017 — top-10 prizes this class (3 cells positive / 2 negative at |Δ| ≥ 0.05).

## Cross-cell summary

For each class, cells where the |intensity delta| crosses 0.15. Direction = sign of (top-10 minus ours).

**By `target_intensity_delta`** (top-10 minus ours, raw):

| class | top-10-prizes (positive Δ) | we-overshoot (negative Δ) | mean Δ | n cells |
|---|---|---|---|---|
| `high_prod_rotating_inner` | low_prod__mostly_static__big_rotating, low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, med_high_prod__mostly_rotating__big_rotating, high_prod__mixed_static__big_static, high_prod__mostly_rotating__big_static, med_high_prod__mixed_static__big_static, med_low_prod__mixed_rotating__big_static, high_prod__mixed_rotating__big_rotating, low_prod__mixed_rotating__big_static, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating, med_low_prod__mostly_static__big_rotating, med_low_prod__mostly_rotating__big_static | low_prod__mixed_rotating__big_rotating, low_prod__mostly_static__big_static | +1.441 | 16 |
| `high_prod_rotating_outer` | — | — | +0.000 | 16 |
| `high_prod_static_inner` | low_prod__mostly_static__big_rotating, high_prod__mixed_static__big_static, med_high_prod__mixed_static__big_static, low_prod__mixed_rotating__big_static, low_prod__mostly_static__big_static | low_prod__mixed_rotating__big_rotating, med_low_prod__mixed_static__big_rotating, high_prod__mostly_rotating__big_static, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating, med_low_prod__mostly_static__big_rotating, med_low_prod__mostly_rotating__big_static | +0.024 | 16 |
| `high_prod_static_outer` | low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, med_high_prod__mostly_rotating__big_rotating, high_prod__mixed_static__big_static, high_prod__mostly_rotating__big_static, med_high_prod__mixed_static__big_static, med_low_prod__mixed_rotating__big_static, high_prod__mixed_rotating__big_rotating, low_prod__mixed_rotating__big_static, med_high_prod__mostly_rotating__big_static, low_prod__mostly_static__big_static, low_prod__mixed_static__big_rotating | low_prod__mostly_static__big_rotating, med_low_prod__mostly_static__big_rotating | +1.390 | 16 |
| `low_prod_rotating_inner` | low_prod__mixed_rotating__big_rotating, low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, med_high_prod__mostly_rotating__big_rotating, high_prod__mixed_static__big_static, med_high_prod__mixed_static__big_static, med_low_prod__mixed_rotating__big_static, high_prod__mixed_rotating__big_rotating, low_prod__mixed_rotating__big_static, med_high_prod__mostly_rotating__big_static, med_low_prod__mostly_rotating__big_static | high_prod__mostly_rotating__big_static, low_prod__mostly_static__big_static, low_prod__mixed_static__big_rotating | +2.041 | 16 |
| `low_prod_rotating_outer` | — | — | +0.000 | 16 |
| `low_prod_static_inner` | med_low_prod__mixed_static__big_rotating, med_high_prod__mixed_static__big_static, med_low_prod__mixed_rotating__big_static, med_low_prod__mostly_static__big_rotating | high_prod__mixed_rotating__big_rotating, low_prod__mostly_static__big_static, low_prod__mixed_static__big_rotating | +0.469 | 16 |
| `low_prod_static_outer` | low_prod__mostly_static__big_rotating, low_prod__mixed_rotating__big_rotating, low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, high_prod__mixed_static__big_static, high_prod__mostly_rotating__big_static, med_low_prod__mixed_rotating__big_static, low_prod__mixed_rotating__big_static, low_prod__mostly_static__big_static | med_high_prod__mostly_rotating__big_rotating, med_high_prod__mixed_static__big_static, high_prod__mixed_rotating__big_rotating, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating, med_low_prod__mostly_static__big_rotating, med_low_prod__mostly_rotating__big_static | +0.332 | 16 |

**By `target_share_delta`** (class-share of total launches; removes the universal aggression deficit):

| class | top-10-prizes (positive Δ) | we-overshoot (negative Δ) | mean Δ | n cells |
|---|---|---|---|---|
| `high_prod_rotating_inner` | low_prod__mostly_static__big_rotating, med_low_prod__mixed_rotating__big_static, med_low_prod__mostly_rotating__big_static | low_prod__mixed_rotating__big_rotating, low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, high_prod__mixed_rotating__big_rotating, low_prod__mixed_rotating__big_static | -0.023 | 16 |
| `high_prod_rotating_outer` | — | — | +0.000 | 16 |
| `high_prod_static_inner` | low_prod__mostly_static__big_static | low_prod__mixed_rotating__big_rotating, med_low_prod__mixed_static__big_rotating, high_prod__mostly_rotating__big_static, med_low_prod__mixed_rotating__big_static, low_prod__mixed_rotating__big_static, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating, med_low_prod__mostly_static__big_rotating, med_low_prod__mostly_rotating__big_static | -0.047 | 16 |
| `high_prod_static_outer` | med_high_prod__mostly_rotating__big_rotating, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating | low_prod__mostly_static__big_rotating, low_prod__mixed_rotating__big_rotating, med_low_prod__mixed_rotating__big_static, high_prod__mixed_rotating__big_rotating, low_prod__mostly_static__big_static, med_low_prod__mostly_static__big_rotating, med_low_prod__mostly_rotating__big_static | -0.049 | 16 |
| `low_prod_rotating_inner` | low_prod__mixed_rotating__big_rotating, low_prod__mostly_rotating__big_static, med_low_prod__mixed_static__big_rotating, med_high_prod__mostly_rotating__big_rotating, med_high_prod__mixed_static__big_static, high_prod__mixed_rotating__big_rotating, low_prod__mixed_rotating__big_static, med_high_prod__mostly_rotating__big_static | high_prod__mixed_static__big_static, low_prod__mostly_static__big_static | +0.100 | 16 |
| `low_prod_rotating_outer` | — | — | +0.000 | 16 |
| `low_prod_static_inner` | med_low_prod__mixed_static__big_rotating, med_low_prod__mixed_rotating__big_static, med_low_prod__mostly_static__big_rotating | high_prod__mixed_rotating__big_rotating, low_prod__mostly_static__big_static | +0.017 | 16 |
| `low_prod_static_outer` | low_prod__mostly_static__big_rotating, low_prod__mixed_rotating__big_rotating, high_prod__mixed_static__big_static, high_prod__mostly_rotating__big_static, med_low_prod__mixed_rotating__big_static, low_prod__mixed_rotating__big_static, low_prod__mostly_static__big_static | med_low_prod__mixed_static__big_rotating, med_high_prod__mostly_rotating__big_rotating, med_high_prod__mixed_static__big_static, high_prod__mixed_rotating__big_rotating, med_high_prod__mostly_rotating__big_static, low_prod__mixed_static__big_rotating | +0.003 | 16 |

## Per-cell tables

### `low_prod__mostly_static__big_rotating` — top10 n=1, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_static_outer` | 4.00 | 3.500 | 0.750 | +2.750 | 0.215 | 0.035 | +0.181 | 0.00 | 0.50 |
| `high_prod_rotating_inner` | 4.00 | 3.750 | 1.167 | +2.583 | 0.231 | 0.163 | +0.068 | 1.00 | 1.00 |
| `high_prod_static_inner` | 8.00 | 3.750 | 2.438 | +1.312 | 0.462 | 0.453 | +0.008 | 1.00 | 1.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_outer` | 4.00 | 1.500 | 1.875 | -0.375 | 0.092 | 0.349 | -0.257 | 0.00 | 0.50 |

### `low_prod__mixed_rotating__big_rotating` — top10 n=1, ours n=1

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 4.00 | 3.250 | 0.000 | +3.250 | 0.265 | 0.000 | +0.265 | 1.00 | 0.00 |
| `low_prod_static_outer` | 8.00 | 1.000 | 0.000 | +1.000 | 0.163 | 0.000 | +0.163 | 1.00 | 0.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 4.00 | 2.000 | 2.250 | -0.250 | 0.163 | 0.237 | -0.074 | 0.00 | 1.00 |
| `high_prod_rotating_inner` | 8.00 | 1.500 | 1.750 | -0.250 | 0.245 | 0.368 | -0.124 | 1.00 | 1.00 |
| `high_prod_static_outer` | 4.00 | 2.000 | 1.875 | +0.125 | 0.163 | 0.395 | -0.231 | 1.00 | 1.00 |

### `low_prod__mostly_rotating__big_static` — top10 n=2, ours n=1

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 10.00 | 1.800 | 0.000 | +1.800 | 0.353 | 0.000 | +0.353 | 1.00 | 0.00 |
| `low_prod_static_outer` | 2.00 | 0.250 | 0.000 | +0.250 | 0.010 | 0.000 | +0.010 | 0.00 | 0.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_outer` | 12.00 | 2.042 | 1.000 | +1.042 | 0.480 | 0.500 | -0.020 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 4.00 | 2.000 | 0.800 | +1.200 | 0.157 | 0.500 | -0.343 | 1.00 | 1.00 |

### `med_low_prod__mixed_static__big_rotating` — top10 n=1, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 4.00 | 12.000 | 1.625 | +10.375 | 0.338 | 0.096 | +0.242 | 1.00 | 1.00 |
| `low_prod_static_inner` | 4.00 | 5.000 | 0.000 | +5.000 | 0.141 | 0.000 | +0.141 | 1.00 | 0.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_outer` | 8.00 | 5.125 | 3.333 | +1.792 | 0.289 | 0.296 | -0.008 | 1.00 | 1.00 |
| `low_prod_static_outer` | 4.00 | 5.750 | 1.938 | +3.812 | 0.162 | 0.230 | -0.068 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 4.00 | 2.500 | 2.167 | +0.333 | 0.070 | 0.193 | -0.122 | 1.00 | 1.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 3.125 | -3.125 | 0.000 | 0.185 | -0.185 | 0.00 | 1.00 |

### `med_high_prod__mostly_rotating__big_rotating` — top10 n=1, ours n=1

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 8.00 | 2.375 | 1.250 | +1.125 | 0.244 | 0.114 | +0.130 | 1.00 | 0.00 |
| `high_prod_static_outer` | 8.00 | 2.500 | 1.500 | +1.000 | 0.256 | 0.136 | +0.120 | 1.00 | 0.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_rotating_inner` | 8.00 | 2.625 | 1.625 | +1.000 | 0.269 | 0.295 | -0.026 | 1.00 | 0.00 |
| `low_prod_static_outer` | 8.00 | 2.250 | 2.500 | -0.250 | 0.231 | 0.455 | -0.224 | 0.00 | 0.00 |

### `high_prod__mixed_static__big_static` — top10 n=2, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_static_outer` | 6.00 | 3.583 | 0.625 | +2.958 | 0.150 | 0.049 | +0.101 | 1.00 | 0.50 |
| `high_prod_static_inner` | 6.00 | 8.167 | 2.667 | +5.500 | 0.341 | 0.311 | +0.031 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 6.00 | 1.917 | 1.500 | +0.417 | 0.080 | 0.058 | +0.022 | 1.00 | 0.50 |
| `high_prod_static_outer` | 10.00 | 5.300 | 1.500 | +3.800 | 0.369 | 0.350 | +0.020 | 1.00 | 1.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_inner` | 6.00 | 1.417 | 1.200 | +0.217 | 0.059 | 0.233 | -0.174 | 0.50 | 1.00 |

### `high_prod__mostly_rotating__big_static` — top10 n=1, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_static_outer` | 8.00 | 1.750 | 1.125 | +0.625 | 0.132 | 0.063 | +0.069 | 1.00 | 1.00 |
| `high_prod_static_outer` | 12.00 | 3.500 | 2.167 | +1.333 | 0.396 | 0.366 | +0.030 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 8.00 | 3.625 | 2.188 | +1.438 | 0.274 | 0.246 | +0.027 | 1.00 | 1.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_inner` | 12.00 | 1.750 | 2.125 | -0.375 | 0.198 | 0.239 | -0.041 | 1.00 | 1.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 3.000 | -3.000 | 0.000 | 0.085 | -0.085 | 0.00 | 0.50 |

### `med_high_prod__mixed_static__big_static` — top10 n=6, ours n=5

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 8.00 | 3.667 | 1.222 | +2.444 | 0.281 | 0.170 | +0.111 | 1.00 | 1.00 |
| `low_prod_static_inner` | 0.67 | 4.000 | 0.000 | +4.000 | 0.026 | 0.000 | +0.026 | 0.00 | 0.00 |
| `high_prod_static_outer` | 10.00 | 4.567 | 2.096 | +2.471 | 0.438 | 0.421 | +0.017 | 1.00 | 1.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_rotating_inner` | 2.67 | 2.750 | 1.375 | +1.375 | 0.070 | 0.085 | -0.015 | 0.50 | 0.60 |
| `high_prod_static_inner` | 3.33 | 4.300 | 1.833 | +2.467 | 0.137 | 0.170 | -0.033 | 0.50 | 1.00 |
| `low_prod_static_outer` | 4.67 | 1.071 | 2.000 | -0.929 | 0.048 | 0.154 | -0.107 | 0.50 | 0.80 |

### `med_low_prod__mixed_rotating__big_static` — top10 n=2, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `high_prod_rotating_inner` | 6.00 | 3.083 | 0.750 | +2.333 | 0.287 | 0.085 | +0.202 | 1.00 | 0.50 |
| `low_prod_static_outer` | 4.00 | 1.000 | 0.000 | +1.000 | 0.062 | 0.000 | +0.062 | 0.50 | 0.00 |
| `low_prod_static_inner` | 2.00 | 1.750 | 0.000 | +1.750 | 0.054 | 0.000 | +0.054 | 0.50 | 0.00 |
| `low_prod_rotating_inner` | 4.00 | 3.500 | 1.917 | +1.583 | 0.217 | 0.217 | +0.000 | 1.00 | 0.50 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 2.00 | 2.750 | 2.875 | -0.125 | 0.085 | 0.217 | -0.132 | 0.50 | 1.00 |
| `high_prod_static_outer` | 6.00 | 3.167 | 2.125 | +1.042 | 0.295 | 0.481 | -0.187 | 1.00 | 1.00 |

### `high_prod__mixed_rotating__big_rotating` — top10 n=1, ours n=2

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 4.00 | 8.250 | 2.000 | +6.250 | 0.324 | 0.048 | +0.275 | 1.00 | 0.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 4.00 | 2.500 | 3.125 | -0.625 | 0.098 | 0.152 | -0.053 | 0.00 | 0.50 |
| `high_prod_rotating_inner` | 4.00 | 7.250 | 2.850 | +4.400 | 0.284 | 0.345 | -0.061 | 1.00 | 1.00 |
| `low_prod_static_outer` | 0.00 | 0.000 | 2.750 | -2.750 | 0.000 | 0.067 | -0.067 | 0.00 | 0.00 |
| `high_prod_static_outer` | 8.00 | 3.750 | 3.200 | +0.550 | 0.294 | 0.388 | -0.094 | 1.00 | 1.00 |

### `low_prod__mixed_rotating__big_static` — top10 n=1, ours n=3

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 8.00 | 3.750 | 2.250 | +1.500 | 0.345 | 0.061 | +0.284 | 1.00 | 0.33 |
| `low_prod_static_outer` | 4.00 | 3.750 | 3.500 | +0.250 | 0.172 | 0.095 | +0.077 | 1.00 | 0.33 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_outer` | 8.00 | 2.375 | 1.583 | +0.792 | 0.218 | 0.259 | -0.040 | 1.00 | 1.00 |
| `high_prod_static_inner` | 4.00 | 3.250 | 2.750 | +0.500 | 0.149 | 0.224 | -0.075 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 4.00 | 2.500 | 2.208 | +0.292 | 0.115 | 0.361 | -0.246 | 1.00 | 1.00 |

### `med_high_prod__mostly_rotating__big_static` — top10 n=1, ours n=4

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_rotating_inner` | 8.00 | 6.000 | 1.167 | +4.833 | 0.378 | 0.200 | +0.178 | 1.00 | 1.00 |
| `high_prod_static_outer` | 12.00 | 4.500 | 1.750 | +2.750 | 0.425 | 0.367 | +0.059 | 1.00 | 1.00 |
| `high_prod_rotating_inner` | 4.00 | 6.250 | 1.179 | +5.071 | 0.197 | 0.157 | +0.040 | 1.00 | 1.00 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_outer` | 0.00 | 0.000 | 1.100 | -1.100 | 0.000 | 0.105 | -0.105 | 0.00 | 0.75 |
| `high_prod_static_inner` | 0.00 | 0.000 | 3.000 | -3.000 | 0.000 | 0.171 | -0.171 | 0.00 | 0.75 |

### `low_prod__mostly_static__big_static` — top10 n=1, ours n=4

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `high_prod_static_inner` | 8.00 | 6.875 | 1.643 | +5.232 | 0.545 | 0.311 | +0.234 | 1.00 | 1.00 |
| `low_prod_static_outer` | 4.00 | 3.500 | 2.750 | +0.750 | 0.139 | 0.074 | +0.064 | 1.00 | 0.25 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_rotating_inner` | 0.00 | 0.000 | 1.000 | -1.000 | 0.000 | 0.027 | -0.027 | 0.00 | 0.25 |
| `low_prod_static_inner` | 0.00 | 0.000 | 3.000 | -3.000 | 0.000 | 0.081 | -0.081 | 0.00 | 0.00 |
| `low_prod_rotating_inner` | 4.00 | 1.500 | 1.750 | -0.250 | 0.059 | 0.142 | -0.082 | 0.00 | 0.50 |
| `high_prod_static_outer` | 4.00 | 6.500 | 1.929 | +4.571 | 0.257 | 0.365 | -0.107 | 0.00 | 1.00 |

### `low_prod__mixed_static__big_rotating` — top10 n=1, ours n=4

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `high_prod_static_outer` | 8.00 | 3.000 | 1.417 | +1.583 | 0.500 | 0.305 | +0.195 | 1.00 | 1.00 |
| `low_prod_rotating_inner` | 4.00 | 1.500 | 2.000 | -0.500 | 0.125 | 0.096 | +0.029 | 1.00 | 0.25 |
| `high_prod_rotating_inner` | 4.00 | 2.750 | 1.321 | +1.429 | 0.229 | 0.222 | +0.008 | 1.00 | 0.75 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 4.00 | 0.750 | 1.375 | -0.625 | 0.062 | 0.066 | -0.003 | 0.00 | 0.00 |
| `high_prod_static_inner` | 0.00 | 0.000 | 1.875 | -1.875 | 0.000 | 0.090 | -0.090 | 0.00 | 0.25 |
| `low_prod_static_outer` | 4.00 | 1.000 | 2.312 | -1.312 | 0.083 | 0.222 | -0.138 | 1.00 | 0.50 |

### `med_low_prod__mostly_static__big_rotating` — top10 n=1, ours n=3

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `low_prod_static_inner` | 4.00 | 2.750 | 1.750 | +1.000 | 0.289 | 0.105 | +0.184 | 1.00 | 0.33 |
| `low_prod_static_outer` | 4.00 | 2.000 | 2.750 | -0.750 | 0.211 | 0.165 | +0.045 | 1.00 | 0.67 |
| `high_prod_rotating_inner` | 4.00 | 1.250 | 0.938 | +0.312 | 0.132 | 0.113 | +0.019 | 1.00 | 0.67 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `high_prod_static_inner` | 4.00 | 1.750 | 2.500 | -0.750 | 0.184 | 0.301 | -0.117 | 1.00 | 0.67 |
| `high_prod_static_outer` | 4.00 | 1.750 | 2.100 | -0.350 | 0.184 | 0.316 | -0.132 | 1.00 | 0.67 |

### `med_low_prod__mostly_rotating__big_static` — top10 n=1, ours n=3

| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |
|---|---|---|---|---|---|---|---|---|---|
| `high_prod_rotating_inner` | 8.00 | 3.250 | 1.125 | +2.125 | 0.366 | 0.164 | +0.203 | 1.00 | 0.67 |
| `low_prod_rotating_inner` | 8.00 | 2.000 | 1.600 | +0.400 | 0.225 | 0.194 | +0.031 | 1.00 | 0.67 |
| `high_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_rotating_outer` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_inner` | 0.00 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.00 | 0.00 |
| `low_prod_static_outer` | 0.00 | 0.000 | 1.000 | -1.000 | 0.000 | 0.024 | -0.024 | 0.00 | 0.33 |
| `high_prod_static_inner` | 0.00 | 0.000 | 2.500 | -2.500 | 0.000 | 0.061 | -0.061 | 0.00 | 0.33 |
| `high_prod_static_outer` | 12.00 | 2.417 | 2.300 | +0.117 | 0.408 | 0.558 | -0.149 | 1.00 | 1.00 |

