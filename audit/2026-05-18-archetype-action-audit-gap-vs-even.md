# Gap-vs-even pool comparison

Pools fingerprint samples across two groups of cells to decide 
whether top-10's edge is **universal** (top-10 diverges from us 
in EVERY cell, gap or not) or **conditional** (top-10 only 
diverges in the gap cells where we're losing).

- **GAP cells** (panel-gap >= +30%, baseline losing): 11 cells; pooled top10_n=19, ours_n=23
- **EVEN cells** (|gap| <= 20%, baseline competitive): 4 cells; pooled top10_n=4, ours_n=14

Classification rule:
- **UNIVERSAL**: |d_gap| > 0.5 and |d_gap - d_even| < 0.3 (same direction, similar magnitude in both pools)
- **CONDITIONAL**: |d_gap| > 0.5 and |d_even| < 0.3 (top-10 only diverges in gap cells)
- **mixed**: everything else

## Per-feature comparison

| feature | top10/ours (GAP) | top10/ours (EVEN) | d_gap | d_even | cross-d | kind |
|---|---|---|---|---|---|---|
| `launches_per_turn` | 0.995 / 0.571 | 0.673 / 0.455 | 1.26 | 1.25 | 0.00 | UNIVERSAL |
| `mean_fleet_size` | 36.568 / 34.674 | 38.630 / 36.271 | 0.22 | 0.31 | -0.09 | mixed |
| `p95_fleet_size` | 95.613 / 82.011 | 98.425 / 88.314 | 0.48 | 0.42 | 0.06 | mixed |
| `mean_target_distance` | 36.327 / 33.932 | 30.261 / 34.323 | 0.52 | -0.71 | 1.23 | mixed |
| `mean_target_production` | 2.826 / 2.548 | 2.794 / 2.344 | 0.44 | 0.85 | -0.41 | mixed |
| `mean_target_garrison` | 29.039 / 26.872 | 32.814 / 27.815 | 0.36 | 0.68 | -0.32 | mixed |
| `mean_garrison_at_launch` | 11.281 / 13.781 | 7.650 / 11.237 | -0.56 | -0.82 | 0.26 | UNIVERSAL |
| `targets_neutral_fraction` | 0.294 / 0.366 | 0.294 / 0.442 | -0.57 | -1.04 | 0.47 | mixed |
| `targets_enemy_fraction` | 0.352 / 0.317 | 0.270 / 0.305 | 0.35 | -0.25 | 0.60 | mixed |
| `launch_angle_var` | 2.295 / 2.469 | 2.095 / 2.984 | -0.19 | -0.65 | 0.46 | mixed |
| `sun_clip_launch_rate` | 0.061 / 0.050 | 0.031 / 0.044 | 0.19 | -0.33 | 0.51 | mixed |
| `mean_planets_owned` | 7.205 / 8.188 | 6.230 / 6.804 | -0.47 | -0.18 | -0.29 | mixed |
| `mean_total_ships` | 338.472 / 424.837 | 293.400 / 337.703 | -0.60 | -0.26 | -0.34 | CONDITIONAL |
| `ships_growth_per_turn` | 7.213 / 10.497 | 7.537 / 8.528 | -0.83 | -0.19 | -0.65 | CONDITIONAL |
| `multi_launch_turn_rate` | 0.450 / 0.279 | 0.335 / 0.253 | 1.48 | 0.96 | 0.53 | mixed |

## Summary

- **Universal features:** 2 — `launches_per_turn, mean_garrison_at_launch`
- **Conditional features:** 2 — `mean_total_ships, ships_growth_per_turn`

