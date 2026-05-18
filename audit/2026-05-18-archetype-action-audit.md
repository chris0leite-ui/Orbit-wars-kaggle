# Archetype action audit: `med_high_prod__mixed_static__big_static`

Comparing top-10 winning play against our submission 52710995 play
in the highest-gap archetype from `audit/2026-05-18-team-archetype-gap.md`.

Sample sizes (2P, fingerprint prefix = 100 turns):
- top-10 wins: **6**
- our wins: **1**
- our losses: **4** (the high-value contrast)

## Per-feature comparison

| feature | top-10 mean | ours-all mean | delta | Cohen's d | ours-loss mean |
|---|---|---|---|---|---|
| `launches_per_turn` | 1.113 | 0.534 | 0.579 | 1.55 | 0.500 |
| `mean_fleet_size` | 31.990 | 35.038 | -3.048 | -0.58 | 32.600 |
| `p95_fleet_size` | 89.383 | 79.160 | 10.223 | 0.50 | 70.100 |
| `mean_target_distance` | 37.326 | 31.729 | 5.598 | 1.08 | 30.241 |
| `mean_target_production` | 2.980 | 2.729 | 0.251 | 0.59 | 2.598 |
| `mean_target_garrison` | 27.553 | 28.462 | -0.909 | -0.14 | 28.824 |
| `mean_garrison_at_launch` | 10.315 | 15.082 | -4.767 | -1.31 | 13.684 |
| `targets_neutral_fraction` | 0.268 | 0.404 | -0.136 | -2.01 | 0.438 |
| `targets_enemy_fraction` | 0.338 | 0.297 | 0.041 | 0.52 | 0.267 |
| `launch_angle_var` | 1.890 | 2.271 | -0.381 | -0.52 | 2.475 |
| `sun_clip_launch_rate` | 0.026 | 0.058 | -0.031 | -1.58 | 0.068 |
| `mean_planets_owned` | 7.278 | 7.764 | -0.486 | -0.37 | 7.855 |
| `mean_total_ships` | 339.475 | 380.830 | -41.355 | -0.53 | 350.230 |
| `ships_growth_per_turn` | 6.390 | 9.687 | -3.298 | -1.20 | 9.647 |
| `multi_launch_turn_rate` | 0.470 | 0.294 | 0.177 | 1.59 | 0.288 |

## Ranked by |Cohen's d|

- `targets_neutral_fraction`: top-10 is **lower** (0.268 vs 0.404, d = -2.01)
- `multi_launch_turn_rate`: top-10 is **higher** (0.470 vs 0.294, d = +1.59)
- `sun_clip_launch_rate`: top-10 is **lower** (0.026 vs 0.058, d = -1.58)
- `launches_per_turn`: top-10 is **higher** (1.113 vs 0.534, d = +1.55)
- `mean_garrison_at_launch`: top-10 is **lower** (10.315 vs 15.082, d = -1.31)
- `ships_growth_per_turn`: top-10 is **lower** (6.390 vs 9.687, d = -1.20)
- `mean_target_distance`: top-10 is **higher** (37.326 vs 31.729, d = +1.08)
- `mean_target_production`: top-10 is **higher** (2.980 vs 2.729, d = +0.59)
