# Cross-cell behavior audit: top-10 vs us across all gap cells

Behavior-diff (15-d fingerprint) on every cell from `audit/2026-05-18-team-archetype-gap.json` that has at least one top-10 and one ours sample. **16 cells aggregated.** Each cell contributes Cohen's d per feature; the cross-cell weighted-mean d is the headline.

## Cross-cell summary (sorted by |weighted-mean d|)

| feature | cells | weighted-mean d | mean \|d\| | + / − cells |
|---|---|---|---|---|
| `multi_launch_turn_rate` | 3 | +3.422 | 4.338 | 3/0 |
| `targets_neutral_fraction` | 3 | -2.800 | 3.197 | 0/3 |
| `mean_garrison_at_launch` | 3 | -1.991 | 2.330 | 0/3 |
| `ships_growth_per_turn` | 3 | -1.849 | 2.172 | 0/3 |
| `sun_clip_launch_rate` | 3 | -1.750 | 1.836 | 0/3 |
| `mean_target_production` | 3 | +1.736 | 2.309 | 3/0 |
| `launches_per_turn` | 3 | +1.675 | 1.738 | 3/0 |
| `targets_enemy_fraction` | 3 | +1.591 | 2.126 | 3/0 |
| `mean_planets_owned` | 3 | -1.080 | 1.434 | 0/3 |
| `mean_total_ships` | 3 | -0.947 | 1.153 | 0/3 |
| `launch_angle_var` | 3 | -0.920 | 1.357 | 1/2 |
| `mean_target_distance` | 3 | +0.767 | 0.609 | 3/0 |
| `p95_fleet_size` | 3 | +0.282 | 1.584 | 2/1 |
| `mean_target_garrison` | 3 | +0.227 | 0.503 | 2/1 |
| `mean_fleet_size` | 3 | -0.092 | 1.464 | 1/2 |

**Reading:**
- `weighted-mean d` = mean Cohen's d across cells, weighted by min(top-10-n, ours-n).
- **Positive** d → top-10 has the feature HIGHER than us.
- **Negative** d → top-10 has the feature LOWER than us.
- **+/− cells** = how many cells show each sign. A feature with 10 cells all-positive or all-negative is a **universal** behavioral gap; mixed signs suggest the behavior is archetype-specific.

## Per-cell deltas

### `low_prod__mostly_static__big_rotating` (top10_n=1, ours_n=2, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.730 | 0.440 | 0.290 | — |
| `mean_fleet_size` | 31.781 | 30.058 | 1.722 | — |
| `p95_fleet_size` | 104.400 | 55.600 | 48.800 | — |
| `mean_target_distance` | 36.358 | 36.600 | -0.242 | — |
| `mean_target_production` | 2.972 | 2.318 | 0.654 | — |
| `mean_target_garrison` | 38.845 | 25.279 | 13.566 | — |
| `mean_garrison_at_launch` | 23.915 | 10.074 | 13.841 | — |
| `targets_neutral_fraction` | 0.380 | 0.451 | -0.071 | — |
| `targets_enemy_fraction` | 0.437 | 0.209 | 0.227 | — |
| `launch_angle_var` | 2.410 | 2.707 | -0.297 | — |
| `sun_clip_launch_rate` | 0.127 | 0.026 | 0.100 | — |
| `mean_planets_owned` | 3.480 | 6.915 | -3.435 | — |
| `mean_total_ships` | 155.530 | 289.695 | -134.165 | — |
| `ships_growth_per_turn` | 4.227 | 7.779 | -3.552 | — |
| `multi_launch_turn_rate` | 0.475 | 0.137 | 0.338 | — |

### `low_prod__mixed_rotating__big_rotating` (top10_n=1, ours_n=1, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.651 | 0.380 | 0.271 | — |
| `mean_fleet_size` | 33.352 | 32.053 | 1.299 | — |
| `p95_fleet_size` | 101.150 | 70.150 | 31.000 | — |
| `mean_target_distance` | 39.746 | 39.043 | 0.703 | — |
| `mean_target_production` | 1.907 | 2.000 | -0.093 | — |
| `mean_target_garrison` | 29.315 | 29.684 | -0.369 | — |
| `mean_garrison_at_launch` | 5.667 | 7.763 | -2.096 | — |
| `targets_neutral_fraction` | 0.444 | 0.368 | 0.076 | — |
| `targets_enemy_fraction` | 0.315 | 0.316 | -0.001 | — |
| `launch_angle_var` | 2.796 | 2.194 | 0.601 | — |
| `sun_clip_launch_rate` | 0.259 | 0.000 | 0.259 | — |
| `mean_planets_owned` | 5.349 | 5.870 | -0.521 | — |
| `mean_total_ships` | 182.337 | 230.920 | -48.583 | — |
| `ships_growth_per_turn` | 6.504 | 2.708 | 3.796 | — |
| `multi_launch_turn_rate` | 0.194 | 0.152 | 0.043 | — |

### `low_prod__mostly_rotating__big_static` (top10_n=2, ours_n=1, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.575 | 0.320 | 0.255 | — |
| `mean_fleet_size` | 27.621 | 29.844 | -2.222 | — |
| `p95_fleet_size` | 57.075 | 59.800 | -2.725 | — |
| `mean_target_distance` | 36.487 | 37.166 | -0.679 | — |
| `mean_target_production` | 1.849 | 1.594 | 0.255 | — |
| `mean_target_garrison` | 25.855 | 22.094 | 3.761 | — |
| `mean_garrison_at_launch` | 10.666 | 6.562 | 4.104 | — |
| `targets_neutral_fraction` | 0.530 | 0.469 | 0.062 | — |
| `targets_enemy_fraction` | 0.255 | 0.250 | 0.005 | — |
| `launch_angle_var` | 2.475 | 1.980 | 0.494 | — |
| `sun_clip_launch_rate` | 0.017 | 0.031 | -0.014 | — |
| `mean_planets_owned` | 6.435 | 8.060 | -1.625 | — |
| `mean_total_ships` | 184.230 | 274.330 | -90.100 | — |
| `ships_growth_per_turn` | 4.746 | 8.503 | -3.757 | — |
| `multi_launch_turn_rate` | 0.364 | 0.250 | 0.114 | — |

### `med_low_prod__mixed_static__big_rotating` (top10_n=1, ours_n=2, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.470 | 0.700 | 0.770 | — |
| `mean_fleet_size` | 38.925 | 35.350 | 3.575 | — |
| `p95_fleet_size` | 111.400 | 89.475 | 21.925 | — |
| `mean_target_distance` | 36.226 | 36.381 | -0.155 | — |
| `mean_target_production` | 2.466 | 2.443 | 0.023 | — |
| `mean_target_garrison` | 29.685 | 27.507 | 2.178 | — |
| `mean_garrison_at_launch` | 19.986 | 15.971 | 4.015 | — |
| `targets_neutral_fraction` | 0.267 | 0.221 | 0.046 | — |
| `targets_enemy_fraction` | 0.288 | 0.479 | -0.191 | — |
| `launch_angle_var` | 4.086 | 2.516 | 1.570 | — |
| `sun_clip_launch_rate` | 0.027 | 0.043 | -0.015 | — |
| `mean_planets_owned` | 7.970 | 10.165 | -2.195 | — |
| `mean_total_ships` | 501.550 | 495.945 | 5.605 | — |
| `ships_growth_per_turn` | 12.466 | 13.150 | -0.684 | — |
| `multi_launch_turn_rate` | 0.620 | 0.409 | 0.210 | — |

### `med_high_prod__mostly_rotating__big_rotating` (top10_n=1, ours_n=1, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.820 | 0.465 | 0.355 | — |
| `mean_fleet_size` | 40.134 | 35.152 | 4.982 | — |
| `p95_fleet_size` | 96.850 | 70.250 | 26.600 | — |
| `mean_target_distance` | 45.247 | 35.463 | 9.783 | — |
| `mean_target_production` | 2.854 | 2.804 | 0.049 | — |
| `mean_target_garrison` | 32.646 | 29.370 | 3.277 | — |
| `mean_garrison_at_launch` | 9.341 | 16.848 | -7.506 | — |
| `targets_neutral_fraction` | 0.280 | 0.348 | -0.067 | — |
| `targets_enemy_fraction` | 0.268 | 0.304 | -0.036 | — |
| `launch_angle_var` | 1.509 | 2.222 | -0.713 | — |
| `sun_clip_launch_rate` | 0.305 | 0.000 | 0.305 | — |
| `mean_planets_owned` | 7.660 | 5.222 | 2.438 | — |
| `mean_total_ships` | 374.810 | 292.889 | 81.921 | — |
| `ships_growth_per_turn` | 8.915 | 1.911 | 7.003 | — |
| `multi_launch_turn_rate` | 0.442 | 0.257 | 0.185 | — |

### `high_prod__mixed_static__big_static` (top10_n=2, ours_n=2, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.495 | 0.540 | 0.955 | 2.02 |
| `mean_fleet_size` | 31.411 | 33.681 | -2.271 | -1.39 |
| `p95_fleet_size` | 72.200 | 87.775 | -15.575 | -2.12 |
| `mean_target_distance` | 31.879 | 28.410 | 3.469 | 0.73 |
| `mean_target_production` | 3.446 | 3.077 | 0.369 | 3.07 |
| `mean_target_garrison` | 28.332 | 25.325 | 3.007 | 1.11 |
| `mean_garrison_at_launch` | 11.202 | 16.392 | -5.190 | -5.52 |
| `targets_neutral_fraction` | 0.208 | 0.342 | -0.134 | -4.11 |
| `targets_enemy_fraction` | 0.557 | 0.310 | 0.247 | 3.58 |
| `launch_angle_var` | 1.345 | 2.722 | -1.377 | -3.19 |
| `sun_clip_launch_rate` | 0.038 | 0.046 | -0.008 | -2.79 |
| `mean_planets_owned` | 7.855 | 8.905 | -1.050 | -3.37 |
| `mean_total_ships` | 403.195 | 484.670 | -81.475 | -2.25 |
| `ships_growth_per_turn` | 7.820 | 12.826 | -5.006 | -2.21 |
| `multi_launch_turn_rate` | 0.564 | 0.225 | 0.338 | 2.70 |

### `high_prod__mostly_rotating__big_static` (top10_n=1, ours_n=2, panel-gap=+100%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.080 | 0.755 | 0.325 | — |
| `mean_fleet_size` | 47.083 | 45.028 | 2.055 | — |
| `p95_fleet_size` | 117.000 | 124.050 | -7.050 | — |
| `mean_target_distance` | 31.446 | 31.111 | 0.335 | — |
| `mean_target_production` | 2.639 | 3.055 | -0.416 | — |
| `mean_target_garrison` | 24.731 | 31.199 | -6.468 | — |
| `mean_garrison_at_launch` | 8.046 | 15.881 | -7.835 | — |
| `targets_neutral_fraction` | 0.324 | 0.298 | 0.026 | — |
| `targets_enemy_fraction` | 0.315 | 0.312 | 0.003 | — |
| `launch_angle_var` | 1.824 | 2.915 | -1.091 | — |
| `sun_clip_launch_rate` | 0.019 | 0.053 | -0.035 | — |
| `mean_planets_owned` | 11.040 | 12.765 | -1.725 | — |
| `mean_total_ships` | 545.200 | 777.165 | -231.965 | — |
| `ships_growth_per_turn` | 12.598 | 18.253 | -5.655 | — |
| `multi_launch_turn_rate` | 0.388 | 0.332 | 0.056 | — |

### `med_high_prod__mixed_static__big_static` (top10_n=6, ours_n=5, panel-gap=+80%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.113 | 0.534 | 0.579 | 1.55 |
| `mean_fleet_size` | 31.990 | 35.038 | -3.048 | -0.58 |
| `p95_fleet_size` | 89.383 | 79.160 | 10.223 | 0.50 |
| `mean_target_distance` | 37.326 | 31.729 | 5.598 | 1.08 |
| `mean_target_production` | 2.980 | 2.729 | 0.251 | 0.59 |
| `mean_target_garrison` | 27.553 | 28.462 | -0.909 | -0.14 |
| `mean_garrison_at_launch` | 10.315 | 15.082 | -4.767 | -1.31 |
| `targets_neutral_fraction` | 0.268 | 0.404 | -0.136 | -2.01 |
| `targets_enemy_fraction` | 0.338 | 0.297 | 0.041 | 0.52 |
| `launch_angle_var` | 1.890 | 2.271 | -0.381 | -0.52 |
| `sun_clip_launch_rate` | 0.026 | 0.058 | -0.031 | -1.58 |
| `mean_planets_owned` | 7.278 | 7.764 | -0.486 | -0.37 |
| `mean_total_ships` | 339.475 | 380.830 | -41.355 | -0.53 |
| `ships_growth_per_turn` | 6.390 | 9.687 | -3.298 | -1.20 |
| `multi_launch_turn_rate` | 0.470 | 0.294 | 0.177 | 1.59 |

### `med_low_prod__mixed_rotating__big_static` (top10_n=2, ours_n=2, panel-gap=+50%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.700 | 0.560 | 0.140 | 1.64 |
| `mean_fleet_size` | 48.710 | 29.850 | 18.860 | 2.42 |
| `p95_fleet_size` | 109.950 | 67.125 | 42.825 | 2.13 |
| `mean_target_distance` | 35.580 | 35.531 | 0.049 | 0.01 |
| `mean_target_production` | 3.039 | 2.086 | 0.953 | 3.27 |
| `mean_target_garrison` | 26.570 | 25.569 | 1.001 | 0.26 |
| `mean_garrison_at_launch` | 10.655 | 11.176 | -0.521 | -0.16 |
| `targets_neutral_fraction` | 0.219 | 0.438 | -0.219 | -3.47 |
| `targets_enemy_fraction` | 0.379 | 0.309 | 0.070 | 2.28 |
| `launch_angle_var` | 2.658 | 2.274 | 0.385 | 0.36 |
| `sun_clip_launch_rate` | 0.065 | 0.120 | -0.054 | -1.14 |
| `mean_planets_owned` | 6.650 | 7.665 | -1.015 | -0.56 |
| `mean_total_ships` | 328.995 | 375.920 | -46.925 | -0.67 |
| `ships_growth_per_turn` | 7.022 | 10.094 | -3.071 | -3.10 |
| `multi_launch_turn_rate` | 0.443 | 0.307 | 0.137 | 8.73 |

### `high_prod__mixed_rotating__big_rotating` (top10_n=1, ours_n=2, panel-gap=+50%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.070 | 0.850 | 0.220 | — |
| `mean_fleet_size` | 65.495 | 47.113 | 18.382 | — |
| `p95_fleet_size` | 196.400 | 121.450 | 74.950 | — |
| `mean_target_distance` | 30.808 | 35.610 | -4.802 | — |
| `mean_target_production` | 4.381 | 3.493 | 0.888 | — |
| `mean_target_garrison` | 46.762 | 30.188 | 16.573 | — |
| `mean_garrison_at_launch` | 10.038 | 17.781 | -7.743 | — |
| `targets_neutral_fraction` | 0.152 | 0.252 | -0.100 | — |
| `targets_enemy_fraction` | 0.467 | 0.355 | 0.112 | — |
| `launch_angle_var` | 4.855 | 2.948 | 1.906 | — |
| `sun_clip_launch_rate` | 0.029 | 0.062 | -0.034 | — |
| `mean_planets_owned` | 6.260 | 8.690 | -2.430 | — |
| `mean_total_ships` | 409.820 | 642.905 | -233.085 | — |
| `ships_growth_per_turn` | 6.226 | 16.450 | -10.224 | — |
| `multi_launch_turn_rate` | 0.500 | 0.385 | 0.115 | — |

### `low_prod__mixed_rotating__big_static` (top10_n=1, ours_n=3, panel-gap=+33%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.870 | 0.537 | 0.333 | — |
| `mean_fleet_size` | 30.598 | 27.704 | 2.894 | — |
| `p95_fleet_size` | 74.700 | 66.433 | 8.267 | — |
| `mean_target_distance` | 38.540 | 34.283 | 4.258 | — |
| `mean_target_production` | 1.931 | 1.870 | 0.062 | — |
| `mean_target_garrison` | 22.931 | 21.487 | 1.445 | — |
| `mean_garrison_at_launch` | 10.414 | 11.940 | -1.526 | — |
| `targets_neutral_fraction` | 0.207 | 0.399 | -0.192 | — |
| `targets_enemy_fraction` | 0.195 | 0.326 | -0.130 | — |
| `launch_angle_var` | 1.829 | 2.294 | -0.465 | — |
| `sun_clip_launch_rate` | 0.000 | 0.041 | -0.041 | — |
| `mean_planets_owned` | 9.590 | 6.713 | 2.877 | — |
| `mean_total_ships` | 392.040 | 312.120 | 79.920 | — |
| `ships_growth_per_turn` | 8.602 | 7.589 | 1.013 | — |
| `multi_launch_turn_rate` | 0.375 | 0.236 | 0.139 | — |

### `med_high_prod__mostly_rotating__big_static` (top10_n=1, ours_n=4, panel-gap=+25%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.310 | 0.560 | 0.750 | — |
| `mean_fleet_size` | 38.015 | 36.809 | 1.206 | — |
| `p95_fleet_size` | 101.000 | 89.263 | 11.737 | — |
| `mean_target_distance` | 34.809 | 36.192 | -1.382 | — |
| `mean_target_production` | 3.214 | 2.494 | 0.720 | — |
| `mean_target_garrison` | 26.916 | 24.214 | 2.702 | — |
| `mean_garrison_at_launch` | 9.092 | 10.949 | -1.857 | — |
| `targets_neutral_fraction` | 0.321 | 0.451 | -0.131 | — |
| `targets_enemy_fraction` | 0.229 | 0.304 | -0.075 | — |
| `launch_angle_var` | 3.402 | 2.934 | 0.468 | — |
| `sun_clip_launch_rate` | 0.015 | 0.107 | -0.092 | — |
| `mean_planets_owned` | 7.450 | 9.380 | -1.930 | — |
| `mean_total_ships` | 344.500 | 456.995 | -112.495 | — |
| `ships_growth_per_turn` | 8.705 | 11.822 | -3.117 | — |
| `multi_launch_turn_rate` | 0.557 | 0.312 | 0.245 | — |

### `low_prod__mostly_static__big_static` (top10_n=1, ours_n=4, panel-gap=+0%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 1.020 | 0.392 | 0.628 | — |
| `mean_fleet_size` | 29.098 | 37.854 | -8.756 | — |
| `p95_fleet_size` | 91.900 | 79.087 | 12.812 | — |
| `mean_target_distance` | 27.965 | 37.813 | -9.848 | — |
| `mean_target_production` | 2.693 | 2.437 | 0.256 | — |
| `mean_target_garrison` | 25.059 | 26.569 | -1.510 | — |
| `mean_garrison_at_launch` | 4.010 | 12.390 | -8.380 | — |
| `targets_neutral_fraction` | 0.257 | 0.445 | -0.188 | — |
| `targets_enemy_fraction` | 0.574 | 0.279 | 0.296 | — |
| `launch_angle_var` | 1.645 | 2.857 | -1.212 | — |
| `sun_clip_launch_rate` | 0.020 | 0.044 | -0.024 | — |
| `mean_planets_owned` | 3.890 | 5.260 | -1.370 | — |
| `mean_total_ships` | 177.620 | 280.862 | -103.242 | — |
| `ships_growth_per_turn` | 2.770 | 6.694 | -3.924 | — |
| `multi_launch_turn_rate` | 0.288 | 0.217 | 0.071 | — |

### `low_prod__mixed_static__big_rotating` (top10_n=1, ours_n=4, panel-gap=+0%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.520 | 0.417 | 0.103 | — |
| `mean_fleet_size` | 34.500 | 29.151 | 5.349 | — |
| `p95_fleet_size` | 100.300 | 68.450 | 31.850 | — |
| `mean_target_distance` | 29.719 | 28.207 | 1.512 | — |
| `mean_target_production` | 2.519 | 2.046 | 0.474 | — |
| `mean_target_garrison` | 29.769 | 27.579 | 2.190 | — |
| `mean_garrison_at_launch` | 8.885 | 7.544 | 1.340 | — |
| `targets_neutral_fraction` | 0.346 | 0.423 | -0.077 | — |
| `targets_enemy_fraction` | 0.115 | 0.411 | -0.296 | — |
| `launch_angle_var` | 3.349 | 3.423 | -0.074 | — |
| `sun_clip_launch_rate` | 0.000 | 0.025 | -0.025 | — |
| `mean_planets_owned` | 6.430 | 5.398 | 1.032 | — |
| `mean_total_ships` | 272.260 | 209.428 | 62.832 | — |
| `ships_growth_per_turn` | 7.156 | 5.141 | 2.014 | — |
| `multi_launch_turn_rate` | 0.378 | 0.273 | 0.105 | — |

### `med_low_prod__mostly_static__big_rotating` (top10_n=1, ours_n=3, panel-gap=+0%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.390 | 0.463 | -0.073 | — |
| `mean_fleet_size` | 52.436 | 43.278 | 9.158 | — |
| `p95_fleet_size` | 127.500 | 111.583 | 15.917 | — |
| `mean_target_distance` | 28.797 | 34.279 | -5.482 | — |
| `mean_target_production` | 2.897 | 2.645 | 0.253 | — |
| `mean_target_garrison` | 47.692 | 31.988 | 15.704 | — |
| `mean_garrison_at_launch` | 6.744 | 12.888 | -6.144 | — |
| `targets_neutral_fraction` | 0.282 | 0.542 | -0.260 | — |
| `targets_enemy_fraction` | 0.103 | 0.220 | -0.118 | — |
| `launch_angle_var` | 1.020 | 2.857 | -1.837 | — |
| `sun_clip_launch_rate` | 0.051 | 0.005 | 0.046 | — |
| `mean_planets_owned` | 5.790 | 7.570 | -1.780 | — |
| `mean_total_ships` | 340.280 | 467.640 | -127.360 | — |
| `ships_growth_per_turn` | 10.297 | 12.612 | -2.315 | — |
| `multi_launch_turn_rate` | 0.219 | 0.234 | -0.016 | — |

### `med_low_prod__mostly_rotating__big_static` (top10_n=1, ours_n=3, panel-gap=+0%)

| feature | top-10 | ours | delta | d |
|---|---|---|---|---|
| `launches_per_turn` | 0.760 | 0.580 | 0.180 | — |
| `mean_fleet_size` | 38.487 | 36.649 | 1.838 | — |
| `p95_fleet_size` | 74.000 | 103.833 | -29.833 | — |
| `mean_target_distance` | 34.561 | 37.870 | -3.309 | — |
| `mean_target_production` | 3.066 | 2.318 | 0.748 | — |
| `mean_target_garrison` | 28.737 | 25.618 | 3.119 | — |
| `mean_garrison_at_launch` | 10.961 | 12.972 | -2.011 | — |
| `targets_neutral_fraction` | 0.289 | 0.361 | -0.072 | — |
| `targets_enemy_fraction` | 0.289 | 0.284 | 0.005 | — |
| `launch_angle_var` | 2.366 | 2.692 | -0.326 | — |
| `sun_clip_launch_rate` | 0.053 | 0.106 | -0.053 | — |
| `mean_planets_owned` | 8.810 | 9.973 | -1.163 | — |
| `mean_total_ships` | 383.440 | 454.587 | -71.147 | — |
| `ships_growth_per_turn` | 9.927 | 11.405 | -1.478 | — |
| `multi_launch_turn_rate` | 0.455 | 0.290 | 0.164 | — |

