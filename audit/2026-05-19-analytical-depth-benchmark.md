# Analytical-depth benchmark — 2026-05-19

Per the v3 plan in `/root/.claude/plans/read-the-handover-
do-abundant-quokka.md`. Measures forward-projection cost to
decide v3 depth-3 design parameters.

Self-play agent for obs capture: `agents/baseline/main.py`

## early obs

- planets: 24, fleets: 2, step: 20

### Q1. Pure fast_sim.step

- **0.159 ms / step**

### Q2/Q3. K-turn projection (median over 20 trials)

| opp policy | K=10 | K=30 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| empty | 0.7 | 19.3 | 20.7 | 23.6 |
| lite_greedy | 1.0 | 20.5 | 24.7 | 53.4 |
| mirror-v2 | 61.1 | 292.7 | 740.5 | 2192.5 |

### Q4. Plans per turn at given budget (lite_greedy projection)

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 4 | 24 | 48 |
| 50 | 4 | 20 | 40 |
| 100 | 1 | 9 | 18 |

### Q4-bis. Plans per turn — full mirror-v2 opp at each step

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 0 | 1 | 3 |
| 50 | 0 | 0 | 1 |
| 100 | 0 | 0 | 0 |

### Q5. Does v2's ~60 candidates fit at K=50?

- lite_greedy: 60 plans × 24.7 ms = **1482 ms** — OVER BUDGET
- mirror-v2: 60 plans × 740.5 ms = **44433 ms** — OVER BUDGET


## mid obs

- planets: 28, fleets: 22, step: 80

### Q1. Pure fast_sim.step

- **0.278 ms / step**

### Q2/Q3. K-turn projection (median over 20 trials)

| opp policy | K=10 | K=30 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| empty | 2.0 | 4.0 | 5.3 | 34.8 |
| lite_greedy | 3.2 | 7.8 | 11.1 | 46.8 |
| mirror-v2 | 312.9 | 1187.3 | 2156.6 | 4662.1 |

### Q4. Plans per turn at given budget (lite_greedy projection)

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 12 | 64 | 128 |
| 50 | 8 | 44 | 89 |
| 100 | 2 | 10 | 21 |

### Q4-bis. Plans per turn — full mirror-v2 opp at each step

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 0 | 0 | 0 |
| 50 | 0 | 0 | 0 |
| 100 | 0 | 0 | 0 |

### Q5. Does v2's ~60 candidates fit at K=50?

- lite_greedy: 60 plans × 11.1 ms = **667 ms** — OK
- mirror-v2: 60 plans × 2156.6 ms = **129396 ms** — OVER BUDGET


## late obs

- planets: 28, fleets: 14, step: 180

### Q1. Pure fast_sim.step

- **0.148 ms / step**

### Q2/Q3. K-turn projection (median over 20 trials)

| opp policy | K=10 | K=30 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| empty | 1.6 | 3.1 | 4.4 | 10.9 |
| lite_greedy | 3.6 | 8.0 | 11.5 | 26.9 |
| mirror-v2 | 522.1 | 1291.8 | 1329.0 | 1342.0 |

### Q4. Plans per turn at given budget (lite_greedy projection)

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 12 | 62 | 125 |
| 50 | 8 | 43 | 87 |
| 100 | 3 | 18 | 37 |

### Q4-bis. Plans per turn — full mirror-v2 opp at each step

| K | budget=100ms | budget=500ms | budget=1000ms |
|---|---:|---:|---:|
| 30 | 0 | 0 | 0 |
| 50 | 0 | 0 | 0 |
| 100 | 0 | 0 | 0 |

### Q5. Does v2's ~60 candidates fit at K=50?

- lite_greedy: 60 plans × 11.5 ms = **689 ms** — OK
- mirror-v2: 60 plans × 1329.0 ms = **79741 ms** — OVER BUDGET

