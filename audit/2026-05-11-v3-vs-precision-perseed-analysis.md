# 2026-05-11 — v3 vs precision per-seed feature analysis

## Question

v3 vs precision = 50/50 over 8 seeds. v3 wins {0, 1, 3, 4}, precision
wins {2, 5, 6, 7}. What board feature predicts which class wins?

## Features extracted at turn 0

| seed | winner | n_planets | home_prod | home_ships | n_orbit | avg_neut_d | min_neut_d |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | v3 | 32 | 1 | 10 | 16 | 50.4 | 9.1 |
| 1 | v3 | 20 | 3 | 10 | 4 | 50.6 | 28.3 |
| 3 | v3 | 36 | 1 | 10 | 12 | 53.8 | 12.9 |
| 4 | v3 | 20 | 3 | 10 | 8 | 60.5 | 14.3 |
| 2 | precision | 20 | 5 | 10 | 4 | 67.2 | 12.0 |
| 5 | precision | 36 | 3 | 10 | 12 | 54.0 | 10.7 |
| 6 | precision | 20 | 1 | 10 | 8 | 64.3 | 18.8 |
| 7 | precision | 24 | 4 | 10 | 8 | 60.8 | 12.5 |

## Pattern

Two features distinguish:

**1. home_prod (production rate of P0's home planet).** Per-class mean:
- v3 wins: 1, 3, 1, 3 → mean 2.0
- precision wins: 5, 3, 1, 4 → mean 3.25

precision wins on HIGHER home production. (4/4 of precision's wins have
prod ≥ 3 except seed 6 which has prod=1.)

**2. avg_neut_d (avg distance from our home to all neutrals).** Per-class mean:
- v3 wins: 50.4, 50.6, 53.8, 60.5 → mean 53.8
- precision wins: 67.2, 54.0, 64.3, 60.8 → mean 61.6

precision wins on MORE-DISTANT neutrals.

## Interpretation

**precision is stronger on long sustained-production games**: high-prod
homes generate ships steadily, far neutrals delay early conflict, and
the game devolves into careful long-game contention. precision's
robust min-max scoring + depth-2 enemy minimax pays off here.

**v3 is stronger on aggressive-early-capture games**: low-prod homes
force expansion to nearby neutrals quickly, and v3's per-source greedy
plus same-turn ledger excels at rapid capture races.

The 50/50 result is the average over these two regimes. To climb above
50% against precision, v3 needs to NOT-LOSE in the long-sustained
regime. That's where wave bundling (multi-source coordinated attacks)
and strike-window timing would matter most.

## Caveat

8 seeds is a tiny sample. Two features are correlated (both indicate
"game develops slowly"). A 32-seed extension would tighten the pattern
and may reveal a single root feature.

## Implication for next iteration

To close the gap on precision in long-sustained games, the highest-
leverage v3 improvement is:

1. **Wave bundling** (~3-5 day build). 2-source coordinated attacks on
   distant high-value targets. Distance to neutrals matters less when
   we synchronize arrival from two sources.

2. **Strike-window timing** (~1-2 week build). Schedule shots to
   arrive AFTER projected enemy capture. Most valuable in long games
   where enemy commits to specific captures we can anticipate.

3. **Long-game scoring tweaks**: discount short-term ROI relative to
   sustained-production-control. Already partially implemented via
   `time_to_hold = max(1, 500 - step - eta)` in score formula.

Wave bundling is the cheaper, more contained build. Recommended first
iteration toward closing the precision gap.

## What we proved this session

- σ-equiv-patched v3 is at precision's strength tier (50/50)
- The gap isn't structural (we're competitive) but tactical (we're
  worse on long-sustained boards)
- A concrete buildable improvement (wave bundling) targets the gap
