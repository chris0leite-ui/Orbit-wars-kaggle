# cProfile post-K1 — kinematic_table wired, seed 1622482326

Date: 2026-05-25. Same harness, same seed as `audit/2026-05-25-consolidation-profile.md` (pre-K1).

## Per-turn timing comparison (with profile overhead, ~30% inflation)

| metric | Pre-K1 | Post-K1 (CONSOLIDATION only) | **Post-K1b (BUILDUP + CONSOLIDATION)** | Δ pre→post-K1b |
|--------|-------:|-----------------------------:|---------------------------------------:|---------------:|
| n_turns | 219 | 216 | 223 | — |
| p50 | 1047.5 ms | 855.2 ms | **824.3 ms** | **-22%** |
| p95 | 1894.0 ms | 1899.5 ms | **918.4 ms** | **-52%** |
| p99 | 2239.5 ms | 2192.5 ms | **963.1 ms** | -57% |
| max | 2515.4 ms | 2536.6 ms | **1014.2 ms** | -60% |
| wall_total | 340.1 s | 325.8 s | 309.2 s | -9% |

**Post-K1 (CONSOLIDATION-only priming)** improved p50 by 18% but left p95 untouched — because BUILDUP turns (steps 0-29) never primed the kinematic_table, so opening_plan's _build_candidates ran on the slow inline path. The hardest 10 turns were ALL in the opening band (12-28).

**Post-K1b (both BUILDUP and CONSOLIDATION primed)** drops p95 by 52% and max by 60%. The hardest 10 turns now spread across the whole game (12, 55, 97, 98, 100, 143, 164, 180) — no opening-band concentration. Under production conditions (no profile overhead), production p95 was 1101 ms; subtracting ~30% overhead suggests **post-K1b production p95 ≈ 645 ms**, well under the 1000 ms Kaggle budget.

## Hardest 10 turns (post-K1b)

| turn | ms |
|-----:|----:|
| 180 | 1014.2 |
| 12  | 999.9 |
| 98  | 963.1 |
| 164 | 951.8 |
| 100 | 943.0 |
| 97  | 942.0 |
| 143 | 940.4 |
| 55  | 934.3 |
| (rest < 934) | |

## cProfile cumulative — top hot functions

| function | Pre-K1 cumtime | **Post-K1b cumtime** | Δ |
|----------|---------------:|---------------------:|---:|
| `predict_relative` (`orbit.py:29`) | 83.8 s | **8.0 s** | **-90%** |
| `predict_fleet_fate` (`trajectory.py:80`) | 116.3 s | **46.2 s** | -60% |
| `propose` (`proposer.py:970`) | 70.8 s | ~40 s | -43% |
| `opening_plan / _build_candidates` | 51.0 s | ~17 s (est.) | -67% |
| `kinematic_table.window` (NEW) | — | ~16 s | new |
| `_table_window_or_none` (NEW) | — | ~17 s | new (mostly window calls) |

The cache fully eliminated 75 s of `predict_relative` own-time. The new `kinematic_table.window` call uses 16 s — net saving ≈ 59 s on this single hot path alone.

## Conclusion

K1 (with both BUILDUP and CONSOLIDATION priming) is a clean wallclock win:

- p95 turn time: -52% (1894 → 918 ms with profile overhead)
- Inner-kernel `predict_relative` cumulative time: -90%
- Production p95 estimated to clear the 1000 ms Kaggle budget by ~350 ms of headroom

Behavior is bit-identical by construction (parity gate `21/21` GREEN; `c48e143`'s 564 byte-identical FleetFate assertions stand). Verification 4 (Wilson-gated A/B) confirms no behavioral regression next.
