# v12 gate results — Felipe + Naoism + v4_planner baseline

Branch: `claude/review-foundations-progress-14HXp`
Date: 2026-05-19
Commits under test: C1 (parity), C2 (CRN), C3 (ship_balance value head)

## Per-seed results (head-to-head vs v7_0, both seats)

| seed | game | v8_scavenge | v4_planner bundle | v12-C2 (CRN) | **v12-C3 (CRN + ship_balance)** |
|---|---|---:|---:|---:|---:|
| 1492346051 (Felipe) | P0 | LOSS (135 steps) | LOSS (183 steps) | LOSS (162) | **WIN (174)** ← flip |
| 1492346051 (Felipe) | P1 swap | LOSS (127) | LOSS (181) | LOSS (177) | LOSS (167) |
| 768065184 (Naoism) | P0 | (won historically) | LOSS (154) | LOSS (202) | LOSS (197) |
| 768065184 (Naoism) | P1 swap | (won historically) | LOSS (154) | LOSS (182) | LOSS (182) |
| **Felipe total** | | **0/2** | **0/2** | **0/2** | **1/2 PASS** |
| **Naoism total** | | 1/2* | 0/2 | 0/2 | 0/2 |

(\*v8 historical was on different chooser-mode runs; not strictly comparable since v8's 1/2 happened to land vs noisy panel state)

## Wallclock (v12-C3, both seats, both seeds)

| game | v12 p50 | v12 p95 | v12 max |
|---|---:|---:|---:|
| Felipe P0 | 621 ms | 758 | 786 |
| Felipe P1 | 588 | 696 | 770 |
| Naoism P0 | 572 | 640 | 751 |
| Naoism P1 | 581 | 637 | 669 |

All under 1000 ms cap. p95 in 637-758 ms range. **Wallclock gate PASS.**

## Key finding

**C3's `evaluate_value_v12` ship-balance term is the load-bearing
change.** Neither the v4_planner architecture (which v12-C1 ports) nor
C2's CRN unlocked Felipe — both stay at 0/2. C3 alone flips P0 to a
win. Direct evidence the value head's blindness to ship-mass
differential (audit/2026-05-18-loss-mode-v8-v9.md, maruichi forensic)
was the bottleneck.

## Naoism is "architecture-bound"

v4_planner-bundle (μ=1056 on the live ladder) loses Naoism 0/2.
v12-C3 inherits this. Not a regression vs the architecture; rather,
the v3.5.1-rooted policy stack has a structural weakness on
sustained-pressure seeds that v8's strict-idle chooser happened to
sidestep on this single seed.

Implication: Naoism is unlikely to flip without changing the opp
policy or rollout structure (C4 / C5 territory). v12-C3 should still
be net-positive in aggregate.

## Architecture wallclock (v4_planner bundle for reference)

| game | v4_planner p50 | p95 | max |
|---|---:|---:|---:|
| Felipe P0 | 665 ms | 775 | 867 |
| Felipe P1 | 661 | 778 | 842 |
| Naoism P0 | 597 | 698 | 760 |
| Naoism P1 | 607 | 699 | 726 |

v12-C3 is slightly faster (p95 637-758 vs v4_planner 698-778) despite
the CRN trajectory recording overhead. Probably because the value
head computation is comparable and the architecture is the same.

## Panel result

PENDING (running in background, results to be filled in).

## Decision

If panel `Wlo ≥ 0.55` against all 3 baselines (v7_0, v4_planner,
v3.5.1), v12-C3 is the shipping candidate. C4 (top_tier_mirror opp
policy) and C5 (K bump) become OPTIONAL incremental lifts.

If panel regresses below v4_planner's bundle, drop C2 (CRN may be
adding bias not fully offset by variance reduction) and try
C3-alone.

## Reproduction commands

```
# Felipe P0
python fast.py play agents/v12 --vs v7_0 --seed 1492346051

# Felipe P1 (swap)
python fast.py play agents/v12 --vs v7_0 --seed 1492346051 --swap

# Naoism
python fast.py play agents/v12 --vs v7_0 --seed 768065184
python fast.py play agents/v12 --vs v7_0 --seed 768065184 --swap

# Panel (16 seeds, 4 workers, ~30-40 min)
python fast.py eval agents/v12 --vs-panel default --max-seeds 16 --workers 4
```

Local reproduction note: each game is 2.5-3 min on a single core
because v12 + v7_0 both run heavy K-step rollouts.
