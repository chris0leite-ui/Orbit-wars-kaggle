# cProfile — `agents/buildup_planner` vs `phi1_only`, seed 1622482326

Date: 2026-05-25. Profile harness: `scripts/profile_consolidation.py` (temp, deleted post-review). Game ran 219 turns (early termination, not 500). Profile overhead ≈ 30% inflation — real production p95 was 1101 ms; here p50 = 1047 ms.

## Per-turn timing (focal, with profile overhead)

| metric | ms |
|--------|------|
| n_turns | 219 |
| p50 | 1047.5 |
| p95 | 1894.0 |
| p99 | 2239.5 |
| max | 2515.4 |
| wall_total | 340.1 s |

## Hardest 10 turns — ALL openings

| turn | ms |
|------|------|
| 12   | 2515.4 |
| 28   | 2360.1 |
| 14   | 2239.5 |
| 15   | 2124.4 |
| 13   | 2110.0 |
| 23   | 2036.4 |
| 27   | 2035.0 |
| 16   | 1997.2 |
| 26   | 1996.1 |
| 25   | 1977.8 |

**Finding A.** Every one of the slowest 10 turns is in the opening (steps 0-29), where the BUILDUP MILP runs. `opening_plan()` is being called 30 times per game (post-`843cc35` transition fix), each call running ~1.7s under profile overhead (≈1.2s production-equivalent).

## cProfile cumulative top-20 (in 219-turn run)

| ncalls | cumtime (s) | per-call (ms) | function |
|-------:|------------:|--------------:|----------|
| 219 | 205.8 | 940 | `agent` (main entry) |
| 189 | 154.5 | 818 | `consolidation.step` |
| 189 | 154.4 | 817 | `agents/baseline/main.py:agent` |
| **19210** | **116.3** | **6.05** | **`predict_fleet_fate`** |
| 230520 | 93.9 | 0.41 | `trajectory.py:175 <listcomp>` (inner) |
| **48689189** | **83.8** | **0.0017** | **`predict_relative` (orbital kernel)** |
| 154 | 79.2 | 514 | `choose_trajectory` |
| 154 | 70.8 | 460 | `propose` |
| 1972 | 67.9 | 34.4 | `score_candidate_v4` |
| **30** | **51.0** | **1701** | **`opening_plan` (`_build_candidates`)** |
| 54624 | 49.6 | 0.91 | `fs_step` |
| 54624 | 48.8 | 0.89 | `interpreter` |
| 54482 | 17.8 | 0.33 | `opp_actions_for_snap` |
| 54482 | 17.2 | 0.32 | `lite_greedy_policy` |
| 22070631 | 13.2 | — | `interpreter.swept_pair_hit` |
| 142 | 11.0 | 78 | `build_trajectory_baseline` |

## cProfile tottime (own work) top-10

| ncalls | tottime (s) | per-call (μs) | function |
|-------:|------------:|--------------:|----------|
| 48,689,189 | 59.5 | 1.2 | `predict_relative` |
| 54,624 | 21.6 | 396 | `interpreter` |
| 230,520 | 14.4 | 62 | `trajectory.py:175 <listcomp>` |
| 54,482 | 14.0 | 257 | `lite_greedy_policy` |
| 22,070,631 | 13.2 | 0.6 | `swept_pair_hit` (interpreter) |
| 19,210 | 11.9 | 618 | `predict_fleet_fate` (own work) |
| 7,077,561 | 8.5 | 1.2 | `swept_pair_hit` (lib/aim.py) |

## Findings — leading time consumers

**FINDING-1 (PERF, high impact).** `predict_fleet_fate` accumulates 116 s across 19,210 calls — **~88 calls per turn**, ~530 ms/turn average WITHOUT profile overhead correction. This is **~50% of the per-turn budget**. Each call walks `predict_relative` ~2500 times (48.7M / 19.2k = 2532). Call sites (Stage 2 will verify):
   - `score_candidate_v4` admissibility filter (`chooser_trajectory.py:533`) — runs per `wait_N==0` candidate
   - `proposer.propose` admissibility filter (`proposer.py:~1040`)
   - `opening_planner._build_candidates` (`opening_planner.py:409`) — MILP candidate enumeration
   - `cheap_marginal_value` indirectly via `aim_and_eta` (probably)

   The same (src, tgt, ships, angle) signature is likely computed 2-3× per turn at different sites within the same world snapshot. **Within-turn memoisation of `predict_fleet_fate` keyed by (src.id, tgt.id, ships, angle) is the highest-ROI optimization** — preserves behavior exactly (same function, same inputs → same output), cuts ~50% of trajectory CPU.

**FINDING-2 (PERF, opening-specific).** `opening_plan` averages 1.7 s/call × 30 calls = 51 s of the 340 s focal wall time. The cache experiment failed because **caching the schedule only saved ONE of the 30 calls' work** — the cost is in `_build_candidates`, which runs the trajectory admissibility on every candidate. The MILP itself (PuLP solve) is cheap; the candidate enumeration is expensive (same hot `predict_fleet_fate` calls). **Lifting Finding 1 (memoise predict_fleet_fate) would also fix this**, because `_build_candidates` is the heaviest predict_fleet_fate caller.

**FINDING-3 (PERF, secondary).** `fs_step` accumulates 49.6 s — 54,624 calls = 250 calls per turn. This is the rollout in `score_candidate_v4` × 1972 candidates × ~28 ticks. Each call is ~0.9 ms. **Reducing horizon or candidate count helps linearly**, but Finding 1 has lower behavioral risk.

**FINDING-4 (DESIGN, observation).** `lite_greedy_policy` (the opp model) runs 54,482 times — once per `fs_step` inside `opp_actions_for_snap`. That's correct but expensive (Θ(|planets|+|fleets|) per call). **Caching by world hash is risky** (the world mutates per fs_step inside the rollout) but a faster opp model (or O(1) policy from a 1-step lookahead cache at the start of each candidate) could halve `opp_actions_for_snap` cost.

**FINDING-5 (CORRECTNESS, low impact).** 219-turn early termination. Game ended before turn 500. Sub-question for the user: is the opp winning or our agent winning at step 219? This affects which side's CONSOLIDATION pattern we're optimizing — but not the timing data itself.

## Stage 1 conclusion

The dominant hot path is the trajectory layer (`predict_fleet_fate` → `predict_relative`). Two of the four findings are PERF candidates that preserve behavior:

- **Finding 1 (memoise `predict_fleet_fate` within a turn)**: ~50% cumulative time saving expected, zero behavior change.
- **Finding 2** is a corollary of Finding 1; the opening-turn slowdown will fall out automatically.

Stage 2 next: walk the actual source files to verify call sites, identify any closed-track entanglement, and check whether the same-tuple-different-result risk exists (e.g., per-tick world mutation invalidating the cache key).
