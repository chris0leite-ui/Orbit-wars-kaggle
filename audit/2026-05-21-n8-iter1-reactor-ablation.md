# Iter 1 — Reactor-aware ablation (n=8 A/B vs sub-52827111)

Date: 2026-05-21
Branch: `claude/review-skills-improvements-moKOR`
Setup: targeted file-import of sub-52827111 source from `origin/claude/audit-workflow-performance-btjeK`
(commits `d642593` + `9a45fea` + `3f123c3`).

## Goal

Diagnose which sub-part of the stacked "comet-aim + reactor-aware" push
(sub 52827111, μ=1122, live −13 μ vs prior submission 52811320) is the
regressor. Env-var ablation with no code change.

## Iteration baseline

`submissions/iter_baseline.py` — `agents/baseline` bundled with defaults
(all of: BASELINE_COMET_AIM=on, PROPOSER_TRAJECTORY_FILTER=on,
PROPOSER_DRAIN_FILTER=on, PROPOSER_HOLD_FEASIBILITY=on, PROPOSER_COST_PARITY=on,
PROPOSER_REACTOR_CANDIDATES=on). Identical mechanism state to sub 52827111.

## Variants

| Variant | Env vars (deltas only) | Bundle file |
|---|---|---|
| 1a | `PROPOSER_COST_PARITY=off PROPOSER_REACTOR_CANDIDATES=off` (comet-aim solo) | `submissions/iter1a_comet_solo.py` |
| 1b | `PROPOSER_COST_PARITY=off` (Part A off; reactor candidates ON) | `submissions/iter1b_no_costparity.py` |
| 1c | `PROPOSER_REACTOR_CANDIDATES=off` (Part B off; cost-parity ON) | `submissions/iter1c_no_reactor.py` |

Each variant bundled as a *separate file* (env-var defaults baked in at
bundle time via the chooser's setdefault path) so the A/B opponent is
deterministic — the harness uses CRN.

## Results — first run (parallel, CPU-contended)

All three A/Bs were launched simultaneously, sharing 8-worker pools each
→ 24 worker processes competing for cores. Wallclock numbers below
reflect the contention, NOT the deployed bundle's actual per-turn cost.

| Variant | Wins | Wlo | Whi | Verdict (gate 0.47) | Focal p50/p95/max (ms) |
|---|---:|---:|---:|---|---|
| **1b** | **12/16 (75.0 %)** | **0.505** | 0.898 | **PASS (borderline)** | 770 / 1248 / 2486 |
| 1a | 11/16 (68.8 %) | 0.444 | 0.858 | INCONCLUSIVE (Wlo < 0.47) | 787 / 1238 / 2592 |
| 1c | 7/16 (43.8 %) | 0.231 | 0.668 | parity (no signal) | 717 / 1131 / 2268 |

n is fast.py's seeds×seats convention: `--max-seeds 8` produces 16 games
(8 seeds × both sides of CRN). Wallclock per A/B ≈ 16 min under contention.

## Diagnostic read

- **Part A (`COST_PARITY_MARGIN=0.7` cost-parity filter) is net-harmful.**
  Removing it (variant 1b) is the best lift. Variant 1c (keep Part A,
  drop Part B) sits at parity — Part A alone is approximately worthless.
  Consistent with the CLAUDE.md Rule 40 critique that a hard-coded
  `< 0.7 × ours` constant is a restriction-tuning anti-pattern, not a
  modeling fix.
- **Part B (`_enumerate_reactor_candidates`) is net-positive.** Removing
  it (variant 1a) drops from 12/16 to 11/16 vs the deployed baseline.
  So Part B is contributing some lift; Part A's drag was cancelling it
  on the live ladder, producing the observed −13 μ net regression.
- **The live regression vs sub 52811320 (μ=1135) is explained.** Original
  push stacked Part A + Part B both on; Part A's hard-coded filter
  rejected genuinely good launches; Part B's recapture lift wasn't
  enough to offset; net live μ dropped.

## Confounds before promotion

1. **CPU contention taint.** Focal p95 ≈ 1248 ms is OVER the env's 1000 ms
   actTimeout — the engine would have dropped 5 % of turns. The
   deployed bundle's Kaggle-measured p95 was 455 ms (commit `037009b`).
   The relative ranking 1b > 1a > 1c is plausible under shared
   contention (both agents in each pair were equally affected), but
   absolute win counts could shift by 1-2 under clean conditions.
2. **12/16 = Wlo 0.505 is directional but tight.** Plan-spec strong
   promotion = ≥7-of-8 pairs = 13-of-16 in fast.py units (Wlo 0.529).
   12/16 falls just below that threshold. Per the PI directive
   (n=8 max, significant lift required), this is "iterate or
   re-confirm," not "promote."
3. **No corresponding panel A/B.** Run was only against the iteration
   baseline; Rule 43 requires per-opponent Wilson-lo ≥ 0.55 on the
   3-opponent panel before any submit decision.

## Next action

Serial re-run of variant 1b (alone, no CPU contention) to confirm the
12/16 holds under clean conditions and to get an honest wallclock.

- If clean-run gives ≥ 13/16 with focal p95 < 600 ms → directional winner;
  queue Iter 3 (enemy-fleet intercept in `predict_fleet_fate`) layered on top.
- If clean-run drops to ≤ 10/16 → Part A's harmfulness call was overstated by
  contention; reconsider whether to act on this iter at all.
- If clean-run lands at 11-12/16 → still borderline; either accept (PI
  call) or run Iter 3 in parallel on the deployed baseline (not on 1b)
  to find a stronger candidate.

No `kaggle competitions submit` until the n=32 + panel gate (Rule 43/45)
clears on a candidate, AND the Rule 42 push-claim board entry is filled.
