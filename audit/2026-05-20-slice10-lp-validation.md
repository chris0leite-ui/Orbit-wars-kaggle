# Slice 10 validation — 2026-05-20 — joint LP, same regression as differential

> Commit `e7217fe` — `agents/baseline/chooser_lp.py`. The big
> architectural shift: replace greedy per-candidate emit with
> Hungarian bipartite assignment over the whole turn's move-set.
> 38 unit tests + 3 property tests + 130 cross-module regression
> pass cleanly.

## Step 1 — Bench parity (5 games vs random)

```
focal lp: n=491 turns
  p50=17ms  p95=70ms  p99=175ms  max=248ms  over_1000ms=0
  verdict: PASS
```

Fast like differential. All 5 random games won.

## Step 4 — Single game (seed 0 vs trajectory baseline)

Outcome: TIE (1, 1) at 500 turns.

## Step 2 — Small A/B (n=16 vs trajectory baseline)

```
n=16  wins=3/16  (18.8%)  Wlo=0.066  Whi=0.430  FAIL  (Whi<0.55)
focal turn-ms  p50=40  p95=513  max=909
total elapsed 582.3s
```

Same dismal result as Slice 8c. The joint LP didn't fix anything
the per-candidate scoring didn't already break.

## Comparison across all analytical-chooser attempts

| Slice | Approach | Wins (n=16) | Wlo | max-ms |
|---|---|---|---|---|
| 8 | Differential leaf eval (per-candidate) | 6/16 (37.5%) | 0.185 | 810 |
| 8c | + wait_N filter | 3/16 (18.8%) | 0.066 | 704 |
| 9 | + migrations | 4/16 (25.0%) | 0.102 | 734 |
| **10** | **Joint LP** | **3/16 (18.8%)** | **0.066** | **909** |

Even with the architectural fix that the deep-diagnosis identified
(joint optimization instead of greedy per-candidate), the
analytical chooser loses by ~3× to the noisy-rollout trajectory.

## Diagnosis (deeper than the chooser)

The session's deep-architecture diagnosis was: "we're plumbing
analytical engines into a chooser that's per-candidate atomic
in nature." Slice 10 fixed that — and the chooser still loses.
So the problem runs deeper still.

Three remaining hypotheses (in order of plausibility):

1. **Value calibration mis-tuning**. Capture values come from
   `_w1_value_bounds` (PV × production × hold-window). Reinforce
   values come from 2 × production × pv. Migration values come
   from `(post_EV − pre_EV − src_EV) × pv_horizon`. These
   formulas were each tuned in isolation; their joint scale
   may be wrong. The LP picks the best ALLOCATION subject to a
   noisy objective.

2. **Single-turn horizon is too short**. The LP optimizes for
   "this turn's best move-set given current state." Trajectory's
   rollout simulates K ticks of future play; its leaf encodes
   downstream consequences. An analytical equivalent would be
   multi-turn LP (state expands exponentially) or dynamic
   programming (planning over a coarse state lattice).

3. **The candidate space is closed.** Even an optimal LP picks
   only from candidates the proposer + migration solver emit.
   If trajectory's rollout finds a winning move via simulation
   that no analytical generator surfaces, no LP can find it.
   This is the user's earlier insight: missing candidate
   classes. We added one (migration); there may be more
   (multi-source coalitions, time-shifted joint launches,
   strategic ship buildups, etc).

## Decision

**STOP this slice line.** The analytical-native path's biggest
architectural lever (joint optimization) has been pulled and the
chooser still loses by ~3×. Further iteration on the same axis
is unlikely to be productive without:

- Multi-turn LP / dynamic programming (large architectural shift,
  exponential state space, not a single-session deliverable).
- A complete value-calibration audit (capture / reinforce /
  migration values normalized to a common scale).
- A much richer candidate space (more analytical generators
  beyond migration).

Each of these is a multi-slice undertaking and the marginal
value-vs-cost trade has clearly degraded. The trajectory chooser
(noisy rollout, μ=1118.8) remains the production default. The
analytical work in this session (10 slices, ~3500 LOC, 130+
tests, 5 audit docs) is preserved as research-only code on the
dev branch — opt-in via env vars.

## Recommendations

1. **Production unchanged**: `BASELINE_CHOOSER=trajectory`
   remains default. Rolling-pair floor μ=1118.8 preserved.
2. **All analytical pieces preserved**: predicates, differential,
   migration solver, strategic LP, joint LP chooser, layered
   composition — all stay in tree as opt-in research substrate.
3. **Next session strategic question for PI**:
   - Continue analytical work on a different axis (multi-turn,
     value calibration, candidate-space enrichment)? OR
   - Pivot to trajectory-side improvements (better opp model,
     better leaf head, leaf-eval cache)? OR
   - Park analytical experiments entirely; focus on submission
     selection / ladder strategy with the existing production
     chooser?
4. **Documentation**: this slice closes the analytical-chooser
   line of investigation. The honest finding is that closed-form
   correctness alone — even with the right control structure —
   doesn't replace what implicit planning gets you. The vision
   isn't wrong; the road is longer than 10 slices.
