# Phase C n=8 A/B results

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Run**: `scripts/phase_c_ab.py` (8 seeds × 2 sides × 3 pairings = 48 games,
4 workers, ~20 min wallclock)
**Artifact**: `audit/tournaments/20260518T133925Z.json`

## Aggregate results

| Matchup | Wins/N | Winrate | Wilson Wlo (95%) | Whi |
|---|---|---|---|---|
| **bundle vs v7_0** | 13/16 | **0.812** | **0.570** | 0.934 |
| bundle vs baseline | 2/16 | 0.125 | 0.035 | 0.360 |
| baseline vs v7_0 (sanity) | ~14/16 | ~0.87 | — | — |

## Timing (per-side p95)

| Pairing | p0 p95 | p1 p95 |
|---|---|---|
| bundle (P0) vs v7_0 (P1) | 752ms | 766ms |
| v7_0 (P0) vs bundle (P1) | 691ms | 752ms |
| bundle (P0) vs baseline (P1) | 751ms | 338ms |
| baseline (P0) vs bundle (P1) | 348ms | 751ms |

Bundle stays at its 750ms internal budget cap. No turn breached
1000ms across the 48-game run. Baseline runs leaner (~340ms).

## Phase C gate decision

Per the foamy-pondering-floyd plan:
> Gate: **Wlo > 0.40 on BOTH**. Submission decision: if BOTH gates
> pass at n=32, submit.

- vs v7_0: **PASS** (Wlo=0.570)
- vs baseline: **FAIL** (Wlo=0.035)

**Do NOT submit per the literal gate.**

## What we learned

1. **cands=5 was a real fix.** Bundle went from 0/0 vs v7_0 (pre-cands fix, eliminated turn 121) to 13/16 at the same matchup. That's the largest single-knob improvement of this multi-session arc.
2. **Baseline is structurally stronger than v7_0.** Baseline beats v7_0 ~87% in the sanity-check pairing, and crushes bundle 14/16. This is true regardless of bundle's improvements.
3. **Bundle is now a v7_0-class agent.** Slots between v7_0 and baseline on the quality scale. Probable live μ-rating: somewhere between v7_0 and baseline's positions on the LB.

## Strategic position

The chooser/scorer axis has had many variants this multi-session arc.
cands=5 was the first that produced a real win (vs v7_0). Per Rule 37,
that resets the failed-variant counter — but only on the v7_0 dimension.
The bundle-vs-baseline gap is a NEW isolated problem.

## Recommended next step (PI to ratify)

One of:

A. **Submit anyway** as a calibration probe. Last-2 live submissions are
   v4 (~1141) and composite_a2 (~1145). If bundle's true μ ranks above
   v7_0-class agents on the LB, it might net positive even though
   baseline crushes it. Cost: displaces one good submission for 24h.

B. **Isolate bundle-vs-baseline gap first** via the same single-turn
   diagnostic from earlier (scripts/diag_single_turn.py works for any
   opponent path). ~30 min. If it's another load-bearing knob, fix it
   and re-A/B. If it's deep, that's the pivot signal.

C. **Both in parallel**: submit AND diagnose. Tradeoff: lose a ladder
   slot for 24h whether or not the diagnostic finds something fast.

My lean: B. We just learned cands=5 was a single hidden knob worth a
massive win-rate swing. Same probability mass on bundle-vs-baseline.
A 30-min diagnostic is cheap relative to a 24h LB experiment.

## Artifacts

- `audit/tournaments/20260518T133925Z.json` — per-game records
- `scripts/phase_c_ab.py` — A/B harness
- `scripts/diag_single_turn.py` — root-cause discriminator (reusable
  for the bundle-vs-baseline gap diagnosis)
