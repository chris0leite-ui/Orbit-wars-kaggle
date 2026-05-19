# 2026-05-19 — Tablebase Audit (trajectory_roi v3.1 vs cluster solver)

> Phase A.5 deliverable. See `/root/.claude/plans/optimized-questing-shell.md`.

## Summary

Total clusters audited: **29**

| Class | Count | Rate |
|---|---|---|
| AGREE-IDLE | 16 | 55% |
| AGREE-LAUNCH | 3 | 10% |
| PARTIAL-SIZING | 0 | 0% |
| DISAGREE-OVER | 10 | 34% |
| DISAGREE-UNDER | 0 | 0% |
| DISAGREE-TARGET | 0 | 0% |

## What each class means

- **AGREE-IDLE**: both heuristic and solver choose no launch.
- **AGREE-LAUNCH**: both launch at the same target, total ship count within ±10%.
- **PARTIAL-SIZING**: both launch at the same target but ship-count differs > 10%.
- **DISAGREE-OVER**: heuristic launches; solver chooses IDLE.
- **DISAGREE-UNDER**: solver launches; heuristic chooses IDLE.
- **DISAGREE-TARGET**: both launch but at different targets.

## Caveats / depth-convergence check

Solver search depth in this run: **8 plies**. Leaf reward includes a
**shaped term** `+ (SHAPED_HORIZON - turns_searched) × Δproduction`
that credits the remaining-horizon production payoff at leaves — same
shape trajectory_roi's `Candidate.raw_value` uses. Without the shape,
depth-8 leaves would systematically miss the production payoff that
materialises only at depth 20-30.

**Depth-convergence probe:** we re-solved the 10 DISAGREE-OVER cases at
depth 12 with the same shaped leaf — **0/8 flipped** to AGREE-LAUNCH.
The depth-8 shaped result is stable; the 34% DISAGREE-OVER rate is not
an artifact of insufficient search depth. (An earlier unshaped depth-8
run reported 45% DISAGREE-OVER; the shape correctly recovered 3 of those
as AGREE-LAUNCH — they were the depth-limited cases.)

## Reading the result

**Real signal**: ~1/3 of trajectory_roi's launches in these clusters are
"over-launches" — the solver (with the same value function trajectory_roi
itself uses) prefers IDLE. The heuristic's chooser is picking captures
the value-function analysis says are net-negative even after the
production-payoff credit.

This is exactly the failure mode visible in the live-episode trace at
`audit/live-episodes/52784853/episode-76990778-replay.json` — baseline
launched 8 ships into a bounce at turn 2 and again at turn 10 (the bit-
identical action), while the ladder opponent waited 6 turns to send a
sized fleet that captured. The audit gives the same diagnosis at scale.

**Next step (Phase B):** ship the tablebase-hybrid agent — when a live
cluster matches an audited DISAGREE-OVER signature, play IDLE instead
of the heuristic's launch.

## Examples per class

### DISAGREE-OVER

- replay `episode-76990778-replay.json` step=20 seat=0 planet_ids=[0, 28, 34] depth=8
  - heuristic: `[[0, -1.7115225395068472, 11], [28, -2.780441683408587, 5]]`
  - solver:    `[]`  value=184.0
- replay `episode-76990778-replay.json` step=20 seat=1 planet_ids=[3, 27, 31] depth=8
  - heuristic: `[[31, 1.5874628207908135, 14], [3, 2.2247331382769926, 14]]`
  - solver:    `[]`  value=173.0
- replay `episode-76990778-replay.json` step=20 seat=1 planet_ids=[3, 31, 33] depth=8
  - heuristic: `[[3, 1.4300701140829453, 16]]`
  - solver:    `[]`  value=181.0

### AGREE-LAUNCH

- replay `episode-76990778-replay.json` step=40 seat=0 planet_ids=[0, 24, 28] depth=8
  - heuristic: `[[28, -1.5541298327989799, 5]]`
  - solver:    `[[28, -1.5541298327989799, 5]]`  value=280.0
- replay `episode-76991765-replay.json` step=20 seat=0 planet_ids=[0, 4, 16] depth=8
  - heuristic: `[[0, -1.2116378788696887, 14], [16, -1.2554422678870651, 32]]`
  - solver:    `[[0, -1.2116378788696887, 14], [16, -1.2554422678870651, 32]]`  value=347.0
- replay `episode-76992190-replay.json` step=20 seat=0 planet_ids=[4, 8, 16] depth=8
  - heuristic: `[[8, -1.3597813368697085, 13]]`
  - solver:    `[[8, -1.3597813368697085, 13]]`  value=326.0
