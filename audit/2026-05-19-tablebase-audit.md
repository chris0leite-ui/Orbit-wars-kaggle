# 2026-05-19 — Tablebase Audit (trajectory_roi v3.1 vs cluster solver)

> Phase A.5 deliverable. See `/root/.claude/plans/optimized-questing-shell.md`.

## Summary

Total clusters audited: **29**

| Class | Count | Rate |
|---|---|---|
| AGREE-IDLE | 16 | 55% |
| AGREE-LAUNCH | 0 | 0% |
| PARTIAL-SIZING | 0 | 0% |
| DISAGREE-OVER | 13 | 45% |
| DISAGREE-UNDER | 0 | 0% |
| DISAGREE-TARGET | 0 | 0% |

## What each class means

- **AGREE-IDLE**: both heuristic and solver choose no launch.
- **AGREE-LAUNCH**: both launch at the same target, total ship count within ±10%.
- **PARTIAL-SIZING**: both launch at the same target but ship-count differs > 10%.
- **DISAGREE-OVER**: heuristic launches; solver chooses IDLE.
- **DISAGREE-UNDER**: solver launches; heuristic chooses IDLE.
- **DISAGREE-TARGET**: both launch but at different targets.

## Caveats

Solver depth is bounded (typical search depth 6-8). Trajectory_roi's value function uses K_HORIZON=30 forward projection. Disagreements may reflect bounded-depth solver missing deep payoffs (e.g. production accumulated over 20+ post-capture turns) rather than heuristic bugs. Higher-confidence audits require either (a) deeper search with shaped leaf rewards, or (b) target-specific deeper probes for any DISAGREE-OVER cluster the heuristic relies on.

## Examples per class

### DISAGREE-OVER

- replay `episode-76990778-replay.json` step=20 seat=0 planet_ids=[0, 28, 34] depth=8
  - heuristic: `[[0, -1.7115225395068472, 11], [28, -2.780441683408587, 5]]`
  - solver:    `[]`  value=74.0
- replay `episode-76990778-replay.json` step=20 seat=1 planet_ids=[3, 27, 31] depth=8
  - heuristic: `[[31, 1.5874628207908135, 14], [3, 2.2247331382769926, 14]]`
  - solver:    `[]`  value=63.0
- replay `episode-76990778-replay.json` step=20 seat=1 planet_ids=[3, 31, 33] depth=8
  - heuristic: `[[3, 1.4300701140829453, 16]]`
  - solver:    `[]`  value=63.0
