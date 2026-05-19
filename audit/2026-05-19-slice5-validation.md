# Slice 5 validation — 2026-05-19

> Commit `e5ca178` — bounded-interval scoring + per-source W1
> dominance commit. Validated per Rule 41 (inspect first, small A/B
> second).

## Single-game introspect (seed 42, vs trajectory)

**Outcome: LOSS -1/+1** (flipped from Slice 4's WIN on same seed)

| Metric | Slice 4 | Slice 5 |
|---|---|---|
| W1/turn | 1.26 | 0.72 (42% drop) |
| W2/turn | 0.36 | 0.28 |
| Inner emits/turn | 0.91 | 0.68 |
| Backstop appended/turn | 0.32 | 0.18 (44% drop) |
| Backstop rate | 20.1% | 18.1% |

Math is working as designed: dominance check is stricter, fires
fewer commits. Single-seed loss is noise.

## Small A/B (n=16, vs trajectory baseline)

| | Slice 4 | Slice 5 |
|---|---|---|
| Wins | 9/16 (56.2%) | 9/16 (56.2%) |
| Wlo | 0.332 | 0.332 |
| p50 turn-ms | 194 | 204 |
| p95 turn-ms | 579 | 582 |
| **max turn-ms** | **1535** | **861** |

Identical win count on the same 16 seeds — the dominance check
didn't materially shift game outcomes at this scale. But **max
wallclock dropped 674ms** (1535 → 861), cleanly under the 1000ms
env cap. Slice 5's stricter gate fires fewer commits, reducing
work on the worst turns.

## Decision

Proceed to Slice 6 (reachability-graph LP). Slice 5 is at minimum
a wallclock improvement; no win-rate regression. The combined
Slice 4 + Slice 5 architecture stays viable; Slice 6 will test
whether a long-horizon strategic anchor adds meaningful lift on
top.

Production unchanged: `BASELINE_CHOOSER=trajectory` remains default.
