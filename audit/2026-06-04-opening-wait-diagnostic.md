# 2026-06-04 — Opening "we wait too long" is the VALUE FUNCTION, not the horizon

Branch: `claude/kaggle-submission-strategy-JzIAr`. Diagnostic:
`scripts/opening_starvation.py`. Trigger: PI replay of live loss on seed
**722289020** (perimeter-ring map, big central sun) — "we wait too long in the
opening, probably because the landscape is so sparse that most targets are out of
reach within K steps, so we don't launch."

## Hypothesis under test

Horizon-K starvation: the proposer hard-drops any launch with arrival ETA > K
(`proposer.py:1162  if eta > _k_tgt: continue`), there is NO fallback, so on a
sparse map where every neutral sits beyond K the agent generates **zero**
candidates and sits idle. The shipped champion runs adaptive-K (K_OPEN=20 → floor
10 by step 30); the adaptive-K audit itself notes opening median neutral ETA ≈ 22,
i.e. K_OPEN=20 is *below* the median — so the fix looked under-powered on the
sparse tail.

## Result — hypothesis REFUTED on two independent measurements

**1. Maps are not nearest-neutral sparse (cheap step-0 scan, 160 seat-boards over
seeds 722289000–079).**
- nearest-neutral ETA > 10 (past static floor): **1%**
- nearest-neutral ETA > 20 (past adaptive K_OPEN): **0%**
- zero neutral reachable within K_OPEN=20: **0%**

The map *looks* sparse, but that's the *cross-map* distance around the sun.
Adjacent-ring expansion is cheap (ETA 4–10). The horizon never zeroes our
candidates.

**2. We wait by CHOICE, not by starvation (full opening trace, seed 722289020,
focal P0, steps 0–30).**

```
opening 31 turns:  LAUNCHED on 4  ·  WAITED-with-candidates on 27  ·  STARVED-by-horizon on 0
```

Every turn the proposer offered 2–12 candidates; by mid-opening 8–12 candidates and
12–32 reachable expansion targets (reach@20–reach@30). The agent fired on **4 of
31** turns. **Zero** turns were horizon-starved. It is hoarding ships — the
chooser/value function declines the available launches.

## Reframe

The lever is **not** the horizon constant K (decisively refuted: 0/31 starved,
0% maps past K_OPEN). It is the **value function's early-expansion appetite** —
27/31 opening turns we sat on launchable candidates.

## What is NOT yet established (do not over-read)

- This is **local self-play** (champion vs a champion bundle). Both sides wait
  symmetrically, so locally the waiting is ~even/correct. Proven: *we* wait 27/31.
  NOT proven: that the waiting is *why* we lost to Merchant API — that needs an
  **aggressive early-expander** opponent to show the waiting is exploited.
- **"Launch more early" has already regressed** (plan file / prior handover) — but
  almost certainly measured in self-play, where launching more vs an equally-waiting
  opponent just overextends. Whether it regresses against an aggressive expander too
  is the **untested question**, and the Merchant loss is exactly that case.

## Decisive next measurement (cheap, no build)

Pit the agent vs a deliberately **aggressive early-expander** (not self-play) and
measure: (a) our opening launch-rate vs theirs, (b) territory/production gap by step
30, (c) whether a higher early-launch appetite (lower chooser launch threshold OR an
early-game expansion bonus) **wins specifically vs that class** while staying neutral
in self-play. That isolates "is our waiting exploited" from "does launching more just
overextend" — which the prior self-play regression could not separate (Rule 41
confound: the regression cohort was all self-play).
