# Why does the no-belief chooser beat lite_greedy on seed 3493?

Date opened: 2026-05-30.
Branch: claude/kaggle-submission-review-gZsCu.

## The observation

In paired-seed asymmetric n=4 A/B (anchor + lite_greedy vs anchor +
no-launch-baked), the focal-as-P0 result was:

| Seed | Winner |
|---|---|
| 2083 | P0 (lite_greedy) |
| 1649 | P0 (lite_greedy) |
| 5199 | P0 (lite_greedy) |
| **3493** | **P1 (no-launch)** |

Seed 3493 is the anomaly. The same seed:
- Was won by lite_greedy against nearest (3.1: 3-2 lite_greedy total).
- Was won by lite_greedy against mirror (3.3: 5-0 lite_greedy total).
- Is lost by lite_greedy here against the *literal absence* of an opp
  belief.

## Why this is interesting

If the no-belief chooser wins on seed 3493 specifically, the
mechanism likely is:
1. The board geometry of seed 3493 makes lite_greedy *over-predict*
   the threat to one or more of P0's planets.
2. The chooser, seeing high projected threat from lite_greedy's
   reactive opp launches in its rollouts, defensively reserves
   ships that aren't actually needed (the real opp wouldn't actually
   launch from those distances).
3. The no-belief chooser sees the same board with no projected threat
   → commits the ships to offense → wins the game P0 with belief
   loses.

If this is the mechanism, **it's the smoking gun for PM3's "expects
opponents from everywhere" diagnosis** — a specific board where the
over-prediction is bad enough to flip the outcome, vs the other
seeds where lite_greedy's threat model is closer to reality.

## What it would take to answer

Single-game trace, both variants playing seed 3493, with per-turn
logging of:
- The chooser's per-candidate Δ (so we see which candidates
  lite_greedy filters out as "would expose us to opp counter").
- The reactive opp launches the rollouts simulate (specifically:
  how many fleets does lite_greedy generate per turn in the rollout?
  How many actually fire in the real game?).
- The fleet ship counts ME holds per planet over time. Does the
  lite_greedy variant garrison more than the no-launch variant?

`scripts/trace_pv_eta_scoring.py` is close to what's needed — would
need an extension to also log opp_actions_for_snap returns per
rollout. ~30 min to extend.

## Why this is parked, not chased

- The no-launch result was a *control* experiment to confirm the
  opp-axis is alive, not a probe of any specific lever.
- Even if the mechanism is exactly what I conjecture, the fix (the
  spatial-restricted lite_greedy variant) is the next probe regardless
  of trace evidence.
- n=1 anomaly out of 4 seeds; might be noise. n=8-16 would say
  whether seed 3493's pattern is reproducible.

## What might invalidate the conjecture

- If the trace shows lite_greedy's rollout fleets are *no more*
  threat-modeled than no-launch's (i.e. lite_greedy is silent on
  seed 3493 because all candidates fail affordability), then the
  diff is something else entirely — maybe the leaf evaluation
  drifted due to a different rollout-seed → game-state interaction.
- If the actual chooser emit counts are identical between the two
  variants on seed 3493, the diff comes from somewhere else in the
  rollout — possibly the opp-action timing affects when fast_sim's
  combat resolution fires.

## Decision

Parked. If spatial-restricted lite_greedy A/B comes back ambiguous,
this trace becomes the next-cheapest disambiguator. Otherwise it
sits.
