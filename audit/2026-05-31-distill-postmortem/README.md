# Postmortem: distill-Tier-2 v2 (sub 53227546) — μ=748.4

**Submitted**: 2026-05-31 17:46 UTC (sub 53227546, commit `ce48fc0`).
**Evicted backstop**: 53212044 b3smoke μ=1147 → μ-LOSS confirmed.
**Settled μ**: **748.4** (9 episodes in).

## Headline

The hypothesis was: replace the falsified filter-on-Tier-1 opp model with a fast distilled-ladder predictor (30-d lite encoder, 6-tree LightGBM trained on 50,482 positives from 775 top-10 Kaggle 2P replays). This would give the chooser realistic leaf states and B.3-style lift would re-open.

It did not. ~400 μ regression vs the b3smoke backstop.

## What we got (9 episodes)

| Type | Count | Wins | Losses |
|---|---|---|---|
| 2P (1v1) | 4 | 2 | 2 |
| 4P FFA | 5 | 4 | 1 |
| **Total** | **9** | **6** | **3** |

Win rate 6/9 = 67% looks fine, but μ averages over OPPONENT skill and rank. 4P wins were vs beginner-tier opponents (Yashraj Prasad, Kevin Zhou, Juracan, wildfire, koheitada, etc); the 2P losses were vs moderate-skill opponents (João Felippe Thurler, Konstantinos Bekos).

## Diagnostic — two patterns in one

### Pattern A: 4P wins (passive)

Episodes 78325183, 78326109, 78326332, 78326619: **15–22 actions across 500 steps** (3–4% action rate). Last action often hundreds of steps before game end. We're getting an early lead and then idling while the three other players kill each other.

This matches PI's "good start then nothing then lose" observation, BUT we won these. The "lose" part of that pattern only fires when our 3 opponents *don't* dogpile each other and one consolidates.

### Pattern B: 2P losses (churn)

Episode 78324483 vs João Felippe Thurler (lost; reward −1):
- 86 actions / 95 emits over 500 steps (17% action rate — very active)
- Planet trajectory at 50-step snapshots: 8 → 8 → 8 → 11 → 10 → 7 → 7 → 7 → 7 → 8 → 12
- Pattern: capture planets (peak 11 at step 200), lose them back (7 by step 300), recapture (12 at step 500). Never consolidate.
- We have 12 planets at the buzzer but still lose the tiebreaker.

Episode 78324838 vs Konstantinos Bekos (lost; reward −1):
- 68 actions / 72 emits over 314 steps (eliminated early)
- 9 planets at step 50, 6 at step 200, **0 at step 400** — we got eliminated mid-game.

## Root cause hypothesis

**Train/inference distribution mismatch on opponent emit rate.**

The distilled booster was trained on TOP-10 ladder players' emit patterns (~5 emits/turn, very disciplined, capture rate ~70%). Real Kaggle opponents at our μ range (900–1100) are moderate-skill and have different emit patterns — likely more aggressive sniping, less disciplined.

In the chooser's fast-sim rollout, the distilled opp predicts FEW counter-launches (because top-10 players are selective). Our chooser then sends ships forward thinking they'll arrive safe. Real opponents DO counter, and the captured planet flips back. Churn-and-lose.

This is exactly the symptom of Rule 26's PI-interaction protocol Q1: "what precedent are we pricing this against?" We priced against top-10 player behavior; we're playing against μ-1000 opponents.

## Other contributing factors (smaller)

1. **No B.3 head**: VH_LAMBDA=0 in this bundle to attribute the opp-model change cleanly. The composite (Phase 6c scaffolding committed earlier) would layer B.3 head on top. Without it, all the Δ signal goes through the opp model.
2. **4P games**: booster was trained on 2P data only. In 4P games, in_flight_enemy counts can be 3× higher than training distribution. Some feature values are out-of-distribution.
3. **KINEMATIC_TABLE_ENABLED hard-disabled**: confirmed in the bundle. Not a regression source.

## Local-evidence gap

Pre-submit local A/B vs `launch_rules_universal` at n=32 timed out at the 60-min wallclock (~7 min/game serial). Only n=1 single-game wins as evidence (vs v7_0 and vs launch_rules in earlier session; vs v7_0 loss this session). At n=1, win rate is uninformative.

If we had completed n=32, we likely would have seen a regression — Rule 45 was designed exactly to catch this kind of false-positive lift on small-n evidence.

## What's locked / what's not

- **Position 1**: ours sub 53227546 μ=748.4 (this regression)
- **Position 2 backstop**: sub 53223160 joint_sync μ=1036.1
- **Daily submission budget**: 1/5 used today
- **Rule 12 caveat**: pushing ANOTHER submission today evicts joint_sync 1036 → μ=748 becomes the floor for the next 24h

## Recommendations

1. **HOLD** today. Don't push another submission — joint_sync 1036 is the protection we have until μ=748 ages off.
2. **Tomorrow**: pick one of:
   - (a) Re-bundle with VH_LAMBDA=1.0 (the composite Phase 6c wrapper is scaffolded and ready) — tests if B.3 head salvages the opp-model lift
   - (b) Revert to a known-good ancestor (b3smoke or pv_eta_vh) — restore floor first, iterate after
   - (c) Re-train booster on a broader population (not just top-10) — closes the train/inference distribution gap, but ~50 min decode + 15 min train
3. **Bind the learning**: add Rule 48 (proposed) — "Don't ship a distilled imitation model without a calibration A/B vs the actual opponent population. Local n≥4 minimum BEFORE Kaggle for any new opp-model architecture."

## Files

- `episode-*.json` and `episode-*-agent-*-logs.json` in this dir — pulled from `kaggle competitions replay/logs` (~125 MB total, gitignored).
- This README is the trackable summary.
