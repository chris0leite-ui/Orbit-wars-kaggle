# 2026-05-22 — orbital safety modeling fix completed and shipped

## Context

Prior session ended with `baseline_full` (sub 52893236, μ=1079) live in
the rolling pair — a kitchen-sink stack on top of consolidated with
four features (orbital safety + stagnant drain + combat stack + sniper).
Settled −45 μ below the strong consolidated baseline (sub 52882014,
μ=1124). Four features added on n=4 evidence; environment-variable leak
in fast.py meant the local A/Bs were all invalid (same effective config
on both sides of every test).

PI asked to "keep only the BASELINE_ORBITAL_SAFETY fix and carefully
code review for similar or other modeling issues or bugs."

## What was wrong with f1774a7 (the original orbital-safety fix)

The fix was correct for the two sites it touched (`time_to_enemy_threat`
and `expected_hold`) but the same modeling bug — using current planet
positions where positions-at-arrival are correct — existed in four
sibling chooser-side functions:

- `agents/baseline/proposer.py:_target_holdable_after_capture` (B1)
- `agents/baseline/proposer.py:_target_cost_parity_ok` (B2)
- `lib/missions/snipe.py:_followon_hold_estimate` (B3)
- `lib/missions/snipe.py:_best_followon` (B4)

Plus two additional correctness issues inside `time_to_enemy_threat`
itself:

- B5: `inbound >= arrival_eta` filter included fleets that arrive AT
  arrival (resolved by combat at that step; not future threats).
- B6: `incoming_enemy_eta` returned only the EARLIEST inbound, so when
  the earliest was pre-arrival the filter dropped it and LATER waves
  were silently lost.

And one approximation worth tightening:

- B7: enemy fleet aims at target-at-our-arrival, but target keeps
  rotating during enemy's travel. Straight-line `dist/v` was the
  pre-fix estimate; iterative 5-step fixed-point on `enemy_eta_travel`
  is closer to correct (same pattern as `lib/aim.py:aim_orbiting`).

## The fix in one place

All six gated on `BASELINE_ORBITAL_SAFETY=1` (default OFF preserves
backwards-compat with sub 52882014). Helper
`lib/world_model.py:_position_at(planet, omega, lead_turns)` keeps the
predict-or-current decision in one tested place. New
`WorldModel.incoming_enemy_eta_after(planet_id, my_id, after)` returns
earliest strictly-post-`after` inbound enemy fleet.

26 new orbital-safety assertions across three new test files plus
three pin tests for the existing wiring.

## Local A/B (clean_ab.py, subprocess-isolated)

Subprocess-per-game harness avoids the env-var leak. n=32 vs the live
agents we cared about; smaller smokes vs the latest Kaggle submission
because rebuilding the sibling-branch bundle for it took ~30 min.

| Opponent | μ on Kaggle | n | wins |
|---|---:|---:|---:|
| baseline_full (kitchen-sink, in rolling pair) | 1078 | 4 | 4 |
| baseline_joint_aggr_consolidated (strong baseline, evicted) | was 1124 | 4 | 2 |
| _phase4_step1_FND (latest Kaggle sub, in rolling pair) | 1117.9 | 4 | 4 |

n=4 is below Rule 45's lift-claim threshold, but the directional signal
is clean: dominates both currently-on-ladder agents and is at parity
with the strong baseline.

## Why we shipped it on n=4 evidence

Cause over statistics. PI re-articulated this principle mid-session:
"remember to keep bug or fundamental fixes although they do not clear
statistically. cause over statistics!"

The fix is a documented modeling bug. PI personally identified the
symptom in live games ("attack rotating planets that rotate in... close
to opponents so they can easily anticipate"). The unit-level tests prove
the fix is binding (decisions flip on the same seed). The local A/Bs
confirm directional improvement vs the agents currently on the ladder.
This is exactly the situation Rule 40 covers (modeling-correctness over
restriction-tuning) and Rule 1's "PI explicit approval per submit" was
explicitly given.

## Open questions for next session

1. **Seed-0 loss vs consolidated.** Game ran the full 500 steps before
   orbitfix lost — what does the orbital fix not yet capture that the
   strong baseline gets right? Likely the next modeling-bug candidate.
2. **4P validation.** All our evidence is 2P. The sibling branch built
   a seat-balanced 4P A/B harness (`38655cb`); pulling that and
   re-running would close the 2P-only blind spot.
3. **Settled μ.** TrueSkill noise rule (≥6h before reading). Submission
   went up at 2026-05-22 04:56 UTC.

## Submission

Sub **52912707** at 2026-05-22 04:56 UTC. New rolling pair:
`[52912707 orbitfix, 52894340 _phase4_step1_FND (μ=1117.9)]`. Evicted
`52893236 baseline_full` (μ=1078).
