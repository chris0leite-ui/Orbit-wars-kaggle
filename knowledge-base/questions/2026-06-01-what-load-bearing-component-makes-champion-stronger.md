# Question — what load-bearing component makes the champion stronger?

**Asked:** 2026-06-01
**Asker:** Claude (Opus 4.7), end-of-session reflection
**Trigger:** N=4+ falsified attempts to make jsr-line beat champion

## The question

`baseline_launch_rules_universal_local` (live μ=1183.7) beats every
jsr-line variant 16/16 locally. What is it doing that jsr can't reach
by chooser/value-head/aggression-handoff modifications?

## What we know

- Champion uses the universal K=10 launch-discipline validator
  (`BASELINE_LAUNCH_RULES=1` + `BASELINE_CAPTURE_HORIZON_K=10`).
- jsr also uses these — same env vars, same launch_rules code path
  at the END of every chooser branch.
- The DIFFERENCE between champion and jsr is the chooser stack
  upstream of launch_rules: jsr uses composite (distilled-Tier-2
  opp + B.3 head + slot_reservation + joint_sync + size_balance +
  surplus_aggression + persistent_attack); champion may use
  something simpler (the bundle name suggests just launch_rules
  validating output from an unspecified upstream).
- Adding ROI handoff to jsr: 7/16 vs jsr (+12pp), 0/16 vs champion.
- Adding v7_add_one handoff to jsr: 11/16 vs jsr (+22pp), 0/16 vs
  champion.

## Hypotheses (cheapest to test first)

1. **Champion's upstream chooser is simpler and more correct.** The
   value-head + opp-model + slot-reservation stack in jsr may be
   adding noise that downstream launch_rules can't filter. If
   champion uses a closed-form trajectory chooser without leaf-value
   evaluation, that would be simpler-and-better.
   - Test: cat champion's bundle, identify its chooser branch.
     Find what BASELINE_CHOOSER value (if any) it sets.

2. **Champion exploits an opp-archetype that jsr-line is bad at.**
   The 16/16 sweep is uncharacteristically clean — even random
   variation should yield 1-2 wins by chance. If the matchup has
   a structural counter, champion beats every seed of jsr-line.
   - Test: multi-opp panel for addone-v5 — does it lose to ALL
     opps or just champion? If just champion, it's matchup-specific.

3. **Champion was tuned against specific opponents that jsr-line
   resembles.** Champion's μ=1183.7 may reflect being TUNED for
   the local-ladder distribution, including beating jsr-line
   specifically.
   - Test: champion vs champion is presumably 50%. Verify.

## Why this matters

The 0/16 vs champion result is the SUBMISSION GATE per Rule 43b. As
long as jsr-line can't clear this, we can't submit any jsr-derived
agent — would lose rolling-pair μ on every push. Answering this
question unlocks the rest of the architectural search.

## Tags

`jsr-line-cannot-beat-champion-axis`
