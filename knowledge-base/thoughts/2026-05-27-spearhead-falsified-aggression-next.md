# 2026-05-27 — Spearhead falsified; aggression-recalibration is the next principled move

## What happened today

Designed and shipped (locally, not to Kaggle) a "spearhead" directional
rule across two layers — relay R-selection bonus + chooser-side
candidate-direction bonus. Hypothesis: fleets ping-pong because the
relay picks any nearest friendly without regard to which way is "front."
Implementation gated behind two env vars (`BASELINE_RELAY_SPEARHEAD`,
`BASELINE_DIRECTIONAL_BONUS`); defaults preserved.

5-cell A/B matrix vs `baseline_joint_aggr_consolidated_orbitfix` (the
sibling-branch live champion line):

- Cell A (reference, relay off, no spearhead): 4/5 = 80% — strongest
- Cell B (relay on, no spearhead): 2/5 = 40% — confirmed relay regression
- Cell C (relay on + relay-spearhead): 3/5 = 60% — partial heal only
- Cell D (relay off + chooser-bonus): 2/5 = 40% — chooser bonus REGRESSED
- Cell E (both): 1/5 = 20% — worst; the two passes fight

The chooser-bonus regression was the surprise. Plan agent had recommended
β=8 sized off PI's "delta ≈ 280" reference; math checked out at design
time. Why it actually regressed: the favor leaf already encodes
directional value implicitly. The trajectory rollout literally simulates
forward; front-line captures hold longer in the rollout because they
produce more before being recaptured, which raises the favor delta
without any explicit directional term. Adding an explicit cosine bonus
on top double-counted and distorted target selection toward
distance-from-self-toward-opp without re-checking force-sufficiency.

## What I learned

The rule (which PI declined to promote yet, fairly — first occurrence):
**rollout-based leaf evaluators implicitly encode positional, temporal,
and directional structure through the trajectory they simulate. Don't
add explicit X-aware bonuses without first checking whether the leaf
already rewards X.**

Sister to Rule 40 (modeling-correctness over restriction-tuning). The
chooser-bonus was a *restriction* (move selection bias) layered on a
*model* (favor delta) that was already correctly modeling the dimension
the restriction was meant to enforce. The model was right; the bonus
was the band-aid.

This is the FOURTH session in a row where the chooser-layer axis has
been falsified for this opponent class:

- Proposer pre-filter tightening (2026-05-25) — closed
- Defensive modeling (2026-05-26) — closed
- Spearhead today — closed
- (Counting v9-v15 chooser saturation earlier as a fifth)

The pattern is unmistakable. The right next moves are NOT another
chooser knob.

## Open questions

1. **Is the favor leaf actually a good directional encoder?** It would
   be useful to write a synthetic-board test that confirms the leaf's
   direction-sensitivity. If it's NOT, then the chooser-bonus could
   still be the right idea with a different formula — but until we
   measure, "trust the leaf" is the conservative default.

2. **Did Cell C's +20 pp lift over Cell B come from R-direction, or
   from the bonus randomly suppressing some of the worst relay picks
   without replacing them with good ones?** The relay's underlying
   premise (idle planets to friendly waypoints) might be the rotten
   core; the directional bias just makes its bad decisions less
   wrong.

3. **Top-10 fingerprint says garrison-at-launch ≈ 7.7 vs ours ≈ 25.**
   Our chooser holds zero reserve (`MIN_SOURCE_RESERVE=0`), so the
   bottleneck isn't chooser-defensiveness. It's the
   `_target_holdable_after_capture` check that rejects captures we'd
   net-win because the post-capture 1.5× safety margin is over-
   pessimistic. Single-knob A/B candidate for next session.

## What I'd do next (if it were my call)

Aggression recalibration A/B as a single-knob sweep on `SAFETY_MARGIN`:
1.5 (current), 1.2, 1.0, 0.8. Four cells × 15 min ≈ 1 hour. If 1.2 or
1.0 lifts, ship the env-gated default. If everything below 1.5 is the
same as 1.5, the holdability axis is not the bottleneck and we move on
to the batched-chooser substrate (`lib/game/batch_interpreter.py` is
already built and parity-tested, just unwired) or the replay-mined opp
model.

PI's framing today: "we want to get on top, simple moves matter, no
rush on recovery." Aggression recalibration is the simplest principled
move with direct EDA evidence behind it.

## Pointers

- Audit: `audit/2026-05-27-spearhead-ab-matrix.md`
- Postmortem: `audit/2026-05-27-postmortem-agent-design-exploration-Q0q9T.md`
- Implementation: commit `85871cd`
- Plan (executed): `/root/.claude/plans/do-it-but-do-bubbly-teacup.md`
- Top-performer EDA: `knowledge-base/concepts/top-performer-strategies.md`
- Batch interpreter (unwired): `lib/game/batch_interpreter.py`
