# 2026-05-20 — when is bypassing the chooser leaf-score safe?

Phase 8 implemented the EpMVP Phase 6 mechanism on btjeK base:
chain candidates skip `score_candidate_v4` and use `cheap_delta`
directly. This worked on EpMVP (per the commit's own A/B notes
in 15e4838) but failed on btjeK (Phase 9 went to 1/16).

What's the difference?

**Hypothesis 1 — different chooser arithmetic.** EpMVP's chooser
returned a different favor scale than btjeK's. cheap_delta and
score_candidate_v4 may have been comparable on EpMVP but not on
btjeK. Test: print `(cheap_delta, score_candidate_v4 output)`
pairs for the SAME candidate set on both branches; compare scales.

**Hypothesis 2 — different opp model in the rollout.** btjeK's
`me_defensive_action` + `me_reactive_action` make the rollout
much more sophisticated than EpMVP's. cheap_delta is a closed-
form approximation that ignores opp counter — so on a richer
rollout, bypassing it discards more strategic depth than on a
weaker rollout.

**Hypothesis 3 — different mechanism set.** btjeK has filter
stages (cost-parity, hold-feasibility, drain-filter) that EpMVP
lacks. Chain candidates may sail past those filters because the
inflated cheap_delta doesn't reflect filter-relevant attributes.
Verify: drop a chain candidate's cheap_delta back to its base-
case value AFTER filters, and observe whether the chain
candidates still pass.

**Hypothesis 4 — different replay-driven population.** EpMVP's
chain-bonus A/B used a different opp pool. The opp baseline
matters: chain-bonus aggressive plays may suit one opp class and
not another.

**The general question is bigger than chain-bonus.** Whenever a
new mechanism's score sits in a different unit from the chooser's
leaf score, the chooser has two paths:
(a) bypass the leaf entirely (Phase 8 design)
(b) add the new score to the leaf as a delta
(c) calibrate the new score to the leaf's units first

(a) is fastest but discards depth. (b) requires the two units to
be roughly comparable. (c) is principled but requires offline
calibration runs.

For Orbit Wars specifically: are there mechanism additions where
the right design is (b) — folded into score_candidate_v4 itself
rather than running before/around it? Probably yes. The reactor-
candidate generator already does this (its candidates flow
through the same score_candidate_v4 path with no bypass).

**Action item for next session, if working on chooser-level
mechanisms:** answer Hypothesis 1 empirically before designing
the bypass.
