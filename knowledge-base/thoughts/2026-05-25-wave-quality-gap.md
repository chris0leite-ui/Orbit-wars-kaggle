# Generator-works ≠ generator-wins

baseline_wave v5.1 (commit `7fc52cf`) is the clearest example yet of
a candidate generator that PASSES every mechanical check and FAILS
the A/B against the gate opponent.

- v3.1 fired 0 waves across 200 turns on Aidan's seed.
- v5.1 fires 89 wave-emission turns on the same seed.
- v5.1 still loses to orbitfix at 4/16 (vs v3.1's 3/16).

The gap is *quality*, not *frequency*. Orbitfix's
`BASELINE_REINFORCE_ANTICIPATE` flag intercepts incoming waves with
defender reinforcement — our waves CONNECT, orbitfix REPLIES, and the
chooser-rollout's leaf value reflects the lost arms race. The
fast_sim rollout in `score_candidate_v4_joint` already knows this
(it sees the reinforced defense at arrival_step); the *proposer*
doesn't, so it keeps emitting candidates that get rejected by the
chooser anyway.

The correct next mechanism isn't "more waves." It's either:

1. **Opp-model-aware wave sizing.** Before emitting, simulate the
   reinforce response and only emit waves whose total exceeds
   `model.ships_at(tgt, arrival_step) + max_inflight_reinforce`.
   Existing `model.ships_at` already tracks inflight; needs a
   second pass that adds defender's REACTIVE reinforcement to
   incoming-wave events.

2. **Tempo-target shift.** Stop attacking orbitfix-class peers'
   strongest defended planets (where RA dominates) and start
   attacking their *production sources* (the rear stockpilers
   Aidan demonstrated). Same wave proposer, different
   target-pool prioritisation.

Approach (2) is closer to the Aidan replay: he targeted our
production engines, not our front line. Cheaper to test (no
proposer change, just a target-pool re-ranker) and exploits the
same combat-rule-1 advantage.

Side observation: Rule 38 (fix-verification reproduces failure
state) is necessary but not sufficient. For candidate-generator
fixes specifically, the verification needs a *quality* probe (per-
leg outcome trace on a game vs the gate opponent), not just a
*frequency* probe (count emissions). This is the kind of friction
that won't go away without an explicit rule. Friction tag logged:
`wave-mechanical-vs-quality-test-gap`.
