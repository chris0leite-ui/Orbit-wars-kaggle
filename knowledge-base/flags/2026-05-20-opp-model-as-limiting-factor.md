# 2026-05-20 — Flag: opp model is the limiting factor in chooser quality

**For careful PI review.**

The ledger failure and the sary-class anchor failure both trace to
the same root cause: **the chooser's value head scores plans
against a too-weak opp model (`lib.opp_model.lite_greedy_policy`).**

Evidence:
- Chooser scored ledger commits as +Δ → ledger emits the plan →
  real opp doesn't behave like lite_greedy → emit fails.
- lite_greedy doesn't coordinate multi-launch attacks on drained
  sources. Real opps do. Our chooser's plans assume they don't.
- The "+1 launch / turn" emission rate (sary) vs "0.8" (us) gap
  isn't a chooser-emit fix problem; it's that our chooser values
  hoarding because lite_greedy doesn't punish hoarders enough.

Implication: **the highest-leverage next move is opp-model upgrade,
not chooser-emit tweaks.** This is item (1) of the 5-mechanism
brainstorm in audit/2026-05-21-ledger-validation.md.

Estimated effort: 2-4 days for a learned-from-corpus opp model.
Could also be hand-coded heuristic that's closer to sary-class
behavior, in 1 day.

This flag is a candidate for PI's "next-axis" decision. Until the
opp model is upgraded, any chooser-emit, ledger, or value-head
change is fighting against a misleading internal scorer.
