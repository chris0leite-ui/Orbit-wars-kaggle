# Per-launch denominators are unsafe when decision-class frequency varies

**Date:** 2026-05-21
**Session:** `claude/audit-workflow-performance-btjeK`
**Context:** v1 large→small audit reported per-launch NET as the headline.
v2 confound-controlled re-audit rejected the leak. PI caught both confounds
before any agent change shipped.

## The pattern

When two decision-classes A and B differ in how often the actor takes
each — for whatever reason intrinsic to the actor's situation, not the
quality of the decision — `metric_per_decision_A` vs
`metric_per_decision_B` reads the difference in **frequency**, not the
difference in **quality per unit resource**.

In this case: large planets (prod≥4) produce more ships, so they
*launch more launches AND launch more ships per launch*. Per-launch NET
production (gain − loss) is computed over a denominator that's biased
toward the frequent-launcher: a large source making 10 launches that
each gain +1 production averages +1 NET/launch, but a small source
making 1 launch that gains +0.5 production averages +0.5 NET/launch.
The small source isn't worse — it's launching less.

The right denominator depends on what's being compared:
- **Comparing decision quality**: normalise by the resource spent, not
  the decision count. Per-ship NET answers "for every ship I commit,
  how much production do I get back?"
- **Comparing total contribution**: don't normalise at all; report sums.
- **Comparing strategic preference**: a counterfactual ("what if I'd
  chosen B instead of A given the same opportunity?") — but this
  requires matched samples, which replay-mining doesn't give.

## End-state attribution is the other half

v1 also debited "src lost by end-of-game" to the last launch from that
src. In a lost episode every planet flips by definition, so every src
gets a loss debit, and the last launch always eats it. In a won episode,
no src flips, so no launch ever pays. The "per-launch NET" then becomes
a thinly-disguised episode-outcome proxy.

The fix: attribute loss inside a SHORT window after each launch, ending
at the next launch from that src OR `landing_step + 20`, whichever
comes first. Each launch is attributed independently of episode outcome.
v2's `src_lost_within_20_pre_relaunch` does this; the leak signal
collapsed.

## Candidate rule for promotion

**Rule 41 (proposed):** When comparing decision-classes that differ in
their natural frequency, the primary metric must normalise by the
underlying resource (ships deployed, time spent, ops budget), not by the
count of decisions. Per-decision metrics are kept as legacy comparisons
but never lead an audit.

PI ratification pending. See
`audit/2026-05-21-large-to-small-confound-controlled.md` for the data.

## Counter-signals to keep in mind

The asymmetry that survived confound-control (small→large +0.029 vs
large→small +0.007 per-ship NET) is small but real. Three explanations
that are NOT "the chooser is leaking":

1. **Geometry**: large planets are central, small planets peripheral.
   Attacks INTO large planets pay more by definition (more production
   captured). Attacks FROM large planets go where opportunity exists,
   which is often the periphery.
2. **Opportunity set**: large planets have surplus ships and attack
   whatever is reachable. Small planets are picky launchers (they need
   to keep ships for defense), so when they do launch they pick the
   highest-value target available.
3. **Marginal-value of the next ship**: the 100th ship on a large planet
   has near-zero defensive marginal value (it's already safe). Spending
   it on a small target with even +0.5 expected production is rational.

If we wanted to test "is the chooser making bad decisions" properly we'd
need a counterfactual: replay the same observation, ask the chooser
"what's the next-best alternative to this launch?", and check whether
the alternative had a higher expected NET. That's the future audit if a
leak signal returns after confound control.
