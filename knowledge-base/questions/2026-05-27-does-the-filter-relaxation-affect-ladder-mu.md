# 2026-05-27 — does the B1 filter relaxation affect ladder μ?

## Question

If we cherry-pick session-EqJuT commit `68c24be` (relax B1 hold
filter in dominant endgame, gated on `my_count > 3 * opp_count`)
onto `agents/baseline/proposer.py:_target_holdable_after_capture`,
re-bundle, and resubmit — does the ladder μ measurably differ
from a resubmit of the unmodified bundle?

## Prior

The B1 filter is load-bearing for +63μ on the ladder in midgame.
The relaxation fires ONLY in dominant endgame (`my > 3 * opp`).
Competitive games virtually never reach that state — they end
via midgame elimination or score-tiebreak.

Expected answer: **no measurable difference** in ladder μ. The
relaxation is invisible to the ladder distribution because the
ladder doesn't probe its activation condition. The orbitfix
bundle settled at μ=1165.4 WITH the bug; should re-settle at
~the same μ WITHOUT.

But this is a prediction, not a measurement. The dominance gate
might fire more often than expected if e.g. on small starting-
seat geometries we run away early. The candidate Rule 48 gate
is built precisely to catch that class of correctness bug, so
running it post-cherry-pick would tell us whether the fix
*activates* — not whether the activation costs ladder μ.

## How to answer

The cheapest probe is **submit both back-to-back and compare**:

1. Resubmit unmodified orbitfix (call settled μ = μ_A).
2. Cherry-pick + re-bundle + resubmit (call settled μ = μ_B).
3. |μ_B - μ_A| < 20 → relaxation is ladder-neutral (expected).
4. μ_B significantly below μ_A → the dominance gate fires more
   than expected on the ladder; the relaxation is mildly
   destabilising; deeper investigation needed.

This is two submission slots and ~48h of ladder convergence
per slot. Worth doing once before we promote the fix into the
shared production bundle.

## Why this matters

If the answer is YES, we have a tension: gate-passing requires
a fix that costs ladder μ. The Rule 48 gate would then become
a strength-vs-correctness tradeoff. Currently Rule 48 is
promoted as a hard pre-submit gate; finding ladder-μ cost would
require re-rating Rule 48 from "mandatory" to "advisory".

If the answer is NO (expected), Rule 48 stays cheap to enforce.