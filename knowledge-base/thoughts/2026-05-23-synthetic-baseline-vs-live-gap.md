# 2026-05-23 — items 1+3+4+5 saturated below the live leader

> Branch: `claude/strategy-axis-decision-3437` (session-EqJuT continued).
> Plan: `/root/.claude/plans/composed-noodling-riddle.md`.

## Load-bearing finding

**Synthetic-baseline A/Bs do not predict live-leader A/Bs.** The
α+β stacked variant scored 5/8 = 62.5% vs `alpha_beta_off`
(no-features) and 9/16 = 56.2% at confirmation, looking like a
+6–12pp directional lift. The SAME bundle scored 3/8 = 37.5% vs
the actual live ladder leader `orbitfix` (μ=1165) AND vs the
prior live agent `_phase4_step1_FND` (μ=1101). All three
measurements at n=8 with Wilson half-width ≈ ±25pp; mutually
consistent with parity ± noise, but **the point estimates are
BELOW 50% vs the live agents and ABOVE 50% vs the synthetic
baseline**.

Mechanism: `alpha_beta_off` is a literal-derivative of
`alpha_beta_on` (only the gate functions differ, rest is byte-
identical). The two production bundles (FND, orbitfix) are
DIFFERENT code lineages — different chooser, different proposer
tuning, different env-var defaults. Synthetic vs. synthetic
measures "the marginal contribution of α+β features holding
everything else constant"; live A/B measures "is α+β better than
the agents that are actually deployed." They're different
questions.

The new orbitfix submission at μ=1165 (a +63μ jump from
`baseline_joint_aggr_consolidated` at μ=1102) came from physics-
modeling improvements (B1-B7 orbital-arrival safety sweep), NOT
from LP value-function refinements. That's where the ladder's
marginal returns currently are.

## Implications

1. **Don't ship α+β stacked.** At best parity, possibly weaker,
   vs live leaders. Rule 12 (rolling-last-2 risk): would evict a
   known-good submission for an unverified variant.
2. **The LP-family value-function axis is saturated** at the current
   substrate. Three sub-axes (smooth ΔW / topology / maximin search)
   tested independently and stacked. None lifts past the noise vs
   live agents.
3. **The most reliable A/B template going forward** is focal-vs-live
   from the start, NOT focal-vs-derived-baseline. The latter
   answers "does my code do anything?" not "does my code beat
   the deployed agent?"
4. **Architecture investment from this session is permanent and
   reusable**: clean_ab_4p subprocess harness (Rule 46 candidate
   for 4P), Lagrangian dual_decomp module (γ phase, parity-tested),
   smooth ΔW value function (Phase α, lazy-gated), diversity
   constraint in portfolio enumeration. These don't need to ship
   to a submission to be useful — they unblock future iterations.

## Off-ramp priorities (highest EV first)

A. **Physics modeling sweep** (orbitfix-style). orbitfix's +63μ
   came from auditing primitives against entity-type behaviour.
   Candidates: 4P-aware orbital safety (currently 2P-gated),
   comet-rotation safety in `predict_fleet_fate`, opp-fleet-
   collision modelling. Rule 47 — primitive entity-type audit.

B. **Konbu17-style ML shot validator**. Only ML approach with
   empirical precedent (+19pp panel lift). ~1 week build but
   well-bounded; conservative (only rejects shots, never proposes).

C. **Stop the LP track**, treat this session as a research
   investment, and roll the substrate forward to whichever track
   (A or B) the PI picks.

D. Continue tightening LP-family work knowing the substrate is
   built but the headroom is small. Per-turn wallclock benchmark
   of dual_decomp + maximin-with-fast_sim-leaf are the open
   wallclock-affordances questions.

## What's NOT useful to do

- **n=32 of α+β vs orbitfix**: at point estimate 37.5%, Wilson
  even at n=64 won't reach the 0.55 gate. The signal isn't there.
- **More λ_W sweep**: sweep already done {0.1, 0.3, 1.0, 3.0};
  optimum is the conservative end and the response surface is
  flat below 1.0.
- **More closed-form-leaf maximin variations**: closed-form leaf
  uses LP math; maximin over LP-evaluated portfolios degenerates
  to LP argmax. Need a different leaf (fast_sim or stronger opp
  policy) before maximin can extract signal.

## Pointer for next session

`audit/2026-05-23/items-1-3-4-5-execution.md` has the full A/B
table + per-phase code references. Re-pull live μ first
(Rule 43), then pick A/B/C/D off-ramp.
