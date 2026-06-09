# Strategic bonuses regressed at clean A/B; multi_tick_recap shipped

**Date:** 2026-06-05. Session shipped sub 53390700 (multi_tick_recap)
and learned a sharp lesson about weight calibration on additive scorer
terms.

## What we learned about the strategic bundle

The denial + opening bonuses are MATHEMATICALLY SOUND but were shipped
with default weights that made them dominate rather than nudge. With
weight=0.1, prod=3, future_h=182 (game_length_est=200, current_step=0,
H=18), each bonus contributes ~55 ship-units per capture candidate.
Combined: ~110 ship-units. The existing `competitive_score` typically
falls in the 10-50 ship-unit range. The bonuses were 2-10× the actual
score they were supposed to nudge.

Concretely: the agent stopped picking moves on their actual merit and
started picking whichever candidate hit both the denial gate (opp
values target) and the opening gate (early game). Both bonuses fire
for *most* opening captures of neutral targets opp would have wanted —
that's almost every opening move. The agent over-raced for everything
the opp might want, in a phase where Producer's planner is already
correctly racing for neutrals via proximity. Result: 0/4 clean vs
producer.

## The fix is calibration, not redesign

The mechanism is the right intuition (per the PI's framing: most
games are decided in the opening + first encounter, blocking opp's
biggest bet is a winning move). The math is the right shape (ship
units, additive, gated by `_compute_captures()`, scales with
prod[T] * future_horizon).

What we got wrong was magnitude. The fix is to set the default weights
EMPIRICALLY:
1. Dump per-candidate `competitive_score` values from one full game
   (~100 candidates per turn × 50 turns = 5000 samples).
2. Find the median absolute score. Probably 10-30 ship-units.
3. Set new term default so its typical contribution is 5-15% of that.
   For median=20 ship-units and a typical bonus of weight × 3 × 180 =
   weight × 540: solve weight × 540 = 2 → weight ≈ 0.004. Order of
   magnitude smaller than what we shipped.

This isn't worth a framework rule promotion (PI declined) but it's a
sharp lesson for future scorer term integration. Add a "calibration
probe" to the scorer-term integration recipe.

## What we actually shipped: multi_tick_recap

Sub 53390700, PENDING in TrueSkill warm-up. Built on `multi_opp_def`
(μ=1285) base by adding:
- Multi-tick opp projection (K=3 in 4P, K=2 in 2P) — scorer sees opp's
  near-term actions across game-ticks 0..K-1.
- Recapture penalty — discounts captures opp can plausibly retake.

These two compose well by design: multi-tick widens the scorer's view
of opp's moves; recap penalty discounts captures opp can punish
*beyond* the multi-tick window (via `K_recap_eff = max(1, K_recap -
K_opp)`). Either mechanism alone fixes a different sub-defect.

Expected μ: somewhere between 1280 and 1320. Multi_opp_def base was
1285.0; the new mechanisms should add modest lift if they work in 2P
ladder games (the 4P cycle stalemate is a separate concern and the 4P
diagnostic ran 1 hour without finishing).

## What's locked for the next session

1. **First diagnostic next session:** clean single-process A/B for
   multi_tick_recap vs producer. Confirms whether the mechanism lifts
   over the base in clean conditions before the live μ settles. ~3
   minutes of work.

2. **Calibrate the strategic bonus weights empirically.** Run the
   probe described above; set defaults at ~0.005-0.02; re-A/B. If the
   re-tuned strategic bundle lifts cleanly, ship it as the follow-up.

3. **Multi-tick wallclock optimization.** Outlined this session, in
   order of ROI:
   - Skip regroup + greedy in opp planner calls (free 1.5-2×)
   - Batch opps within a round (3× combined)
   - Approximate rounds 1+ via eta-shifted reuse (further 2×)
   - Eventual learned opp policy (microseconds per call instead of
     tens of ms)
   Pre-requisite for any heavier mechanism stack.

4. **Force-concentration mechanism (the deeper lever).** Restructures
   `_greedy_select` to pool sources onto top targets instead of picking
   one launch per source. Higher altitude than another scorer term;
   bigger surgery. Was deferred this session in favor of the
   strategic bonuses; the strategic-bonus regression suggests it's
   time to consider this.

5. **4P self-match cycle diagnostic on multi_tick_recap.** We never
   got clean data this session. Run it next session under uncontended
   CPU to actually measure whether the recap mechanism breaks the
   cycle or merely shortens it.

## Calibration data point for the warm-up rule

Sub 53384340 (multi_opp_def) settled at μ=1285 by 2026-06-05 ~11:00 —
~28 hours after the 2026-06-05 06:58 submit. Climbed monotonically
through 947 → 1159 → 1287. Confirms Rule 12: read live μ after 24h+.
Earlier session-mid reads (947 at 4h, 1159 at 14h) were both
midway-warm-up reads that did NOT predict final settle.
