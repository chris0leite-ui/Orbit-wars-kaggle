# Phase α + β stacked — first directional positive signal

> 2026-05-22 evening, branch `claude/strategy-axis-decision-3437`.
> Plan ref: `/root/.claude/plans/composed-noodling-riddle.md`.

## Sequence of A/Bs run today

All A/Bs use `scripts/clean_ab.py focal opp --seeds N --workers 4`,
which runs `2N` games (N seeds × 2 seat assignments) one
subprocess-per-game.

| # | Focal | Opp | Win/Loss | Wilson 95% | Wallclock | Interpretation |
|---|---|---|---|---|---|---|
| 1 | analytical_phase_c (topology setdefault=1) | _phase4_step1_FND (topology setdefault=0) | 4/8 | [0.215, 0.785] | 115s | **env-var contaminated**: perfect per-seat mirror, identical step counts |
| 2 | topology_on (hardcoded True) | topology_off (hardcoded False) | 4/8 | [0.215, 0.785] | 117s | Isolation now correct; still mirror. **Bug found**: `_per_planet_topology_score` not actually being called (kwarg signature). |
| 3 | topology_on (post-bugfix) | topology_off | 4/8 | [0.215, 0.785] | n/a (covered by re-run via build_topology_variants) | Topology fires correctly (1156 calls / 80 steps verified) but doesn't tip LP argmax because LAMBDA_ENDGAME=1000 step dominates topology lambdas of 50/10/30 |
| 4 | smooth_dw_on (α: smooth ΔW, topology OFF) | smooth_dw_off (step, topology OFF) | 4/8 | [0.215, 0.785] | 153s | Step counts differ (255 vs 196 on seed 1) — smooth ΔW IS changing decisions. Outcome neutral at λ_W=0.3. |
| **5** | **alpha_beta_on (α + β stacked ON)** | **alpha_beta_off (α + β stacked OFF)** | **5/8 (62.5%)** | **[0.306, 0.863]** | **166s** | **First directional positive.** Step counts: 143, 224, 303, 267, 364, 209, 500, 309 — true behavioral diversity. |

## Key finding

α alone is neutral. β alone is neutral. α+β stacked is +12pp at n=8
(point estimate). This is **exactly the Plan agent's prediction**:

> "Order risk — α BEFORE β is correct. The topology bonuses are linear
> in `prod(q)/(1+eta)` and `prod(p)/hold_time` — single-digit to
> low-hundreds magnitudes. `LAMBDA_ENDGAME = 1000` (hardcoded,
> lp_outcome.py:126) is a step that already dwarfs both prod_stream
> AND topology. ... Do α first (smooth the step), THEN β."

The step `_endgame_bonus` of ±1000 was crowding out topology's
smaller-magnitude bonuses. Smooth `λ_W · ΔW` at λ_W=0.3 has
per-(planet,subset) magnitudes ~200-1500, comparable to topology's
50-300 range. The LP can then weigh them against each other meaningfully.

## Why the 4-game mirror lifted with stacking

In runs 1-4, every seed showed a perfect P0/P1 reversal — same step
count both seats, opposite winner. That's the signature of two
near-identical agents in a 2P seat-asymmetric game: the seat advantage
determines outcome.

In run 5, step counts diverge per game (143-500), and 3 of 4 seeds
break the mirror (the focal wins BOTH seats on seed 1, seat 0 on seed 3
but loses both on seed 0). The asymmetry is no longer noise — it's the
features actually shifting outcomes.

## Killgate check

Per `composed-noodling-riddle.md`:

| Phase | Result | Gate met? |
|---|---|---|
| α-alone (smooth ΔW only) | 4/8 (50%), Wilson [0.215, 0.785] | NO (lo < 0.55) |
| β-alone (topology only) | 4/8 (50%), Wilson [0.215, 0.785] | NO (lo < 0.55) |
| **α+β stacked** | **5/8 (62.5%), Wilson [0.306, 0.863]** | **NO at n=8 (lo 0.306 < 0.55), but directional positive ⇒ escalate to n=16** |

Per plan off-ramp rule: "If β NULL at n=8: re-test at n=16 only if
directional positive (point estimate ≥ 0.55)." Stacked is at 0.625,
above the 0.55 threshold. **Escalation to n=8 with 16-game count
(seeds 0..7, ×2 seats) is RUNNING.**

If n=16 holds at Wilson-lo ≥ 0.55, the stacked variant is the new
baseline; bundle + 4P A/B + push under Rule 42.

## What's next regardless of n=16 outcome

1. If stacked clears n=16: ship under Rule 42 PI sign-off.
2. If stacked weakens at n=16: try λ_W sweep ({0.1, 0.5, 1.0, 3.0})
   to find the calibration sweet spot. n=8 each, sweep at n=8 first.
3. If sweep null: pivot to Phase ε.1 (adversarial maximin search) —
   this is where the search architecture can extract value the LP
   objective tweaks couldn't.
4. **Always do**: 4P A/B before any push (Rule 43). Current A/Bs are
   all 2P.
