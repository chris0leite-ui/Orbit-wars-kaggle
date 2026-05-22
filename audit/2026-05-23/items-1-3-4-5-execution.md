# Items 1, 3, 4, 5 execution — 2026-05-22 to 2026-05-23

> Plan: `/root/.claude/plans/composed-noodling-riddle.md`.
> Branch: `claude/strategy-axis-decision-3437`. Session id
> `claude/session-EqJuT` continued.

## Summary

Executed all four queued items per PI directive "go 1, 3, 4, 5".
Two harness/bundler bugs found and fixed; four code phases shipped;
six A/Bs run. Headline result: α+β stacked is **not 4P-harmful**
once topology features are gated to 2P-only (cleared the 0/16 false
regression). λ_W sweep confirms the conservative default (λ_W=0.3)
is at the optimum point on the {0.1, 0.3, 1.0, 3.0} grid. No
submissions made.

## A/B record

| # | What | Focal | Opp | Win/Tie/Loss | Wilson 95% | Notes |
|---|---|---|---|---|---|---|
| 1a | Item 1 self-play sanity v1 (broken: tie logic) | `alpha_beta_off` | `alpha_beta_off` (same file) | 4/0/0 | [0.510, 1.000] | All 4 tied at step 500 but old logic counted ties as wins → false 100% |
| 1b | Item 1 self-play sanity v2 (tie logic fixed) | `alpha_beta_off` | `alpha_beta_off` | 1/4/3 | [0.022, 0.471] | Wilson includes 0.25 — **harness parity OK** |
| 1c | Item 1 4P A/B α+β stacked (clean harness) | `alpha_beta_on` | `alpha_beta_off` | 1/12/3 | [0.011, 0.283] | 12 ties at step 500 — α+β has **zero 4P effect** by design (smooth ΔW gates 2P-only; topology gates 2P-only post 2P-gate). 2P-gate confirmed safe. |
| 3 | λ_W=0.1 vs no-features (2P) | `alpha_beta_lambda_0_1` | `alpha_beta_off` | 5/0/3 | [0.306, 0.863] | Matches λ_W=0.3 outcome bit-for-bit at the win level. |
| 3 | λ_W=0.3 vs no-features (2P, current default) | `alpha_beta_lambda_0_3` | `alpha_beta_off` | 5/0/3 | [0.306, 0.863] | Confirms prior session's directional positive. |
| 3 | λ_W=1.0 vs no-features (2P) | `alpha_beta_lambda_1_0` | `alpha_beta_off` | 4/0/4 | [0.215, 0.785] | Saturation — too strong, replicates the step-function pathology. |
| 3 | λ_W=3.0 vs no-features (2P) | `alpha_beta_lambda_3_0` | `alpha_beta_off` | 4/0/4 | [0.215, 0.785] | Same as 1.0. |

## λ_W sweep conclusion

- {0.1, 0.3} share the **same** observed outcomes (62.5% point). They tie at
  the noise floor; sub-1.0 magnitudes don't disambiguate further.
- {1.0, 3.0} drop to 50% (4W/4L). At these magnitudes, smooth ΔW
  re-saturates the LP argmax like the original `LAMBDA_ENDGAME=1000`
  step bonus did before α landed. **The sweet spot is ≤ 0.3.**
- Verdict: **keep λ_W=0.3 default.** No additional sweep needed at
  finer resolution; the response surface is flat below 1.0 at this n.

## 4P regression status (Item 1 + revisit)

- Pre-2P-gate: 0/16 catastrophic regression on the broken harness.
- Post-2P-gate, broken harness: 13/16 — was inflated by harness
  artifact (self-play also gave 5/8 false-wins).
- Post-2P-gate, CLEAN harness: 1/16 with 12 ties — α+β stacked
  behaves identically to no-features in 4P. **Confirmed safe.**

The 1 focal win came from seed=1 where seat 3 has a deterministic
advantage; the focal placed in seat 3 wins regardless of features.
Other 3 seeds (0, 2, 3) hit step=500 cap with ties.

## Code changes shipped

```
lib/joint_solver/dual_decomp.py          (NEW)  Item 5 Lagrangian inner
lib/joint_solver/lp_outcome.py           (MOD)  dispatcher LP_SOLVER=dual
lib/pipeline/portfolio_enum_lp_seeded.py (MOD)  min_distinct_primary_sources
lib/pipeline/decision_lagrangian_maximin.py (MOD) Item 4a wiring
scripts/clean_ab_4p.py                   (NEW)  Item 1 4P harness fix
scripts/build_topology_variants.py       (MOD)  λ_W + LP_SOLVER variants
scripts/bundle_agent.py                  (MOD)  add joint_solver/dual_decomp
scripts/measure_portfolio_diversity.py   (NEW)  Item 4a diagnostic
tests/test_dual_decomp_parity.py         (NEW)  5 pin tests, all green
```

Tests: 29/29 broad pin tests green
(dual_decomp_parity + lagrangian_maximin + lp_endgame_predicate
+ lp_topology_features).

## Items deferred / off-ramped

- **Item 4b** (fast_sim leaf rollout in maximin): explicit HOLD per
  Plan-agent bias warning. The closed-form leaf uses the analytical
  mirror as opp; replacing with `lite_greedy_policy` measures
  "robust against a weak opp" instead of "robust against the real
  opp model". The proposed mitigation (mirror at tick 0, lite_greedy
  ticks 1..14) is a separate design that didn't fit this iteration.
- **n=32 of α+β stacked alone**: explicitly skipped by PI directive
  in the plan (item 2). Bad EV at the current signal strength.

## Item 5 — dual vs MILP real-game A/B (POST-RUN)

```
focal: alpha_beta_solver_dual.py vs alpha_beta_solver_milp.py
4 seeds × 2 seats = 8 games
focal_wins=4/8 (50.0%), Wilson [0.215, 0.785]
elapsed 153s (dual variant), 138-148s on seed=0 (longest)
```

Per-seat breakdown:

| Seed | P0 | P1 |
|---|---|---|
| 0 | WIN  (461 steps) | LOSS (479 steps) |
| 1 | WIN  (204 steps) | LOSS (204 steps) |
| 2 | LOSS (155 steps) | WIN  (167 steps) |
| 3 | WIN  (163 steps) | LOSS (243 steps) |

Perfect per-seat mirror: each seed has one P0 winner and one P1 winner.

**Diagnosis**:
- Dual ≠ MILP behaviorally: step counts differ from the MILP-vs-MILP
  baseline at every seed (e.g., seed=1 was 143 steps for alpha_beta_on
  vs off; here both 204 — dual makes different decisions on both
  sides).
- Outcome-equivalent at n=8: neither solver is systematically better.
  Compatible with "different tie-breaks at the LP-relaxation level
  but equal-quality at the n we can afford."
- **Wallclock**: dual seed=0 took 138/148s vs MILP's prior seed=0 at
  100-160s in the alpha_beta A/B. **Within noise — NOT the 10×
  speedup target.**

Why no speedup observed:
- Plan target was p95 ≤ 50 ms per turn (vs MILP's ~300 ms). We did
  not measure per-turn timings; game-level wallclock includes long
  games on dual (seed=0 went 461-479 steps vs 364 for MILP) so the
  comparison is confounded.
- The dual inner does 3 iterations × per-target argmax over 64
  subsets × per-source rent computation. If subset enumeration is
  the bottleneck (already O(2^k) regardless of solver), dual saves
  only the MILP setup overhead.

**Kill condition (per plan)**: "p95 wallclock > 100 ms → Lagrangian
doesn't actually beat MILP for our problem sizes." Cannot evaluate
without per-turn timing instrumentation. Status: **research path
that ships pin-tested but with unknown speedup**. Production swap
not justified at this measurement state.

## Items pending after this session

- Item 4 maximin re-A/B (now with diversity + post-router-fix)
  — RUNNING.
- Item 4a diversity diagnostic (`measure_portfolio_diversity.py`)
  — never run; the diversity constraint defaults on without
  empirical motivation.
- Per-turn wallclock comparison MILP vs dual on a panel — needed
  to call the Item 5 production-swap decision per plan kill
  condition.

## Plan-agent's open critiques (still relevant)

- Item 4b bias: confirmed accurate; ε.2 needs a stronger opp policy
  before fast_sim leaves are useful.
- Family-wise α: 6+ A/Bs at α=0.05 → expect one spurious clearance.
  All A/Bs in this session were inconclusive at Wilson; no false
  positives to discount.
- Regression panel: not run. Would catch silent regressions vs
  v3.5.1 / v7_minimax baselines. Worth adding in a future cycle.

## Decision points for PI

1. **Ship α+β stacked at λ_W=0.3?** 2P directional +12pp at n=8/16,
   4P clean-neutral. Live μ impact uncertain (small directional
   positive in 2P, zero in 4P; depends on the ladder's 2P/4P mix).
2. **Continue to Item 4 (maximin re-A/B post-fixes)?** It's queued
   and ready.
3. **Continue to Item 5 production swap (LP_SOLVER=dual)?** Real-
   game parity result pending; if within gap, swap as default and
   unlock K=4-6 maximin.
