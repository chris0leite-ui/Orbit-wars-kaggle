# 2026-05-22 evening session — composed-noodling-riddle execution

> Branch: `claude/strategy-axis-decision-3437`. Session ID
> `claude/session-EqJuT`. Plan ref:
> `/root/.claude/plans/composed-noodling-riddle.md`.
>
> Sister sessions ran concurrently and contributed parallel commits;
> we rebased twice during execution. Their commits delivered:
> - clean_ab.py harness + Rules 44-47 promoted (`7f0b607`).
> - Bundle multi-line import + alias-rebind fixes (`ef613eb`).
> - W(s) calibration on 20 replays from sub 52894340 (`96ca45f`).
> - Joint-solver value-before-mpc bundle order (`62c6429`).
> - Bundle symbol-collision fixes (`4f77878`).

## TL;DR

- **Phase 0** (bundle infra): 3 distinct bug classes fixed. Bundles
  now load cleanly with correct topology env-var threading and bundle-
  vs-source parity.
- **Phase 0.5** (physics probe): GREEN. 2022 fleets across 8 seeds,
  100% target rate, 0 sun, 0 oob via `predict_fleet_fate`.
- **Phase α** (smooth ΔW value function): coded, pin-tested (6 new
  tests), null in isolation (4W/4L at n=8 vs step form).
- **Phase β** (topology features): bug fixed (kwarg signature
  silently swallowed), null in isolation (4W/4L at n=8).
- **Phase α+β stacked**: first directional positive (5W/3L → 9W/7L
  at n=16, point estimate 0.56-0.62, Wilson [0.332, 0.769]). Plan-
  agent's predicted ordering effect confirmed.
- **Phase ε.1** (adversarial maximin search): coded, pin-tested,
  null at K=3 + closed-form leaf (4W/4L). Top-K not diverse enough
  at current implementation.
- **4P A/B** (Rule 43 pre-push gate): α+β stacked focal vs
  no-features background, n=4 (16 games). Running.

## Phase-by-phase A/B record

All A/Bs via `scripts/clean_ab.py` (subprocess-per-game isolation),
seeds 0..N-1, workers=4.

| # | Focal | Opp | W/L | Wilson 95% | Wallclock | Per-seat |
|---|---|---|---|---|---|---|
| 1 | analytical_phase_c (env-var contam) | _phase4_step1_FND | 4/8 | [0.215, 0.785] | 115s | mirror |
| 2 | topology_on (hardcoded β) | topology_off | 4/8 | [0.215, 0.785] | 117s | mirror |
| 3 (post bugfix) | topology_on | topology_off | 4/8 | (covered) | n/a | mirror |
| 4 | smooth_dw_on (α) | smooth_dw_off | 4/8 | [0.215, 0.785] | 153s | mirror |
| **5** | **alpha_beta_on (α+β)** | **alpha_beta_off** | **5/8 = 62.5%** | **[0.306, 0.863]** | **166s** | **3/4 mirror** |
| **5b** | **alpha_beta_on n=16** | **alpha_beta_off** | **9/16 = 56.2%** | **[0.332, 0.769]** | **275s** | **7/8 mirror** |
| 6 | maximin_on (α+β+ε.1) | maximin_off (α+β) | 4/8 | [0.215, 0.785] | 175s | mirror |

## Bugs found / fixed

1. **Bundler indent-preservation** (both `bundle_agent.py` and
   `bundle_analytical_phase_c.py`): function-local intra-package
   imports inside `try:` / `if:` blocks emitted alias rebinds at
   column 0 → IndentationError. Fixed by preserving `_leading_ws(line)`.

2. **`lib/kinematic_table.py` missing from DEFAULT_LIB_ORDER**: bundler
   didn't inline it; `agents/baseline/main.py:224` import failed.

3. **`lib/joint_solver/trajectory_matrix.py` + `opening_search.py`
   missing from JOINT_SOLVER_ORDER**: Phase η.1/η.2 modules
   transitively required by `lib/pipeline/opening.py` weren't bundled.

4. **Module-level name collisions** (`_DEFAULT`, `get_default`,
   `clear`, `_as_dict`, `_build_columns`, `_num_seats`,
   `_ships_to_capture`, `_solve_milp`, `_greedy_fallback`,
   `_source_inventory`, `_kinematic_table_enabled`, `_build_candidates`):
   13 colliding private names across modules. Each rename was
   surgically scoped — public API preserved via `import as` aliases.

5. **Topology env-var threading**: `_TOPOLOGY_FEATURES_ENABLED` was
   a module-level constant evaluated at lib-section import (before
   the agent's `os.environ.setdefault` runs). Result: both bundles
   reported False regardless of intent. Converted to lazy functions
   reading env at call time. Plus: bundle_analytical_phase_c was
   double-inlining joint_solver — disabled the redundant pass.

6. **Topology code path silently dead**: `_per_planet_topology_score`
   was called positionally as `(pid, world, model, sense, my_id)`
   but signature uses keyword-only (`*, my_id`). `try/except` swallowed
   the TypeError, leaving `topology_scores=None` and the entire
   topology block short-circuiting to 0. Fix: `my_id=int(my_id)`
   in call. Diagnostic post-fix: 1156 calls over an 80-step game,
   sum|value|=27947 — topology now contributes correctly.

7. **env.run contamination during A/B**: even with subprocess-per-
   game (clean_ab), env.run loads both agents in the SAME Python
   process; first-loaded agent's setdefault wins. Solved with
   `scripts/build_topology_variants.py` — produces 8 hardcoded-
   constant variants by rewriting each `_*_enabled()` body to a
   literal `return True/False`. Cross-load test confirms independence.

## Phase α — smooth ΔW value function

Replaces the step `_endgame_bonus` (±LAMBDA_ENDGAME=1000) with
`λ_W · ΔW` where `ΔW = winning_margin_after − winning_margin_before`
and `winning_margin = prod_advantage × remaining_turns − opp_pool`.

Closed-form per (planet, subset). Signed, magnitude-proportional to
planet importance — properties the step form lacks. Default λ_W=0.3
per sister session's W calibration (r=0.545, marginal).

A/B result alone: 4W/4L at n=8, Wilson [0.215, 0.785]. Step counts
DIFFER from Phase β (255 vs 196 on seed 1) — confirms LP IS picking
different actions. Outcome neutrality at λ_W=0.3 is expected when
smooth ΔW and the step form are both reasonable.

Files: `lib/joint_solver/predicate.py` (winning_margin helper),
`lib/joint_solver/lp_outcome.py` (step/smooth dispatcher),
`tests/test_lp_endgame_predicate.py` (6 new tests, all green).

## Phase β — Level 1 topology features

Bug 6 was the dominant finding here — topology code was firing zero
times. Post-fix, topology contributes correctly but doesn't tip the
LP argmax in isolation because LAMBDA_ENDGAME=1000 step dominates.

A/B result alone: 4W/4L at n=8, Wilson [0.215, 0.785]. Identical
step counts to the env-var-contaminated first run, but for the
RIGHT reason now: topology values are small relative to the step.

## Phase α+β stacked

n=8: 5W/3L (62.5%), Wilson [0.306, 0.863].
n=16: 9W/7L (56.2%), Wilson [0.332, 0.769].

Step counts diverge across games (143-500), unlike the seat-mirror
pattern of phases 1-4. **First directional positive in this session**.

Confirms Plan-agent's pressure-test prediction: α must come BEFORE
β because the step crowds out topology. With smooth ΔW + topology
sharing comparable magnitudes (200-1500 vs 50-300), the LP weighs
them against each other meaningfully.

Per Rule 45 (n≥32 for lift claims): Wilson-lo of 0.332 at n=16 does
NOT clear the gate. To CLAIM a +6pp lift requires n ≈ 80 — bad EV.
Documented as inconclusive, escalated to Phase ε.1.

## Phase ε.1 — adversarial maximin search

Built per plan: top-K our portfolios (LP-seeded) × top-K opp
responses (mirror analytical, one per our portfolio) × closed-form
leaf eval (`leaf_value_for_portfolios`). Maximin selection — pick
the our-portfolio that maximizes worst opp response.

A/B (vs α+β stacked baseline): 4W/4L at n=8, Wilson [0.215, 0.785].
Perfect per-seat mirror on all 4 seeds. **Maximin at K=3 + closed-
form leaf does not add signal on top of α+β.**

Plan-agent's "Top-K not diverse enough" risk confirmed. The closed-
form leaf uses the same math the LP optimizes against, so top-K
portfolios all score similarly against the K opp responses; maximin
degenerates to argmax.

## Recommendation

**Do not submit anything from this session without further n=32
validation.** The α+β stacked variant has the best directional signal
(point estimate 0.56-0.62) but is inconclusive at Wilson noise.

**Open paths to a real lift**:

1. **n=32-64 confirmation of α+β stacked** — directly measures the
   true effect size. ~25-50 min wallclock at 4 workers. Could resolve
   the question definitively.
2. **λ_W sweep** at α+β config: try λ_W ∈ {0.1, 1.0, 3.0}. Finding a
   sweet-spot calibration could push the lift above the noise.
3. **Top-K diversity in maximin** (Plan-agent #3): enforce primary-
   source diversity in top-K selection. Currently the LP-seeded
   enumeration picks variants of the same idea.
4. **fast_sim leaf evaluation in maximin** (Phase ε.2): closed-form
   leaf uses the same math as the LP, so it provides no new signal
   relative to argmax. fast_sim 15-tick rollouts with lite_greedy
   on both sides would give a more independent estimate.
5. **Phase γ (Lagrangian inner)**: planned but skipped this session.
   Would speed up the LP from ~300ms to ~20ms, unblocking K=4-6
   maximin and Stackelberg outer.

**Off-ramp (per plan)**: if all of the above null at n=32, pivot to
precision-physics probe (`agents/precision/`) OR Konbu17-style ML
shot validator (H14 in `state/hypothesis-board.md`).

## Files created / modified this session

```
audit/2026-05-23/
  phase0-bundle-fixes.md
  phase-beta-result-and-next.md
  phase-alpha-beta-stacked.md
  session-summary.md          (this file)

lib/joint_solver/
  lp_outcome.py               smooth-ΔW dispatcher + lazy gates
  predicate.py                winning_margin helper

lib/pipeline/
  decision_lagrangian_maximin.py     (NEW) Phase ε.1 module

scripts/
  build_topology_variants.py  8-variant builder (topology / smooth-ΔW / maximin × on/off)
  build_fnd_baseline.sh        sed-based FND comparison bundle
  phase0_5_physics_probe.py   (NEW) 20-seed sun/oob inventory
  bundle_analytical_phase_c.py   PIPELINE_ORDER + JOINT_SOLVER_ORDER updates

tests/
  test_lagrangian_maximin.py   (NEW) Phase ε.1 pin tests
  test_lp_endgame_predicate.py   +6 Phase α tests
  test_lp_topology_features.py   env-var monkeypatch refactor
```

## Live ladder context (NOT re-pulled this session — must re-pull at next session start per Rule 43)

Per `state/MULTI_BRANCH.md` (snapshot 2026-05-23 21:00 UTC):
- Rolling pair: sub 52894340 (μ=1117.9, our `_phase4_step1_FND`) +
  sub 52893236 (μ=1078.0, `baseline_full`).
- Team peak EVICTED: sub 52744856 μ=1149.2 (composite_a2_hybrid).
- Daily submission budget: 5/day. Used today: 0.

No submissions this session.

## Decision points for next PI sync

1. Submit α+β stacked anyway, accepting Rule-45 risk that the +6pp
   may be noise? (Cost: a rolling-pair slot.)
2. n=32 confirmation of α+β before any submission?
3. Pivot to ε.1 improvements (diversity + fast_sim leaves) before
   any new submission?
4. λ_W sweep first?
5. Step away from LP family entirely — precision-physics or ML
   shot validator?
