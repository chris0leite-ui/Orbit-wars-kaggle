# Phase β result + decision to do Phase α next

> 2026-05-22 evening, branch `claude/strategy-axis-decision-3437`.
> Plan ref: `/root/.claude/plans/composed-noodling-riddle.md`.

## Phase β — topology features A/B

### First run (env-var contaminated)

```
focal=analytical_phase_c.py (topology setdefault=1)
opp=_phase4_step1_FND.py   (topology setdefault=0)
seeds=4 (×2 seats = 8 games)
focal_wins=4/8 (50.0%), Wilson [0.215, 0.785]
```

Every seed flipped with seat swap — perfect mirror, identical step
counts both seats. Root cause: `clean_ab.py` subprocess-isolates per
GAME but `env.run` loads BOTH agents in the SAME Python process
(kaggle_environments behavior). The first-loaded agent's
`os.environ.setdefault` wins; the second's is a no-op; both then
read the same env via the lazy `_topology_features_enabled()`.
Per-seat swap toggles which loads first, hence the perfect mirror.

### Second run (hardcoded-constant variants)

Built `submissions/topology_on.py` and `submissions/topology_off.py`
via `scripts/build_topology_variants.py` — the four lazy
`_*_enabled()` functions in each bundle have their body rewritten to
`return True` / `return False` literally, bypassing env reads. Verified
cross-load independence (on→True even when off loaded first).

```
focal=topology_on.py (hardcoded True)
opp=topology_off.py  (hardcoded False)
seeds=4 (×2 seats = 8 games)
focal_wins=4/8 (50.0%), Wilson [0.215, 0.785]
```

Identical 4W/4L, identical per-seed step counts. So **isolation is now
correct**, but the topology features have **near-zero behavioral effect
on outcomes at this calibration**.

### Why isn't topology moving the LP?

A diagnostic monkey-patch confirmed the topology code path FIRES
when the LP runs (52 `solve_outcome_aware` calls + 1156
`_per_planet_topology_score` calls + sum|value|=27,948 across an 80-step
game). The features are active. They just aren't tipping the LP's
argmax.

Two reasons in concert:

1. **`LAMBDA_ENDGAME = 1000` is a STEP function dominating the
   objective.** When a subset tips us into "winning state" (a closed-form
   predicate), the bonus is ±1000. Topology's λ_REACH=50 / λ_DEFENSE=10
   / λ_FRONT=30 are dwarfed; the LP's argmax is determined by the step,
   not by topology. The Plan agent's pre-pressure-test flagged this
   precise risk:
   > "Calibrating topology with that step still in the objective will
   > fit topology weights to compensate for endgame mis-pricing, not
   > to reflect topology value."

2. **Bug found and fixed during this session**: `_per_planet_topology_score`
   was being called positionally as
   `_per_planet_topology_score(pid, world, model, sense, my_id)` but
   the signature uses keyword-only (`*, my_id`). The TypeError was
   caught silently by a surrounding `try/except`, leaving
   `topology_scores=None` and the entire topology block short-circuited
   to 0. Fix: `my_id=int(my_id)`. Now the function fires correctly.

   The first hardcoded-variant A/B may have run with this bug still
   live (the rebuild happened after the bug-fix commit; both bundles
   should have had the fix, but the variants script may have been
   compiled against the older bundle). Re-running the A/B post-fix
   would be a good gate-check, but priority is now Phase α.

3. **Bundler structural fix landed**:
   `scripts/bundle_analytical_phase_c.py` was double-inlining the
   joint_solver/* modules (once via baseline_bundle from
   `bundle_agent.py`'s new `DEFAULT_LIB_ORDER`, once via its own
   `JOINT_SOLVER_ORDER`). Disabled the redundant inline; bundle size
   1063367 → 873977 bytes; topology-variant builder no longer sees
   8 lazy-fn blocks (was 4 expected).

### Decision per `composed-noodling-riddle.md` killgate table

| Gate | Result |
|---|---|
| Phase β at n=8, Wilson-lo ≥ 0.55 | **NULL** (point estimate 0.50, Wilson [0.215, 0.785]) |
| Off-ramp criterion | n=16 re-test only if point ≥ 0.55. We're at 0.50. **Topology axis exhausted at this calibration.** |

But: the Plan agent's pressure-test predicted exactly this — Phase α
must come BEFORE Phase β. The smooth-ΔW objective replaces the step
function; once topology and the smooth ΔW share the same magnitude
regime (low-thousands), topology's lambdas can matter. The current null
on topology is **diagnostic of the wrong order, not a falsification of
topology**.

## Next step: Phase α (smooth ΔW value function)

Per Plan-agent ordering: α before β. Sister session's W-calibration
on 20 replays (`audit/2026-05-23/calibrate_W_results.json`):

| Check | Result | Gate | Pass |
|---|---|---|---|
| (b1) Pearson r(W, focal_reward) | 0.545 | ≥ 0.6 | marginal |
| (b2) ΔW spread vs threshold | stdev=2092 vs 109 | non-zero | YES |
| (b3) Pearson r(ΔW_per_action, outcome_shift) | 0.044 | ≥ 0.3 | NO |

Interpretation: W has weak but signed global predictive power; per-
action ΔW is approximately additive but noisy. Implication: use a
**conservative λ_W** — start at 0.3, not 1.0 — and re-run the A/B
with the smooth term replacing the step.

After α clears (or null-but-isolated), re-do Phase β with topology on
top of smooth ΔW.

## Artifacts produced this Phase β cycle

- `scripts/build_topology_variants.py` — produces hardcoded ON/OFF bundles.
- `submissions/topology_on.py` / `submissions/topology_off.py` — diff
  only in 4 lazy `_*_enabled()` function bodies.
- Bug fix: `lib/joint_solver/lp_outcome.py:852` keyword arg.
- Bundler fix: `scripts/bundle_analytical_phase_c.py` no longer
  double-inlines joint_solver.
- Two clean A/B records under `audit/2026-05-23/` (this file).
