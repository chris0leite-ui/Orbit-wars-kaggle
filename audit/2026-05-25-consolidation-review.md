# CONSOLIDATION review — findings + closed-track cross-reference

Date: 2026-05-25. Inputs: `audit/2026-05-25-consolidation-profile.md` (cProfile data) + read-only walk of `agents/buildup_planner/consolidation.py`, `agents/baseline/main.py`, `agents/baseline/chooser_trajectory.py`, `agents/baseline/proposer.py`, `agents/baseline/value.py`, `lib/trajectory.py`, `lib/kinematic_table.py`.

Per `state/MULTI_BRANCH.md`, the **wallclock performance axis is documented as UNTRIED** and orthogonal to the 8 closed behavior tracks.

## Stage-2 findings

### FINDING-K1 [PERF, ★★★ — top priority]: kinematic_table is built, parity-gated, and **not wired** on this branch

`lib/kinematic_table.py` exists in our branch (pulled in via `72fe45a` to origin/main). `lib/trajectory.py:264 _table_window_or_none` is the hook in `predict_fleet_fate`: when `KINEMATIC_TABLE_ENABLED=1` AND the singleton was primed via `begin_turn(world)`, the inner `predict_relative`-per-planet-per-step loop becomes a one-call dict lookup.

**The wiring exists on sibling branches but NOT on `claude/agent-design-exploration-Q0q9T`**:

- `923852e` (`origin/claude/extract-physics-trajectory-Vjaz9`) added the priming hook to `agents/baseline/main.py` — calls `kinematic_table.begin_turn(world)` after `World.from_obs(obs_d)`.
- `c48e143` (`origin/claude/strategy-axis-decision-3437`) added `os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")` to each active agent's main.py. Commit body documents:
  > 564 brute-force FleetFate parity assertions byte-identical across 500 random + 14 edge + 50 comet-target cases (5 seeds × game states). 2 full-game byte-identical move lists (seeds 42, 7; 120 steps each). Wall-clock measurement on the same runs: **47 ms/step saved on seed 42 (425 → 378), 114 ms/step saved on seed 7 (622 → 508)**.

**Why this hasn't reached us:** the two sibling branches that built the wiring never merged to main. Our branch inherited the LIBRARY (extracted-substrate commit `72fe45a`) but not the SCHEDULER hook (the `begin_turn` call) and not the `setdefault` flip.

**Impact estimate:** the documented wallclock saving is 47-114 ms/step — well over the 100ms gap between our production p95 (1101ms) and the budget (1000ms). Per the cProfile, `predict_fleet_fate` cumulative time is 116s / 219 turns = 530ms/turn average; the table removes the inner `predict_relative` rebuild (84s of own time across 48.7M calls) which is exactly the loop the cache targets.

**Remediation:** two-line plumbing change — port the `begin_turn(world)` block from `923852e` into our `agents/baseline/main.py` AND add `os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")` to `agents/buildup_planner/main.py`. NO library-side changes; the parity gate has already passed on the same library code. Zero new logic.

**LOC risk:** ~5 lines, one file each. Risk = 1 (isolated, parity gate already proves bit-identical).

---

### FINDING-K2 [PERF, ★★ — secondary]: additional trajectory-perf commits on sibling branches

Two more not-on-our-branch commits target `predict_fleet_fate`:

- `5702e8b perf(trajectory): per-fleet planet pre-filter in predict_fleet_fate` — only on `origin/claude/extract-physics-trajectory-Vjaz9`.
- `518414c perf: cache planet_positions per (world, wait_N) in predict_fleet_fate` — only on `origin/claude/review-skills-improvements-moKOR`.

These would compose with K1 (different layers of the same hot path). Estimated incremental savings unknown without measurement.

**Risk vs reward:** lower than K1 because (a) less validated (no documented parity gate); (b) compose-with-K1 effect may be sub-additive; (c) `518414c` is a per-(world,wait_N) cache that re-implements part of what kinematic_table does — likely overlaps. **Recommend defer until K1's empirical lift is measured on our branch.**

---

### FINDING-O1 [DESIGN]: opening_plan() is called 30× per game post-`843cc35`

Per `audit/2026-05-25-consolidation-profile.md` Finding 2: `opening_plan` runs 30 times (once per opening turn), cumulative 51s in a 219-turn game. The MILP solve itself is cheap; the cost is `_build_candidates` (`opening_planner.py:409`), which calls `predict_fleet_fate` per candidate.

**Composition with K1:** if K1 lands, `_build_candidates` automatically gets the same speedup; the 1.7s/call opening turn drops proportionally to its predict_fleet_fate share. We DON'T need a separate fix here — K1 fixes both the consolidation hot path AND the opening hot path simultaneously.

This explains why the previous cache-the-schedule experiment only saved 5ms p95: the schedule cache removed 29 of 30 calls but each call's COST was internal to predict_fleet_fate, which is per-candidate and per-turn regardless. Fixing the inner loop is the right level.

---

### FINDING-S1 [DESIGN, low impact]: score_candidate_v4 admissibility filter overlaps with proposer's

`chooser_trajectory.py:533` calls `predict_fleet_fate(src, tgt, angle, ships, world)` per candidate. `proposer.py:1071` ALSO calls `predict_fleet_fate(src, tgt, float(angle), int(ships), world)` for the SAME (src, tgt, ships, angle) tuples (the proposer's own admissibility filter, opt-in via `PROPOSER_ADMISSIBILITY`, default off — but the chooser one is always-on).

**If `PROPOSER_ADMISSIBILITY=on`** (rare in production), the same trajectory gets computed twice per candidate. **In production (default off)**, only the chooser-side call fires, so this is mostly a hypothetical issue. Still, a turn-local memoisation of `predict_fleet_fate` keyed by `(src.id, tgt.id, ships, round(angle,6), wait_N)` would deduplicate even single-site repeated calls within the same world snapshot.

**Composition with K1:** K1 caches the inner `predict_relative` loop. A turn-local memo on `predict_fleet_fate` itself is a HIGHER level cache. Could compose. **Defer until K1 measured.**

---

### FINDING-C1 [CORRECTNESS, low impact]: profile termination at step 219

The seed 1622482326 game ended at step 219 of 500. Either our agent or `phi1_only` was eliminated. Not a bug per se, but the profile data is biased toward the opening/midgame; we haven't measured deep-late-game turn times where state is densest.

**Remediation:** when validating K1's lift, also profile a seed that DOESN'T early-terminate (e.g. one of the geometry-panel seeds where games typically run to step 500). Defer to validation step.

---

### FINDING-V1 [BUG? — needs verification]: `consolidation.py:36` setdefault race

`agents/buildup_planner/consolidation.py:36` says:

```
# NOTE: BASELINE_VALUE_HEAD=phi setdefault lives in
# agents/buildup_planner/main.py (which runs BEFORE the baseline import
# in this module, so the bundle's earlier-running baseline setdefault to
# "hybrid" loses).
```

But `agents/buildup_planner/main.py:59` actually says:

```
os.environ["BASELINE_VALUE_HEAD"] = "phi"  # HARD SET (not setdefault)
```

So the comment in consolidation.py is wrong — main.py uses HARD SET, not setdefault. **Documentation drift, not a bug.** Worth fixing the comment in a one-LOC cleanup later but doesn't block anything.

---

## Stage 3 — closed-track cross-reference

| Finding | Closed tracks touched? | Verdict |
|---------|-------------------------|---------|
| K1 (wire kinematic_table)  | None. Wallclock axis is UNTRIED per `state/MULTI_BRANCH.md`. Parity already gated. | **OPEN-OK** |
| K2 (extra trajectory perf) | None on our branch. | **OPEN-OK** but DEFER |
| O1 (opening_plan calls)    | Caching schedule already falsified (commit `9870575` revert). K1 fixes this at a lower level. | **OPEN-OK** (different mechanism) |
| S1 (per-turn memo)         | None. | **OPEN-OK** but DEFER |
| C1 (profile bias)          | N/A (verification only). | — |
| V1 (stale comment)         | N/A (cleanup). | — |

No closed-track entanglement. K1 is unambiguously OPEN-OK and the highest-priority candidate.

## Stage 4 — prioritised fix plan

**Single recommended move: implement FINDING-K1.** Two-file patch:

1. `agents/baseline/main.py` — port the `begin_turn(world)` block from commit `923852e`. Exact diff (10 lines):

   ```python
   world = World.from_obs(obs_d)

   if os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() in (
       "1", "true", "on", "yes",
   ):
       try:
           from lib import kinematic_table as _kt
           _kt.begin_turn(world)
       except Exception:
           pass

   model = WorldModel.from_world(world)
   ```

2. `agents/buildup_planner/main.py` — add the setdefault near the other env defaults (line ~47, beside `BASELINE_ORBITAL_SAFETY`):

   ```python
   os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")
   ```

### Verification gates (Rule 38)

1. **Parity reproduce.** Run `python -m pytest tests/test_kinematic_table_parity.py -q` on our branch — must be 100% GREEN (the gate that `c48e143` ran).
2. **Local timing spot-check.** Re-run `scripts/profile_consolidation.py 1622482326` with the patch applied. Expected: `predict_relative` cumulative time drops by ≥ 60% (currently 84s; expect ≤ 35s).
3. **p95 measurement.** Run one full game at the same seed with timing recorded. Expected: p95 drops from 1101ms (pre-fix production baseline) by 50-100ms.
4. **Rule-45 A/B.** `python fast.py eval agents/buildup_planner --vs submissions/buildup_planner_phi1_only.py --max-seeds 32 --gate 0.50`. Behavior change should be ZERO (kinematic_table is bit-parity by design); the test is that we don't accidentally introduce a non-determinism. Wilson-lo ≥ 0.50 expected — but **the goal here is wallclock, not winrate**; even a strict parity result (50% ± noise) is a pass.
5. **Rule-46 bundle gate.** `bundle_agent.py` + `tests/test_bundle.py` + `fast.py play <bundled>`.
6. **Rule-42 push-claim board entry.**

### Why this is "different" from previous wallclock work

The previous cache-the-MILP-schedule experiment (`9870575`, reverted) tried to cache OUTPUT of expensive computation. That regressed because re-deriving each turn was a behavioral feature (responding to mid-opening state change). K1 caches an INPUT to the computation (planet positions) — which are deterministic by construction (orbital mechanics, no decision dependence). Parity is guaranteed; behavior is unchanged.

## Risks

- **Coverage drift.** The table is sized for `max_lead=500`; `predict_fleet_fate` uses `max_steps=200` plus wait_N (up to ~50). 500 is generous, but a corner case with wait_N > 450 would fall through to inline build. Acceptable: `_table_window_or_none` handles the miss case by returning `None`, falling back to current behaviour.
- **Sibling-branch drift.** Since `923852e` was on `extract-physics-trajectory-Vjaz9` (a branch that didn't merge), the priming code MAY have minor differences from what our `baseline/main.py` expects today. **Verify by reading both versions side-by-side before applying the patch.**
- **Bundle inlining.** `scripts/bundle_agent.py:48` mentions kinematic_table specifically — the bundler is already aware. Double-check it correctly inlines `lib/kinematic_table.py` into the submission file at bundle time.

## Next steps (post-PI approval)

1. Read `agents/baseline/main.py` line ~860 area and `923852e`'s diff side-by-side to apply the priming patch cleanly.
2. Apply the patch (one commit).
3. Run all 6 verification gates.
4. If all green, present the wallclock numbers + parity status to the PI for the submit decision.
