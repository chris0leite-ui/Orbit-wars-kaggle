# HANDOVER.md — next-session brief

> Last written: 2026-05-23 by `claude/consolidate-codebase-refactor-dQAWA`
> (coord Day-13 — five new features shipped in two back-to-back submissions;
> PI flagged "we may need to prune what doesn't work" once v3's μ settles).
> Prior: 2026-05-20 PM by `claude/review-skills-improvements-moKOR`
> (n=8 iteration loop attempt; no candidate found, structural-change
> pivot queued).
> Prior PM session on this branch (cross-branch consolidation pass)
> notes preserved under "What just landed (2026-05-20, this session)".
> Prior writers (per-branch, now superseded): `kaggle-baseline-strategy-lO4mm`,
> `audit-workflow-performance-btjeK`, `strategy-framework-design-OyoYR-rebased`,
> `ml-competition-strategy-PFhzM`, `analyze-game-strategy-EpMVP`.

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, three-track
   registry (Analytical / Hybrid-Sim / Verify-first), closed tracks,
   push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, single-game diagnostics,
   validation suite, consolidation-merge gate.
3. **`CLAUDE.md`** — rules 1-47 (rules 41-47 added 2026-05-20).
4. **This file** — session-start prompt below.
5. `audit/friction.md` if you're about to touch a fragile path.

## Where we are (2026-05-23 — coord branch wrap)

**Live submission: sub coord v3** (just submitted; not yet matched).
- 4P-aware bundle-market coord with smooth-ΔW endgame bonus + Option 3
  LITE demand-spread mixing + raised barriers (LEAF_FLOOR=2.0,
  REDUCED_FLOOR=2.0) + deadline-bounded enumerate.
- Self-evicted sub 52936894 (coord v2, never settled — was up for ~10 min
  before v3 replaced it).
- Rolling pair will be: coord v3 (new, PENDING μ) + sub 52935965
  (orbitfix_kt_p23, μ=1091.3).

**Next-session first action:**
1. `kaggle competitions submissions orbit-wars | head -5` to read v3's μ.
2. **Decision tree based on v3 μ:**
   - μ ≥ 1100: features collectively help. Tune
     COORD_OPP_CAPACITY_FACTOR + DEMAND_REACH_WINDOW. Upgrade Option 3
     LITE → canonical (per-opp shadow price).
   - μ ∈ [900, 1100]: at-parity with old coord (μ=905). The five-knob
     pile-up cancelled out. **PRUNE** in order: DEMAND_SPREAD off →
     REDUCED_FLOOR=0 → LEAF_FLOOR=0 → DELTA_W=0. Stop at the knob whose
     removal improves A/B vs orbitfix.
   - μ < 900: features collectively HURT. Revert to pre-Day-13 coord;
     keep ONLY the deadline fix and code-review correctness fixes.

**The prune-needed concern (PI explicit 2026-05-23):** five env-var-gated
features were added between Day 12 and Day 13. They're all default-ON in
sub coord v3. If μ moves significantly in either direction without
single-feature isolation, we won't know which feature is responsible.
Pruning must happen before any new feature work.

See `knowledge-base/flags/2026-05-23-coord-five-knobs-need-pruning.md`
for the full knob list and pruning method; `knowledge-base/questions/
2026-05-23-which-knobs-help.md` for per-knob predicted directional
effects.

## (Older context follows) Where we were (2026-05-20 17:00 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **34 days remain.**
- **Rolling-last-2 (Kaggle auto-keeps these two):**
  - 52857903 (μ 806.5) — analytical_wait_N_traj_plus_endgame_play (2026-05-20 16:12)
  - 52854094 (μ 829.1) — analytical (2026-05-20 13:59)
- **Team peak (EVICTED):** μ 1149.2 (sub 52744856, composite_a2_hybrid, 2026-05-17).
- **Floor lost in 24 h:** ~320 μ. The five-step eviction chain that
  caused this is documented in `state/MULTI_BRANCH.md` and is the
  origin of new Rule 42 (pre-submit cross-branch coordination gate).
- **Daily submission budget:** 5/day. 5/20 used: 2. 3 slots remain.
- **Floor-at-risk flag:** **TRUE** — rolling pair is 320 μ below team peak.

## Day-N PM review-skills-improvements-moKOR (2026-05-20 evening)

**Session shape:** n=8-capped A/B iteration loop attempting to beat sub
52827111 ("comet-aim + reactor-aware", μ=1122). PI directive: no
submission until a candidate shows significant lift at n=8 (gate
≥14/16 = Wilson-lo 0.524). Result: no candidate found. Pivot
direction surfaced at end of session.

### What landed (code + docs)

- **Setup (3 commits + bundler fix):** targeted `git checkout` of sub
  52827111's mechanism source from `claude/audit-workflow-performance-btjeK`
  onto this branch (`d642593`). Imported files:
  `agents/baseline/{proposer,chooser,value,main,chooser_trajectory,chooser_roi}.py`,
  `lib/{world_model,trajectory,aim,opp_model,value_heads}.py`,
  matching tests, and `scripts/bundle_agent.py` (btjeK upgrade with
  parity-gate cache).
  - Bundler indent-preservation fix (`9a45fea`): bundler was breaking
    function-local intra-package imports by hoisting alias rebinds to
    column 0 inside function bodies → IndentationError. Fixed at
    `scripts/bundle_agent.py:268-275`.
  - `.gitignore` for `audit/bundle-parity-cache.json` (`3f123c3`).
- **Pinned baseline:** `submissions/iter_baseline.py` = clean re-build
  of the deployed sub-52827111 bundle (parity-gate green).
- **Iter 1 audit:** `audit/2026-05-21-n8-iter1-reactor-ablation.md`
  (filename off by one day vs UTC; content correct). Documents the
  parallel-vs-serial discrepancy that invalidated the original Iter 1
  diagnostic.

### Load-bearing findings

1. **CPU-contention contaminates n=8 A/Bs.** Three parallel `fast.py
   eval` instances (24 worker processes) produced focal p95=1248ms (over
   the 1000ms env actTimeout). Variant 1b reported 12/16 (75%) under
   contention; same bundle re-tested serially gave **6/16 (37.5%)**.
   Variant 1a similarly fell from 11/16 to 7/16 serial. **Mandatory
   convention going forward: all n=8 A/Bs run serially, no parallel
   fast.py invocations.**

2. **No env-var ablation produces ≥14/16 lift over the deployed
   baseline.** Four serial n=8 runs (all clean wallclock):

   | Variant | Δ vs deployed | Wins | Wlo |
   |---|---|---:|---:|
   | A1 — comet-aim solo (reactor-aware OFF) | 7/16 (43.8%) | 0.231 |
   | A2 — Part B (reactor candidates) OFF | 7/16 (43.8%) | 0.231 |
   | A3 — BASELINE_COMET_AIM=off | 9/16 (56.2%) | 0.332 |
   | A4 | killed before completion (PI directive — see #3) |

   Three runs all landed at 7/16, A3 at 9/16. All INCONCLUSIVE; no
   candidate cleared the gate.

3. **PI verdict mid-loop ("your tests are meaningless, we need a big
   lift"):** env-var ablations tap out at ±5pp which is invisible at
   n=8 (Wilson CI ~±20pp). To produce a ≥14/16 lift over a near-optimal
   bundle requires a STRUCTURAL change, not a knob flip. Loop halted
   at A3 result.

4. **Structural-change candidates that are NOT yet new code on this
   branch:**
   - **`used_tgts` lock removal in `chooser_trajectory.py:898`.**
     Currently blocks multi-source-same-target SOLO emits even when
     JOINT is on; JOINT only fires for pre-paired candidates (capped
     JOINT_TOP_K_PER_TARGET=3, JOINT_MAX_PAIRS=20).
   - **JOINT expansion** — raise the per-target / global pair caps by
     5-10×; remove the lock-checks at `chooser_trajectory.py:885-888`.
   - **Composite value head + A2 restoration** (the μ=1149 team-peak
     architecture). `value.py` has `BASELINE_VALUE_HEAD=composite` opt-in;
     A2 4P-weakness logic also imported.
   - **New chooser** (MCTS / beam search over candidate set) — 1+
     day build.
   - **Increase N_VALIDATE / WALLCLOCK budget** — squeezes the existing
     chooser only marginally; unlikely to be a "big lift."

5. **Confirmed already-implemented (not new work):** `BASELINE_LEDGER=on`
   (wait-N inter-turn commitment memory, the original Iter 4 idea —
   already in chooser_trajectory.py lines 904-915, gated by env var
   defaulting to "off"). `BASELINE_JOINT=1` multi-source coalitions
   (already ON by default, just capped low).

### Verified gaps in the current chooser

- **`agents/baseline/proposer.py:926-928`**: wait_N>0 candidates bypass
  the trajectory filter (`predict_fleet_fate` returns wrong results
  because it doesn't pre-rotate src/tgt to launch time). This is real
  H44 surface: filter has zero coverage for the multi-wait grid.
  Iter 3 (planned, not yet implemented) would extend
  `predict_fleet_fate` with a `launch_step` arg.
- **`predict_fleet_fate` does NOT check enemy-fleet intercepts.** This
  is correct behavior — game rules confirm fleet-vs-fleet collision
  doesn't exist. Original Iter 3 framing ("add enemy fleet ray-cast")
  was based on a misread of the game spec.

### Falsified or weakened this session

- **"Part A (cost-parity filter) is the regressor."** Iter 1's
  parallel-run 12/16 was CPU-contention noise; clean serial gives
  parity-or-loss (6/16). Cannot blame Part A based on this data.
- **"Comet-aim is the key lift in sub 52827111."** A3 turned comet-aim
  OFF and got 9/16 (better than 7/16 from other ablations).
  Directional signal that comet-aim itself may be neutral-or-mildly-
  harmful, not the value-add of the push.
- **Floor-recovery via rebundle of `iter_baseline.py` (== sub 52827111).**
  PI rejected: "we can learn nothing from that." Path is OFF.

## Next-session first action (this session's pivot)

**Priority 1 — Pick one structural change from the list above and ship
it (~few hundred LOC, single axis).** Recommend `used_tgts` lock
removal + JOINT cap expansion in `chooser_trajectory.py` as the
cheapest structural-shape change: combat rule 1 (same-owner same-step
arrivals stack) is well-understood; the existing lock literally
forbids the most powerful combat pattern. Risk: Plan agent flagged
this as needing n=32 minimum (prior asymmetric chooser attempts
0/32). Run n=8 serial first; if directional, escalate to n=32.

**Priority 2 — if Priority 1 doesn't clear:** Composite value head +
A2 restoration (μ=1149 architecture). Code already imported; needs
the right env-var combo + bundle bake. Significant ladder evidence
(sub 52744856 live μ=1149).

**Priority 3 — out-of-session-scope:** Konbu17 shot-validator MLP
(~1 week build, but the only ML attack with empirical precedent
+19pp panel lift).

**Reading order for the next agent:** this section first, then
`audit/2026-05-21-n8-iter1-reactor-ablation.md`, then
`/root/.claude/plans/go-effervescent-mochi.md` for the full
iteration ladder context.

## What just landed (2026-05-20, this session)

This session was a **doc-only consolidation pass** across 8 active
branches. No code changed. New / edited docs:

| File | Change |
|---|---|
| `state/MULTI_BRANCH.md` | **NEW.** Single source of truth across branches. |
| `state/TOOLS.md` | **NEW.** Tools registry (A/B + diag + validation). |
| `CLAUDE.md` | Rules 41-47 appended. Pointers section adds MULTI_BRANCH + TOOLS. |
| `.claude/skills/kaggle-comp/SKILL.md` | Step 0 "load MULTI_BRANCH + TOOLS first" preamble. |
| `.claude/skills/kaggle-comp/day-loop.md` | Step 1 amendment for code-comp branch coordination. |
| `.claude/skills/kaggle-comp/improvements.md` | Rotated: 7 items promoted to rules; 2 superseded. |
| `.claude/skills/kaggle-comp/improvements-archive-2026-05-20.md` | **NEW.** Rotation archive. |
| `state/current.md` | Deprecated to pointer-only banner. |
| `state/mechanism-ledger.md` | Appended 2026-05-18 → 5-20 entries. |
| `HANDOVER.md` | Rewritten (this file). |

**Rules 41-47 summary (read CLAUDE.md for full text):**

- **41.** Confound-sweep before correlational conclusion (btjeK origin).
- **42.** Pre-submit cross-branch coordination gate (the ~320 μ loss origin).
- **43.** Multi-opponent panel mandatory pre-submit (supersedes `--vs-panel` pending item).
- **44.** State-of-truth read before subsystem edits (supersedes "read state docs" pending item).
- **45.** n ≥ 32 minimum for A/B lift claims.
- **46.** Bundle + parity smoke before any submission.
- **47.** Physics-primitive verification before agent design (PFhzM origin).

## Three parallel tracks — current state

| Track | Lead branch | Best result | Status | Next action |
|---|---|---|---|---|
| **A — Analytical chooser** | `strategy-framework-design-OyoYR-rebased` | μ 829.1 (sub 52854094) — both live pushes regressed | knowledge-base 5/20: "axis closed (10 slices, 0 lift)"; architectural bind: analytical needs multi-turn glue OR must replace rollout entirely | Decide: park, or pivot to analytical-leaf-inside-rollout |
| **B — Hybrid-sim production** | `audit-workflow-performance-btjeK` (production) + `analyze-game-strategy-EpMVP` (phases) | μ 1149.2 (EVICTED) | Live champion lineage. H44 finding 5/20: 65% fleet-destroyed-in-flight — new physics-driven mechanism candidate | (i) hold-feasibility solo validation (btjeK Phase B); (ii) H44 defensive mechanism design; (iii) EpMVP Phase 4/6 commissioning |
| **C — Verify-first + Goal-directed** | `ml-competition-strategy-PFhzM` (+ `precision-physics-engine-ymJkA` substrate) | Phase A Test 3 PASS; wrap-baseline 12/32 = 37.5% (only positive signal vs production) | greedy_expand (60 LOC) tied goal_planner (500 LOC); chooser axis confirmed neutral | Decide: is wrap-baseline-as-veto a viable design? Or is Track-C work substrate-only? |

## Next-session first actions (ranked by EV / cost)

### Priority 1 — code consolidation pass (start small, parity-tested)

Following the 6-step consolidation-merge gate in `state/TOOLS.md`:

1. **Substrate primitives** (no chooser changes piggy-backed):
   - Merge `lib/trajectory_layer.py` + `tests/test_trajectory_layer_positions.py` from PFhzM.
   - Merge `agents/precision/sim.py` + `agents/precision/intercept.py` + `agents/precision/tests/` from precision-physics branch.
   - Run consolidation gate; expected: GREEN.
2. **Bundler upgrade** from EpMVP — "inline agent submodules + explicit-name imports."
3. **Diagnostic scripts** (zero-risk leaf merges):
   - `scripts/h44_landing_capture_diagnostic.py` (btjeK)
   - `scripts/probe_emits_via_fate.py` (PFhzM)
   - `scripts/inspect_goal_planner_game.py` (PFhzM)
4. **Oracle tests** (test-only, zero-risk):
   - `tests/test_baseline_replay_regression.py` (EpMVP)
   - `tests/test_migration_solver.py` (EpMVP)

### Priority 2 — recovery submission planning

The rolling-last-2 is 320 μ below team peak. Three sub-IDs have evidence
of being strong:

- **52744856** (μ 1149.2, composite_a2_hybrid 2P + A2 4P)
- **52754310** (μ 1143.7, trajectory v4 + wait_N + wallclock)
- **52811320** (μ 1135.1, hold-feasibility solo)

**Open PI question:** which lineage to rebundle and push? The push will
itself need to clear Rule 42 (claim board) and Rule 43 (panel + h2h)
before submit. Do NOT push without explicit PI sign-off.

### Priority 3 — Track-B physics mechanism design

H44 finding: 65% of landing-capture failures are fleet-destroyed-in-flight.
This is a substrate-level signal that the trajectory chooser's
fleet-safety filter is not catching. Design a defensive mechanism
(NOT a restriction-tuning constant bump — Rule 40) that emerges from
the underlying physics.

## Pointers

- `state/MULTI_BRANCH.md` — cross-branch state-of-truth.
- `state/TOOLS.md` — tools registry + consolidation-merge gate.
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas, killed list.
- `CLAUDE.md` — rules 1-47 + R-defaults.
- `audit/friction.md` — current friction summary.
- `.claude/skills/kaggle-comp/` — skill (now multi-branch-aware).
- `audit/2026-05-21-h44-phase1-CORRECTED.md` (btjeK) — physics-failure analysis.
- `audit/2026-05-20-postmortem-strategy-framework-design-OyoYR-rebased.md` — analytical axis closure.
- `audit/2026-05-19-postmortem-PFhzM-physics-gate-and-mvp-confirmation.md` — Track-C verdict.
- `audit/2026-05-21-n8-iter1-reactor-ablation.md` (this branch, filename off by one UTC day) — Iter 1 ablation results + the parallel/serial contention finding.
- `/root/.claude/plans/go-effervescent-mochi.md` — full iteration-loop plan including the structural-change pivot list.

## Rule reminders (most relevant this session)

- **Rule 1:** submissions PI-approved, single-shot, no retry loops.
- **Rule 12:** rolling-last-2; weak late submits unrecoverable for ~24h.
- **Rule 32:** session-start git fetch; verify rolling pair via Kaggle CLI.
- **Rule 35-36:** PI thoughts append-only; session-end second-brain update.
- **Rule 37:** 3-variant axis cap. v9-v15 chooser hit it; chain-bonus hit it; analytical-slice hit it (10/3+).
- **Rule 40:** prefer modeling-correctness over restriction-tuning.
- **Rules 41-47 (new today):** see CLAUDE.md.

## Open questions for PI

1. Track A (analytical) — park or pivot to analytical-leaf-inside-rollout?
2. Track C — wrap-baseline-as-veto, or substrate-only contribution?
3. Recovery submission — which lineage to rebundle for the next push?
4. Should the SessionStart hook implementation (improvements.md TOP
   PRIORITY) get priority over the code consolidation pass?
