# HANDOVER.md — next-session brief

> Last written: 2026-05-19 PM by `claude/ml-competition-strategy-PFhzM`
> after 5 trajectory_roi iterations all failing 0-1/32 vs baseline.
> **Next session: analytics verification suite FIRST, then v4
> goal-directed portfolio planner.** Copy-pasteable session prompt
> lives at the bottom of
> `/root/.claude/plans/read-the-handover-do-abundant-quokka.md`.
> See `## Day-19 PM ml-competition-strategy-PFhzM` below for the
> load-bearing section.
>
> Prior writer: 2026-05-17 by `claude/kaggle-baseline-strategy-lO4mm`
> (clean modular re-baseline of v15).
> Earlier sessions: `claude/recover-main-foundations-MV0e2` and
> `claude/merge-2026-05-16-knowledge` (the v9 → v15 → v20 chooser line).

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC (35 days).
- **Last submission:** composite+A2 hybrid (sub #52744856, pushed
  2026-05-17 PM). Was PENDING at last session log; **live μ unknown
  as of 5/19**. Query at session start:
  `kaggle competitions submissions orbit-wars`.
- **Team peak agent (until live μ confirms otherwise):** v15_banded
  (sub #52710995, 5/16). The composite+A2 hybrid sub is the rolling
  candidate to replace it; verify on session start.
- **Rolling-last-2** (Kaggle auto-keeps these two for final eval;
  third push auto-evicts):
  - composite+A2 hybrid (5/17, sub #52744856) — PENDING last check
  - v15_banded (5/16, sub #52710995) — current champion (until
    composite+A2 clears)
- **Daily submission budget:** 5/day; 5/19 0/5 used (local A/B only).
- **Calibration WARNING** (active): -20 to -30 pp local-vs-live on
  the last three submissions. The new ROI agent uses
  **observation-grounded synthetic scenarios** as its primary gate
  (not tournament winrate) precisely because of this gap — see
  Day-19 PM section below.
- **Working foundation:** `agents/baseline/` (clean modular v15
  re-impl, current live-champion source). `agents/bundle/` is
  SHELVED — not iterated (see `knowledge-base/flags/2026-05-19-
  bundle-decision-stack-shelved-not-deleted.md`). `agents/trajectory_roi/`
  is the new build target for the next session.

## Pointers

- `agents/baseline/` — clean modular re-baseline of v15 (this branch).
- `tests/test_baseline_*.py` — 26 baseline tests.
- `lib/fast_sim.py`, `lib/game/interpreter.py` — the bit-exact forward
  simulator + game-rule engine; do NOT rewrite.
- `lib/opp_model.lite_greedy_policy` — the reactive opp model used in
  both `agents/baseline/chooser.build_idle_baseline` and `score_action`.
- `state/current.md` — submitted-agent state (no μ values; query Kaggle).
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas (H40, H42) + killed list.
- `audit/2026-05-16-v15-final-results.md` — v15's panel + h2h.
- `audit/2026-05-16-v16-v20-asymmetric-compounding-postmortem.md` —
  the v15→v20 chooser saturation iteration; Rule 37 application.
- `knowledge-base/thoughts/2026-05-17-baseline-functional-parity-with-v15.md` —
  this session's wrap-up.
- `fast.py` — single-file iteration entry: smoke / bench / eval / play.
- `scripts/bundle_agent.py` — bundler for submissions.

## Rule reminders

- Rule 1: submissions are single-shot, PI-approved. No retry loops.
- Rule 12: rolling-last-2 — Kaggle auto-keeps last 2 submits for final
  eval. Never push a speculative variant after a known-good submit
  unless you're willing to lose the good one's ladder spot.
- Rule 27 analogue: h2h vs the **current submitted agent** (not just
  a fixed baseline) is the FIRST gate, not the LAST. Panel pass alone
  is insufficient (v17, v18 lost h2h vs v15 despite panel pass).
- Rule 32: session-start `kaggle competitions submissions orbit-wars`
  is the source of truth for μ. State files do NOT record μ.
- Rule 37: 3-variant axis cap. The v9–v15 chooser axis hit it; future
  work must pivot to a different axis.
- Rule 40: prefer modeling-correctness over restriction-tuning
  (no MAX_WAIT / MAX_HORIZON / MIN_FLEET_SIZE bumps to fix symptoms).

---

## Day-19 PM ml-competition-strategy-PFhzM

**Strategic state.** Five trajectory_roi iterations shipped and
all losing to baseline 0-1/32 A/B. Architecture has hit a
ceiling. PI ratified pivot to: (A) build an analytics
verification suite FIRST; (B) replace value-maximization with a
goal-directed portfolio planner (winning-state predicate +
backwards capture sequence). No more iteration on trajectory_roi.

**Pivot in one sentence.** Verify the analytics, then build a
goal-directed planner that picks the smallest set of planets we
need to own to GUARANTEE victory (closed-form: prod-advantage ×
remaining-turns > opp recovery pool) and plans backwards to
acquire them.

**Iteration ladder this session (all A/B at n=32, vs baseline):**

| commit | version | wins | bench p95 | reason for failure |
|---|---|---|---|---|
| `d8db862` | v1 | 0/32 | 10ms | no in-flight awareness |
| `6707a90` | v1.1 | 0/32 | 22ms | turn-0 mirror blind to future counters |
| `e006b91` | v2 | 1/32 | 265ms | joint 2-opt, multi-source, defense — but single-snapshot mirror still misses reactive counters |
| `5bf2c23` | v3 | 0/32 | 1131ms | K=50 forward-projection blew env's 1000ms cap; actions dropped in late game. **Found and fixed `get_last_callable` loader bug — v3 silently emitted [] for entire games before this** |
| `5d88f4b` | v3.1 | 0/32 | 357ms | latency fixed, headroom used, but captures still under-margined; lite_greedy projection underestimates real opp |

Plus Phase 1a (replay-mine, `2498516`) and the analytical-depth
benchmark (`f2ed987`) that demonstrated mirror-v2-as-opp in
projection is computationally infeasible (1300-2200 ms/plan vs
12 ms/plan with lite_greedy).

**Next-session entry point (in order):**

1. Read `/root/.claude/plans/read-the-handover-do-abundant-quokka.md`
   — the v4 plan with full detail and a copy-pasteable session prompt.
2. **Phase A — Analytics verification suite (~150 LOC, do FIRST):**
   - Build `scripts/verify_analytics.py` + `tests/test_analytics.py`.
   - Five tests: projection-vs-reality (the killer), determinism,
     self-play balance, vs-random, capture-math unit tests.
   - Output `audit/2026-05-20-analytics-verification.md` with PASS/FAIL
     per check. **Block v4 build until all pass or known-bug
     annotated.**
3. **Phase B — `agents/trajectory_portfolio/main.py` (~250 LOC):**
   - `identify_winning_state(world)` — closed-form predicate.
   - `identify_target_portfolio(world)` — smallest planet set sufficient.
   - `portfolio_acquisition_plan(world, portfolio)` — backwards-from-goal
     capture sequencing.
   - `defense_actions(world, portfolio)` — preserve portfolio members.
   - NO `fast_sim.step` calls in agent. All decisions from closed-form.
4. Gates before A/B: analytics tests pass + DI1+G1 pass + bench
   p95 < 200 ms + zero turns > 1000 ms + self-play balanced.
5. A/B vs baseline at n=16. Gate: positive win rate (> v3.1's 0/32).
6. NO Kaggle submission without explicit PI sign-off.

**Falsified-or-dead this session:**
- Forward-projection joint-optimization with lite_greedy opp as a
  WINNING architecture: structurally limited because lite_greedy
  underestimates real-opp threat → our captures under-margin →
  bounce in real games.
- Mirror-v2-as-opp inside projection: benchmark says 130-216 seconds
  per turn — infeasible.
- Value-maximization without state-verification: 5 iterations of
  parameter tuning on top of unverified analytical primitives,
  zero wins. Verify first.

**Pointers added this session:**
- `audit/2026-05-19-replay-mine-pre-roi.md` — Phase 1a (a-e failure
  modes observed in live submissions).
- `audit/2026-05-19-analytical-depth-benchmark.md` — K=50 +
  lite_greedy is viable at ~60-80 plans/sec; mirror-v2 infeasible.
- `audit/replays/replay-mine-2026-05-19.{json,md}` — 56,842
  fleet bucket roll-up across 5 most-recent submissions.
- `scripts/benchmark_analytical_depth.py` — benchmark harness.
- `scripts/run_scenarios.py` — Phase 1b standalone scenario runner.
- `tests/scenarios/base.py` + `tests/scenarios/test_observed.py` —
  DI1 + G1 scenarios + Scenario ABC.
- `agents/trajectory_roi/main.py` — v3.1 reference (do NOT iterate
  on this in next session).
- `/root/.claude/plans/read-the-handover-do-abundant-quokka.md` —
  v4 plan + next-session copy-paste prompt.

**Discipline anchors live this session:**
- Rule 37: axis exhaustion. Trajectory_roi value-maximization axis
  saturated at 5 variants. Pivot to goal-directed planner.
- Rule 6: closed-form heuristics before heavy compute. v4 has NO
  rollouts in the agent decision path.
- Rule 19: this session documented the nulls (5 failed versions)
  rather than hiding them.
- Rule 38: DI1 + G1 remain the fix-verification rigs for v4.
- Rule 40: no hardcoded defense × 2 multiplier. The architecture
  itself (portfolio preservation as goal) makes defense natural.
- `audit/tournaments/202605190*.json` — 5 sweep run JSONs.
