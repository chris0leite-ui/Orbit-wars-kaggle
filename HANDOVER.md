# HANDOVER.md — next-session brief

> Last written: 2026-05-19 by `claude/ml-competition-strategy-PFhzM`
> (Phase 3 compound-weight sweep finalised + PI strategic pivot to
> clean ROI agent on kept architecture + observation-grounded
> scenario gate). See `## Day-19 PM ml-competition-strategy-PFhzM`
> below for the load-bearing section for next-session execution.
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

**Strategic state.** Bundle's chooser/scorer axis is fully
characterised and exhausted. Phase 3 compound-weight sweep
({0.05, 0.1, 0.2, 0.3, 0.5}) lifts bundle-vs-baseline from Wlo 0.035
to 0.142, saturating at 0.3. Lever real but well below the 0.55
gate. Bundle is structurally v7_0-class and cannot reach the live
champion via more scorer coefficients. PI ratified the architectural
pivot.

**Pivot in one sentence.** Drop bundle's decision stack entirely;
keep the `lib/*` primitives; rebuild a clean ROI agent at
`agents/trajectory_roi/` with 6 first-class primitives, and gate it
on observation-grounded synthetic scenarios that ROI must pass 100%
before any tournament A/B.

**Next-session entry point (in order):**

1. **Query live μ for sub 52744856.** Was PENDING at last session;
   determines whether the live champion (composite+A2 hybrid) moved
   off v15. Use the session-start hook's `kaggle competitions
   submissions orbit-wars`.
2. **Phase 1a — replay-mine recent live games.** `python
   scripts/replay_mine.py --recent 5` against the latest COMPLETE
   submission(s). Goal: confirm/refute PI's five named failure modes
   (a recapture, b drift, c garrison-counter, d split-majority,
   e distant-idleness) in actual live data. Document in
   `audit/2026-05-19-replay-mine-pre-roi.md`. Open question filed
   at `knowledge-base/questions/2026-05-19-do-failure-modes-c-and-e-
   appear-in-live.md` — close this with explicit observed-or-not
   per failure mode.
3. **Phase 1b — scenario substrate.** `tests/scenarios/base.py`
   with `Scenario` ABC (single-turn + multi-turn flavours), clean-
   state helpers (round-trip, ray-cast reachability). Reuse
   `tests/test_bundle_oracles.py` helpers (`_planet`, `_obs`, `_emit`).
4. **Phase 1c — author V0 scenarios** (8 total): S1-S3 (sanity), R1
   (a), D1 (b), G1 (c), SM1 (d — PI's canonical 100+100 vs 50),
   DI1 (e). Source-tagged to the replay finding from step 2.
5. **Phase 1d — clean-validate** each scenario (no sun-in-the-way,
   physically reachable, manual eyeball). Phase 1 commit gate:
   ≥6/8 fail against `baseline` (confirms suite encodes real gaps).
6. **Phase 2 — `agents/trajectory_roi/main.py`** with the 6-primitive
   architecture. Path-integration shape from `BundleEvaluator.score`
   reused (the one bundle insight worth carrying forward — see
   `knowledge-base/flags/2026-05-19-bundle-decision-stack-shelved-
   not-deleted.md`).
7. **Phase 3 — validate ROI 100%** against the scenario suite. Each
   FAILURE → extend a primitive (never a hotfix `if` patch). Iterate.
8. **Phase 4 — A/B vs current live champion** at n=8, expand on signal.
9. **No Kaggle push** unless predicted μ > live champion (Rule 12).

**Full plan**: `/root/.claude/plans/no-go-forward-test-fluttering-token.md`
(approved this session).

**What changed in `state/current.md`:** date 5/17→5/19, days_to_deadline
37→35, submissions_used_today 2→0, plateau_days 0→2,
saturation_count 0→1, session_log prepended with today's entry.

**Discipline anchors live this session:**
- Rule 37: axis exhaustion. We exited the chooser axis (overdue by 3
  variants — friction logged at
  `audit/friction.md::plan-doc-survives-strategic-redirect`).
- Rule 40: PI's "no hotfixes" is Rule 40 applied to the new ROI
  architecture. Every scenario fix extends one of 6 primitives.
- Rule 38: every scenario IS a fix-verification rig for its named
  failure mode.

**Falsified-or-dead this session:**
- Compound-weight as a competitive lever (vs baseline). Real but
  saturates at Wlo 0.142.
- Bundle as a submission candidate (it's v7_0-class, can't evict
  live champion without losing ladder spot).
- Speculative scenario authoring — PI rejected. Scenarios must be
  observation-grounded.

**Pointers added this session:**
- `audit/2026-05-19-phase-3-sweep-and-roi-pivot.md` — sweep results
  + pivot verdict + plan reference.
- `knowledge-base/thoughts/2026-05-19-roi-pivot-scenario-gated-
  clean-architecture.md` — the pivot in PI's words + the
  architectural reasoning + the anti-pattern I logged about myself.
- `knowledge-base/flags/2026-05-19-bundle-decision-stack-shelved-
  not-deleted.md` — what "drop bundle" means in working-tree terms.
- `knowledge-base/questions/2026-05-19-do-failure-modes-c-and-e-
  appear-in-live.md` — open question Phase 1a answers.
- `audit/tournaments/202605190*.json` — 5 sweep run JSONs.
