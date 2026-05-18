# HANDOVER.md — next-session brief

> Last written: 2026-05-18 PM (late wrap) by
> `claude/audit-workflow-performance-btjeK`.
> **Submission 52784853 (PV off + bug #3/#4/#12) CURRENT μ=1083.1
> as of today's snapshot — under-performing by ~30μ vs prior
> floor. Note: Kaggle scores DRIFT continuously and DO NOT settle;
> every μ below is a snapshot, not a final value. Local A/B (81.2%)
> did NOT translate to ladder.** Next session: pivot to wasted-ships
> hold-feasibility filter + asymmetric opp model + tighter
> pre-submit gates.

## Live state

| Submission | μ (snapshot, drifts) | Status | Role |
|---|---:|---|---|
| **52784853** | **1083.1** | Active rolling pair | NEW FLOOR — PV off + clean math, under-performed |
| 52754310 | 1143.7 | Active rolling pair | Trajectory champion (kept) |
| 52766596 | 1113.4 | Evicted | Joint v3 (replaced by 52784853) |
| 52744856 | 1149.2 | Older | Composite_a2 hybrid |

Daily submission budget: 5/18 used **1** (52784853). 4 unused.
Rolling pair: pushing the next submission evicts the OLDER of
[52784853, 52754310] (= 52754310 — the 1143.7 champion). Be VERY
careful pushing — losing the 1143.7 floor would be catastrophic.

## Headline finding (today's session)

**Local A/B vs the prior bundle does NOT predict ladder μ for this
agent family.** The 81.2% n=32 win-rate vs `/tmp/baseline_hybrid
_bundle.py` translated to a 30μ LOSS on the ladder. The bundle is a
single sparring partner; the ladder's opponent distribution is
different. New friction tag
`local-ab-vs-ladder-calibration-miss-30mu` documents this; the fix is
to gate future submissions on a 3-opponent panel + 4P sub-panel,
not just the bundle A/B.

## What this session shipped

11 commits on `claude/audit-workflow-performance-btjeK`:

```
be7a3b8  state/current.md: submission 52784853 (PV-off + clean math fixes)
82df5b8  value_heads: disable PV term in production (n=96 A/B regressed 39.6%)
884423d  Day-9 PM wrap: bug #15 v2 + bug #14 option 5 both A/B-failed at 39.6%
8e60a6a  Bug #14 option 5 v2: idempotent policy + tick-0-only call
e7f94cf  Bug #14 fix: option 5 — reactive defense in candidate rollouts
384cd54  Bug #4 fix: proposer-side drain-frontier pre-cut
b285882  Bug #15 fix v2: PV-term-only (drop double-counting counterfactual)
5f22ea8  Bug #14 cheap-mirror attempt: NEGATIVE RESULT, toggle preserved off
333b884  value_heads: env-var kill-switches for bug #15 fix halves (diagnostic)
9eb882d  Bug #3 + #12 fix: symmetric reinforce sizing + widened multi-wave window
466fc98  Bug #15 fix: counterfactual capture credit + production-PV term
```

Production effect: −30μ floor (1113.4 → 1083.1). Bug #3 / #4 / #12
clean math fixes are unconditionally good; the regression came from
the bundle A/B's inability to predict ladder shift.

## Falsified-or-dead

- **Bug #15 v1** (PV + per-fleet credit): A/B-failed 40.6% n=64.
  Diagnosed as "double-counting" — wrong diagnosis.
- **Bug #15 v2** (PV only): A/B-failed 39.6% n=96. The PV term
  ITSELF over-credits (chooser calibration mismatch).
  `_COMPOSITE_PV_ENABLED=False` is now the production default;
  kill-switch retained for future PV recalibration work.
- **Bug #14 option 1** (cheap-mirror with lite_greedy for ME):
  regressed defense oracles. Toggle off.
- **Bug #14 option 5** (smart reactive defense): A/B-failed 39.6%
  n=96 — bug-#14-cures-PV hypothesis fully falsified. Toggle off.
- **Bundle-as-only-A/B-baseline**: insufficient for predicting
  ladder shift. Need panel + 4P + h2h-vs-current-floor.

## Next-session plan (audit/2026-05-18-next-session-plan-tiered.md)

Five tiers, ordered by ROI:

1. **Tier 1 — Bundling-tax cleanup.** Cheap mechanical. Make
   iteration faster. ≤ 30 min.
2. **Tier 2 — Hold-feasibility filter** (THE wasted-ships lever).
   PI observed live games where we send fleets from far to capture
   neutral planets adjacent to strong opp planets — opp counters
   cheaply from short range and recaptures. Encodable as 3
   synthetic oracles (write FIRST); fix is a proposer-side pre-cut
   sibling to `_source_survives_launch` (bug #4 drain-frontier).
   1-2 hours. **Highest expected ladder lift.**
3. **Tier 3 — Asymmetric opp model.** Replace `lite_greedy_policy`
   in the rollout with `top_tier_mirror_policy` (already exists at
   `lib/opp_model.py:92`) or a ME-targeted counter-policy. Forces
   our chooser to find ME-robust strategies. Pair with Tier 2.
4. **Tier 4 — Active-planets / coalition proposer.** PI's
   structural critique: every planet should pitch a candidate; far
   planets contribute ships to the highest-EV mission. Generalises
   the existing pair-joint path
   (`chooser_trajectory.py:score_candidate_v4_joint`) to N-way.
   Bigger redesign; commit only after Tiers 2 + 3 are ladder-
   validated.
5. **Tier 5 (parked) — PV recalibration.** Open question filed at
   `knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated
   -for-PV.md`.

Full plan at
`audit/2026-05-18-next-session-plan-tiered.md` — read FIRST next
session.

## Hard-won lessons

1. **A/B vs one bundle is not enough.** The bundle's local-vs-live
   gap is unpredictable. Add 3-opp panel + 4P sub-panel + h2h
   vs current floor before any submission push.
2. **Pre-submit calibration math doesn't add up.** Multiple recent
   submissions had +20pp local A/B → current −20 to −30μ vs floor.
   Recurring pattern; the warning in state/current.md is real and
   binding.
3. **Sanity oracles detect structural bugs but not calibration
   debt.** Bug #15's sanity oracle correctly identified "captures
   score Δ ≈ 0"; the fix structurally addressed it but the chooser's
   emit gate wasn't recalibrated → over-emission → ladder loss.
4. **Convergent failure across hypotheses is informative.** Three
   independent A/Bs (bug #15 v1 / v2 / +option 5) all at 39.6% told
   us the root cause was upstream of all three "fixes."
5. **PI's intuition about wasted captures is right and encodable.**
   The "we capture a neutral that opp easily takes back" pattern is
   the SAME shape as bug #4 (drain-frontier) but applied to the
   TARGET instead of the SOURCE. Synthetic oracles + pre-cut filter
   should land cleanly.

## How to start next session

1. **Read this file first.**
2. **Check origin/main for updates** — PI has prepared seed-set
   artifacts there for A/B testing across geometries. The seed
   sets should be picked up before running any A/B.
3. **Read** `audit/2026-05-18-next-session-plan-tiered.md` — the
   full tier-by-tier plan with file paths and reusable utilities.
4. **First action**: Tier 1 (bundling tax) is a quick win;
   then pivot to Tier 2 (hold-feasibility filter) — write the
   three synthetic oracles BEFORE the filter implementation.
5. **Submission gate**: do NOT push a new submission until ALL FOUR
   gates clear: oracles + bench + 3-opp panel + 4P sub-panel. The
   bundle A/B alone is insufficient (this session's −30μ lesson).
6. **Be careful with rolling-pair**: the older entry is 52754310
   (1143.7 — the champion). Losing it would be catastrophic.

## Pointers (new this session)

- `audit/2026-05-18-bug-catalog.md` — 15-bug catalog (yesterday's
  artifact, still current).
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — full postmortem of the three failed A/Bs (PV + option 5).
- `audit/2026-05-18-next-session-plan-tiered.md` — tier-by-tier
  plan for next session (Tier 1-5).
- `knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`
  — general lesson on value-head/chooser-gate calibration mismatch.
- `knowledge-base/flags/2026-05-18-pv-term-regression-shipped-as-default-on.md`
  — historical flag (PV now disabled by default).
- `knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated-for-PV.md`
  — investigation sketch for re-enabling PV.
- `tests/test_planner_oracles.py` — oracle suite (8 tests +
  3 xfails; conditional xfails keyed on
  `_value_heads._COMPOSITE_PV_ENABLED`).
- `tests/test_me_defensive_policy.py` — option-5 unit tests
  (dormant feature, idempotency contract pinned).
- `agents/baseline/proposer.py:355` — `_source_survives_launch`
  (the bug-#4 drain-frontier filter; PATTERN TO MIRROR for the
  Tier 2 hold-feasibility filter).
- `lib/opp_model.py:92` — `top_tier_mirror_policy` (the
  asymmetric-opp candidate for Tier 3).

## Rule reminders for next session

- **Rule 1**: submissions are PI-approved, single-shot, no retry
  loops.
- **Rule 12** (Orbit Wars caveat): rolling-last-2 = [52754310
  (1143.7), 52784853 (1083.1)]. Pushing 1 evicts the older
  (52754310 — the CHAMPION). Verify the new submission is ≥
  the champion BEFORE pushing.
- **Rule 32**: session-start git fetch — `git fetch origin && git
  log HEAD..origin/main && git diff HEAD..origin/main` BEFORE any
  new compute. PI has new seed-set artifacts on main.
- **Rule 38**: fix-verification reproduces failure state. Use the
  oracle suite as the regression harness.
- **Rule 40**: prefer modeling correctness over restriction tuning.
  Tier 2 (hold-feasibility) is modeling correctness. Tier 4 (active
  planets) is modeling correctness. Tier 5 (PV recalibration) is
  modeling correctness paired with chooser-gate retuning.
