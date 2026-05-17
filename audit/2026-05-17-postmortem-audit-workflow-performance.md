# Postmortem — 2026-05-17 audit-workflow-performance-btjeK

## What went wrong

Three items worth flagging, all caught in-session without PI override:

1. **N_VALIDATE=60 cap was a bad first-cut wallclock fix.** Mirrored
   composite chooser's exact constant without thinking through the
   different per-candidate cost profile. Cost: ~10 min re-running the
   A/B + 1 commit's worth of churn (the 57.8% intermediate result is
   in commit message history but not a separate commit). Self-caught
   when the A/B came back at 37/64. Rule 40 was the right frame
   (restriction-tuning vs modeling-correctness) but I didn't apply it
   until AFTER seeing the winrate drop. Lesson worth filing: when
   copying a tuning constant across architectures, the per-unit cost
   model is usually different — verify, don't assume.

2. **2P-only A/B coverage for a default-on production change.** The
   trajectory chooser was tested at n=64 vs v15 only in 2P games. 4P
   ladder games are ~36% of live matchups. Flag filed at
   `knowledge-base/flags/2026-05-17-trajectory-default-4p-untested.md`.
   PI approved submission with full awareness; not a bypass. Worth
   noting that the A/B harness (`fast.py eval`) defaults to 2P and we
   never explicitly ran a 4P sub-panel. Pattern recurs.

3. **Pre-summary: misquoted the SUN_SAFETY=0 fix lift.** Conflated
   n=32 (filter-on) with n=64 (filter-off) when describing the
   "14pp false-reject rate" — at same-n the gap was ~3pp. Self-
   corrected within the same turn. No downstream impact (the fix
   shipped correctly at 29e0d27; PI didn't act on the bad number).

## Frictions logged this session

- `validate-cap-too-tight-cost-winrate-not-just-wallclock` (NEW) —
  mirrored composite's N_VALIDATE=60 + n_aff cap blindly; trajectory
  v4's per-candidate cost profile is shallower (prop_horizon clamps
  to MIN_HORIZON=25 vs composite's avg=32), so n_aff floored at 8
  on heavy turns, choking off ~95% of candidate breadth. Fix:
  N_VALIDATE=200 + rely on safe_deadline pre-bail as the real
  binder. Rule 40 applies.

## Promotion candidates

None this session. The N_VALIDATE friction is a specific application
of Rule 40 (already in CLAUDE.md). The 4P-untested flag is a
calibration warning, not a generalisable rule.

Quote from friction.md preamble: "New tags get one cycle of grace
before promotion. If a tag fires 3+ times, it goes to improvements.md."
N_VALIDATE fires once this session; grace applies.

## PI additions

Pending (will ask in next turn). Today's PI overrides: none. Today's
PI directives followed: submission of trajectory chooser as production
default, then Direction B for next session.

## Framework version at session-end

- Commit SHA: f192cf4 (about to commit wrap-up)
- Branch: `claude/audit-workflow-performance-btjeK`, 39 ahead of
  origin/main (40 after wrap-up commit)
- Active rules: 1..40 from `CLAUDE.md ## Operating rules — concise`
- Loaded skills this session: postmortem (this), kaggle-comp (Rules 12,
  22, 26 enforcement)

## Decision-quality scorecard

- **Decision to ship trajectory chooser as default** (vs hold for
  more 4P testing): GOOD given priors. Local A/B was 65.6% vs 62.5%
  composite at parity, +3pp Wlo edge, better wallclock profile,
  rolling-pair partner (composite_a2) is the safety floor at
  μ=1158.6. If 4P regresses, 3-line rollback is available.
- **Decision to fix wallclock first (Option A from AskUserQuestion)**:
  GOOD. The pre-fix max=2416ms would have confounded live μ. Cost was
  ~30 min; benefit is a cleaner signal.
- **Decision to ship N_VALIDATE=60 first (then revise)**: MIXED. The
  experiment was cheap (one A/B run, ~15 min), and the revised cap
  (N=200) is the correct answer. But I should have predicted the
  cap-bind issue before running the A/B — composite has N_VALIDATE=60
  because composite's per-candidate cost is HIGHER (so 60 is the
  binding constraint there). For trajectory v4 (cheaper per candidate),
  60 was over-conservative.
