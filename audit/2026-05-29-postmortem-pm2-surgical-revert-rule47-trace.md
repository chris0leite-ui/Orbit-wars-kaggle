# Postmortem — 2026-05-29 PM2 (kaggle-submission-review-gZsCu, surgical revert + Rule 47 trace)

Branch: `claude/kaggle-submission-review-gZsCu`
Session window: ~18:00 — ~20:40 UTC (resumed mid-session from earlier window).

## Session 1-liner

Surgical revert (commit `2224324`) shipped as sub **53163774**; verification A/B at-parity with the frozen PV_ETA anchor (15/30 at n=30 — killed at seed 30 by session resume); Rule 47 trace cleanly diagnosed physics-OK + **49.1% idle-with-ships** as the chooser-pricing root cause for the wasting-fleets / underutilized-fleets symptoms PI surfaced.

## What went wrong

**Bad decisions.** None I would retake differently given priors. The surgical revert was on PI's crisp diagnosis (*"you were not supposed to remove the patience, only the [ledger]. Waiting needs to be an option, of course."*). Pre-submit gates (Rule 42 + 46) ran cleanly. Picking Plan A (instrument first) over Plan C (chooser-fix-from-guess) was the right call — it produced a decisive answer in <5 min CPU and reordered our priority stack.

**PI overrides (calibration data).**

- PI explicit override of Rule 45 (n=32 Wilson-lo ≥ 0.50; had n=30 lo=0.33) AND Rule 43 (multi-opponent panel; not run) to push sub 53163774. Rationale: *"submit, so we validate against multiple opponents online."* Calibration data: PI treats the Kaggle ladder as the de facto multi-opponent panel when (a) local A/B says at-parity-with-anchor AND (b) the local panel is too slow to run inside session budget.

**Rule-bypass failures.** None this session. Pre-submit gates were all run; claim row was written before the push.

**Rule-gap failures.**

1. The trace harness I wrote (`/tmp/trace_rule47.py`) produced a decisive answer (physics-waste 0%, idle-with-ships 49.1%) but lives in /tmp — will be gone next session if the container reclaims. PI declined to promote this session, but it's worth re-surfacing if it survives.
2. Rule 46c (fast.py play smoke) only checks "runs without crash" — does NOT gate on max-turn-ms against the 1000ms env cap. Today's bundle hit max=782ms (safe), but a bundle that creeps to >1000ms would pass Rule 46c and silently DQ in live. PI declined to promote this session.

## Frictions logged this session

- `audit/friction.md::2026-05-29 PM2::rule-47-trace-tool-lived-in-tmp`
- `audit/friction.md::2026-05-29 PM2::rule-46c-no-max-turn-ms-gate`

## Promotion candidates (PI ratified: no)

Two candidates surfaced, both declined for this session:

- **A — check in `/tmp/trace_rule47.py` to `scripts/`.** Tool, not a rule. Decisively useful (<5 min CPU, reordered next-session priority). PI declined; deferred to next session if the trace survives the container reclaim.
- **B — tighten Rule 46c to gate `max-turn-ms ≤ 900`.** 10% safety margin to the 1000ms env cap. PI declined; promotion candidate remains valid for a future wrap-up. Surface again if a future bundle's fast.py play shows max-turn ≥ 900.

## PI additions (from step 4)

PI: "nothing to add or promote, push."

## Framework version at session-end

- Commit SHA: `0a1c053`
- Active rules: 1..48 (CLAUDE.md top-level rules; Rule 48 most-recent — "same-day Kaggle ladder readings are climb snapshots, not verdicts")
- Loaded skills this session: `postmortem`

## Session arc (for next-session pickup)

1. Resumed mid-A/B (over-strip revert vs anchor, n=15 at resume). The original ab_noswap process was killed by the previous session's resume — relaunched, ran to 30 seeds, killed again at seed 30 by THIS session's resume. Verdict: **15/30 = 50%, Wilson 95% [0.33, 0.67]**, at parity with frozen anchor.
2. Pre-submit gates: Rule 42a (kaggle rolling-pair state read), Rule 42b (claim row appended to `state/MULTI_BRANCH.md`), Rule 42c (evicted-μ 1080.7 < predicted-mode 1130 → GREEN). Rule 46a (bundler --force byte-parity), 46b (test_bundle.py 10/10), 46c (fast.py play seed=7 p0_win 261 steps max=782ms).
3. Submitted sub **53163774** at 19:50 UTC. Rolling pair after: [53163774 NEW, 53131296 baseline_validated μ=1114.1]. Evicted 53117942 (μ=1080.7) as predicted.
4. PI: "think hard how to improve our strategy." 5 symptoms surfaced: wasting fleets, slow-far fleets, weak opening, no streamlined attacks, underutilized fleets-on-planets. PI proposed the test "replace light-greedy with nearest." I gave a ranked 3-move plan (A=instrument, B=panel-swap, C=symmetric-PV-no-action-leaf).
5. PI: "do A." Wrote `/tmp/trace_rule47.py`. One-game trace, revert vs frozen anchor, seed=7. Results below.

## Headline finding (knowledge-base entry separate)

```
PHYSICS WASTE
  Total fleets launched:  300
  Sun / OOB / timeout:    0  (0.0%)        ← CLEAN

NO-ACTION TURNS
  Turns emitting nothing:                139/265  (52.5%)
  Of those, with launch-able surplus:    130/265  (49.1%)  ← HUGE
```

predict_fleet_fate is effective everywhere it's called. The chooser is mispricing "do nothing" relative to "launch this candidate" in HALF of our turns when we have ships ready. This is a Rule 40 modeling-correctness call (symmetric PV on both sides, not a MIN_DELTA_TO_LAUNCH threshold bump).

Next-session priority reorder:
- **First** — split the 49% by step quartile to see if it's opening-heavy, mid-game-heavy, or uniform. Decides scope of the chooser fix.
- **Second** — symmetric PV on the no-action leaf (was Plan C — promoted to next-action by this trace).
- **Third** — Plan B (nearest-panel A/B) drops in priority; both anchor and our agent share the over-patience pathology so the anchor A/B can't see it.
