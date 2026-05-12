# Postmortem — 2026-05-12 game-theory-strategy-analysis-0oH4N

## What went wrong

- **Terminated cannot-lose analysis prematurely** at "v3 IS the
  floor" after seeing 81% v3-vs-v3 draws in the first 16-seed probe.
  PI override ("this is kind of useless insight") was load-bearing —
  surfaced that the 19% non-draws were FIXABLE σ-equiv bugs in v3's
  tie-break path, not inherent v3 weakness. Three surgical patches
  (commits 6c12b9f, 7b60938, 24bae06) closed the gap to 100%.
  Cost: 1 PI correction; benefit: the actual cannot-lose realization.
  Friction tag: `useless-tautology-framing`.

- **Built v5_psp with ROI as Sim<K> rollout policy** without
  re-reading the audit
  (`audit/2026-05-11-lookahead-phase2-forward-sim.md:137-141`) that
  explicitly warned about policy-mismatch. Result: PSP scored at
  12.5% W/D vs v3 (worse than v4_endgame's 37.5%) because the ROI
  rollout systematically under-scored v3-style aggressive openings.
  Cost: ~one iteration's work (~3hr). Avoidable.

- **6 falsified mirror-overlay iterations** (Tier 0-2 mirror, hybrid,
  v4_endgame, v5_psp, v6_steady) before discovering the meta-pattern.
  Each iteration was reasonable given priors but the meta-pattern
  "cannot-lose is intrinsic to near-Nash, not an addable layer"
  emerged only from empirical falsification. A pre-iteration
  shock-absorber (literature-style: "have we tried checking the
  property as intrinsic to the agent first?") would have caught it.
  Friction tag: `structural-overlay-frame-error`.

- **Trusted stale state/current.md** for rolling-last-2 (claimed
  precision μ=984.6; actual μ=1009.0). Caught before submission
  via `kaggle competitions submissions` API check, but the
  initial strategic framing was based on the stale number.
  Friction tag: `stale-current-md-rolling-last-2`.

- **Replay-parity test failing as expected** on submit, blocking
  pytest green. Behavior-changing patches inherently break frozen-
  bundle tests; need a workflow to update the fixture in the same
  commit-or-flag-as-expected. Friction tag:
  `replay-parity-test-expected-fail`.

## Frictions logged this session

`audit/friction.md` under `## 2026-05-12`:

- `useless-tautology-framing` — terminated analysis at apparent
  resolution; PI override revealed residual was the lever
- `structural-overlay-frame-error` — six falsified iterations from
  treating cannot-lose as additive overlay
- `stale-current-md-rolling-last-2` — state notes lag live ladder;
  verify via Kaggle API before submit decisions
- `replay-parity-test-expected-fail` — intentional behavior changes
  must update frozen-bundle fixture
- `stop-hook-cant-commit-gitignored-bundle` — include bundle sha256
  in commit messages since bundle itself is gitignored

## Promotion candidates (PI ratification PENDING)

These are drafted for `.claude/skills/kaggle-comp/improvements.md`.
NOT committed there until PI signs off.

### [ ] kaggle-comp/SKILL.md — look-at-the-residual rule

**Tag:** `look-at-residual-before-declaring-done`

**Where to insert:** under "Diagnostic discipline" section (or
similar). New rule numbered after Rule 7 (research before saturation).

**What to add:**

> **Rule (proposed): Look at the residual before declaring done.**
> When an analysis terminates in "current code already solves
> it" or "the property is already there at level X%," look hard
> at the X% (or 100-X%) that doesn't fit the claim. The residual
> is usually the actual lever. Today: v3-vs-v3 at 81% draws was
> declared "v3 IS the floor"; the 19% residual turned out to be
> three FIXABLE σ-equivariance bugs in v3's tie-break path.

**Why:** Cost = 1 PI override that PRODUCED the session's headline
result (100% v3-vs-v3 draws via 3 surgical patches). Friction
entry `useless-tautology-framing`.

### [ ] kaggle-comp/SKILL.md — intrinsic-vs-overlay framing rule

**Tag:** `intrinsic-vs-overlay-frame`

**Where to insert:** companion to the look-at-residual rule.

**What to add:**

> **Rule (proposed): For game-theoretic claims, check intrinsic
> properties before building overlays.** If the target property is
> CANNOT-LOSE, NASH-EQUIVARIANT, GROUP-INVARIANT, or similar
> mathematical structure, first verify whether the property is
> already approximately TRUE of the existing agent (by direct
> measurement) before constructing structural overlays on top.
> Today: 6 falsified iterations built mirror-overlay constructions
> on top of v3 before discovering the cannot-lose property is
> intrinsic to v3 itself once tie-break asymmetries are eliminated.

**Why:** Cost = 6 iterations (~24h+) before iter 7 found the
correct frame. Friction entry `structural-overlay-frame-error`.

### [ ] kaggle-comp/SKILL.md — pre-submit ladder API verification

**Tag:** `verify-rolling-last-2-via-api`

**Where to insert:** pre-submit checklist (companion to Rule 27
about head-to-head diffs).

**What to add:**

> **Pre-submit verification (proposed addendum to Rule 27): Run
> `kaggle competitions submissions <slug>` immediately before any
> submission to confirm rolling-last-2 state.** state/current.md
> can lag the live ladder by hours; the rolling-last-2 you'll
> evict may differ from the recorded one. Materially changes
> slot-cost calculus.

**Why:** Today: state/current.md said precision μ=984.6 (oldest in
rolling-last-2); actual μ=1009.0. Friction entry
`stale-current-md-rolling-last-2`.

### [ ] WRAPUP.md or kaggle-comp — frozen-bundle workflow

**Tag:** `frozen-bundle-after-intentional-change`

**Where to insert:** WRAPUP.md section A as new step 6.5 (post-commit
pre-push), OR new rule in kaggle-comp SKILL.md.

**What to add:**

> **When a submission ships intentionally-different agent behavior:**
> the next post-submission commit MUST either (a) rebuild the
> frozen-bundle fixture from the new submission's bundle, or (b)
> tag the failing replay-parity test with an `@expected_failure_until`
> marker citing the new submission ID. Don't leave the test as a
> silent red light.

**Why:** Today: test_v3_snipe_frozen_bundle_replay_parity_100pct
failed at 94.96% match; pytest exit code 1 hides genuine future
regressions. Friction entry `replay-parity-test-expected-fail`.

## PI additions (from step 4)

PI was asked: "Anything you'd add to the postmortem? Frictions I
missed, rules you want extracted, decisions worth flagging?"

PI reply: PENDING at session-end. Stop-hook forced commit before
PI's response could be incorporated. Promotions to
`improvements.md` are NOT performed; postponed to next session's
opening or via a follow-up commit.

## Framework version at session-end

- Commit SHA: defabfa (Submit σ-equivariance v1 — first ladder push)
- Branch: claude/game-theory-strategy-analysis-0oH4N (22+ commits
  ahead of origin/main pre-merge; post-merge synchronized with v3.4)
- Active rules: CLAUDE.md rules 1-36 (the 36 operating rules
  documented in CLAUDE.md). 5 rules tagged [TABULAR-ONLY] (3, 24,
  25, 27 partial, 33) are inactive for code-comp.
- Loaded skills this session: kaggle-comp, postmortem, claude-api
  (passively available)
- Submission this session: #52565034 (σ-equivariance v1), PENDING

## Calibration snapshot

Predicted vs actual ladder μ for the σ-equiv v1 submission is
PENDING (submitted 04:39 UTC; first μ datapoint typically ~24h).
Predictions documented in `audit/2026-05-12-sigma-equiv-v1-submission.md`:

| Prediction | Threshold | Outcome |
|---|---|---|
| Validation Error | <5% chance | PENDING |
| μ < 995 | "σ-equiv hurts" | PENDING |
| μ 1000-1015 | "expected μ-floor" | PENDING |
| μ > 1020 | "lock more impactful than predicted" | PENDING |

Next session's first action: pull live μ, score predictions, update
calibration ladder in state/.
