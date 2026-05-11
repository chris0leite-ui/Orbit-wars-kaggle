# Postmortem — 2026-05-10 evening, competition-strategy-brainstorm-ZK6XT

> Per `.claude/skills/postmortem/SKILL.md` step 5. Decision-quality-based:
> a good outcome from a bad decision is still a bad decision; a bad
> outcome from a good decision is fine. See
> `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`.

## What went wrong

**Nothing structurally.** The session ran as a clean
"hypothesis → A/B → measure → ship-or-revert" loop. The two
reversals (arrival_ledger in DEFAULT_MECHANISMS; v2 with ship-bumping)
were the experiment loop working as designed: try a plausible move
informed by Roman's code, measure it locally, revert when the A/B
regressed. Both were decisions worth taking ex-ante; the regression
signal was the load-bearing output.

Decision-quality breakdown:

- **Bad decisions:** None I'd retake differently given the priors
  available at decision-time.
- **PI overrides:** One — time estimates. My original strategic-
  direction plan budgeted v3 in 12-15 *days*. PI overrode to
  "tomorrow morning you will be done" (≈ 6 hours). Calibration data:
  my code-comp implementation estimates are ~5× too pessimistic.
- **Rule-bypass failures:** None.
- **Rule-gap failures:** Two — see promotion candidates.

## Frictions logged this session

Cross-references to `audit/friction.md` § 2026-05-10 (evening):

- `arrival-ledger-mechanism-without-planner-regresses` —
  audit/tournaments/20260510T215332Z.json (50% WR).
- `strategy-level-affordability-filter-prefers-low-roi` —
  audit/tournaments/20260510T215806Z.json (0/64 WR).
- `bundle-output-clobbers-prior-bundles` — recoverable via Kaggle
  state; the v1.2 bundle artefact at `submissions/roi.py` was
  silently overwritten by re-bundling `agents/simple/roi.py`.
- `handover-md-over-line-cap` — HANDOVER.md grew to 439 lines.
  Archived in this wrap-up to
  `audit/archive-2026-05-10-handover-prior-pm-sessions.md`.
- `time-estimates-too-pessimistic-by-5x` — PI-corrected.

## Promotion candidates (PI ratified: pending — PI replied
"nothing to add" on the postmortem text but did not separately
yes/no the promotion question; treat as NOT ratified until next
session)

### [ ] kaggle-comp/SKILL.md — pipeline-layer rule: filter in strategy, drop in mechanism

**Tag:** `strategy-vs-mechanism-filtering` (2 separate regressions in
one session)

**Where to insert:** under the strategy/mechanism guidance section.

**What to add:**
> Strategy `propose_intents` should NEVER filter out intents based on
> affordability or "would drop anyway" predictions. Emit the
> highest-value intent and let the mechanism layer (`validate`,
> `arrival_size`) handle bumping and dropping. Two regressions in one
> session (arrival_ledger in DEFAULT_MECHANISMS = 50% WR; v2
> affordability filter = 0% WR) confirm: filtering at strategy level
> starves high-value choices without enabling re-pick across the
> pipeline. The only valid strategy-level filter is "skip because
> WorldModel says the target is already ours" (idempotent dedup) —
> NOT "skip because we can't currently afford the bumped ships."

**Why:** audit/tournaments/20260510T215332Z.json (arrival_ledger),
20260510T215806Z.json (v2 bump). Both regressions cost ~3 min of
compute each + a revert decision. Cost is small but the pattern is
clearly generalisable across future strategy iterations.

### [ ] scripts/bundle_agent.py — versioned default output filename

**Tag:** `bundle-output-clobbers-prior-bundles`

**Where to insert:** `scripts/bundle_agent.py::bundle()` default
output path.

**What to add:**
> Default output path should include the agent's version / branch /
> SHA so re-bundling doesn't overwrite a frozen earlier bundle.
> Suggested format: `submissions/<agent-stem>-<git-short-sha>.py`
> or accept a `--version` flag (required, no silent default).
> Refuse to overwrite an existing file without `--force`.

**Why:** Bundling `agents/simple/roi.py` overwrote the v1.2 bundle.
Recoverable here because v1.2 is on Kaggle, but the next time we
need a local A/B against a frozen prior version, the artefact
will be silently missing.

## PI additions (from skill step 4)

PI replied "nothing to add" — no additions.

## Framework version at session-end

- Commit SHA: `b7e8b80` (before this wrap-up commit)
- Branch: `claude/competition-strategy-brainstorm-ZK6XT`
- Active rules: CLAUDE.md rules 1–36 (unchanged this session)
- Skills loaded this session: `postmortem`

## Calibration note

Predicted-vs-actual:

- **v1.2/roi μ prediction at submit:** "700–1000" (per the morning
  session's HANDOVER). Actual settled μ at session-end: **996.5**.
  Top of band. Good calibration.
- **v1.3 / v2 predicted live μ at submit:** "1050-1200" (this
  postmortem's branch). Actual: PENDING — submission #52532938.
  Read at next session start.

Rule 26 device: BOTE / prediction-before-experiment was applied for
each A/B. Two A/Bs landed in the regression band (arrival_ledger 50%,
v2 bump 0%) where we predicted 60-70% — those were honest
"hypothesis was wrong" outcomes. The third (v2 simplified) hit
64% panel WR vs a 60-70% prediction band — squarely in.

## Session lift summary (outcome, separate from decision-quality)

Deliverables: capture-success probe, public-kernel teardown (4
notebooks), Block A physics-module upgrade, Block C arrival-ledger
substrate, Block D v2 strategy, Block E1 mission-framework plan
(NOT executed; queued for next session). 181 tests green. v2
submitted as #52532938. Two staged bundles: v1_3_roi_physics.py
and v2.py.
