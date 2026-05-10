# Postmortem — 2026-05-10 review-competition-handover-0pGNc

Branch: `claude/review-competition-handover-0pGNc`. Second session of
2026-05-10 (after `bootstrap-irewT` wrapped earlier the same day).
Decision-quality review per `.claude/skills/postmortem/SKILL.md`.

## What went wrong

- **Bundler-test default-list duplication (~30 min recover).** The
  bundler self-test referenced an inline-duplicated module list rather
  than reading `DEFAULT_INCLUDE` from the bundler itself. When new
  `lib/` modules were added during Step 3.5, the test continued to
  pass while the produced bundle silently `NameError`-d at import in
  the local tournament fixture. Caught on first ablation run, not on
  the bundle gate. Drafted as a promotion candidate
  (`bundler-default-lists-referenced-by-tests`); **PI declined**.
- **Mechanism-strategy coupling discovered late.** `comet_aim` and
  `sun_avoid` both showed negative ablation lift against the v1
  greedy strategy and were EXCLUDED from `DEFAULT`. Working theory
  (un-validated): both mechanisms are net-positive only against a
  more production-aware v2 strategy that respects sun-pull arrival
  arcs. Solo-mechanism ablation against the wrong strategy can mark
  a useful mechanism dead. Drafted as a promotion candidate
  (`mechanism-contribution-is-strategy-coupled`); **PI declined**.
- **Phase-1 research skipped for N-instances brief.** When the
  competition brief lists "many" / "a series of" / "standard X for
  every Y" the reflexive move is to dispatch the research subagent
  before the planner. I went straight to plan-mode, hypothesised two
  mechanisms (comet_aim, sun_avoid) on first principles, and PI had
  to override with "research these rules" — which then refuted both.
  Drafted as a promotion candidate
  (`phase-1-research-mandatory-for-n-instances`); **PI declined**.

PI's "no promotions" verdict is itself a calibration data-point: the
framework is judged to already cover these via Rules 7 (research
before saturation), 16 (6Q pre-flight, esp. Q6), and 21 (≥3 variants
before falsification). Adding ad-hoc rules-of-thumb alongside the
existing top-level rules increases noise more than it adds signal.

## Frictions logged this session

`audit/friction.md` was not appended for this session — WRAPUP
section A step 4 was cut short by PI's directive to stop the
question tool. The three drafts above (under "What went wrong")
stand as the would-be entries; carry to next session's wrap-up if
they recur.

No additional frictions surfaced during the session that aren't
already covered by `2026-05-10 (day-1 agent — bootstrap branch)`
entries in `audit/friction.md`.

## Promotion candidates (PI ratified: no)

Three drafts were presented and **all rejected by PI** in the
postmortem step-4 ratification:

1. `bundler-default-lists-referenced-by-tests` — REJECTED.
2. `mechanism-contribution-is-strategy-coupled` — REJECTED.
3. `phase-1-research-mandatory-for-n-instances` — REJECTED.

`.claude/skills/kaggle-comp/improvements.md` is **unchanged** by
this postmortem.

## PI additions (from step 4)

- "Nothing to add — proceed" on the postmortem-flags question.
- "no, stop the question tool" on the promotion question.

Recorded verbatim; no further interactive prompts issued.

## Framework version at session-end

- **Branch:** `claude/review-competition-handover-0pGNc`.
- **Commit SHA at postmortem write:** `dbcbed3` (HEAD; this artifact
  staged on top of it for the wrap-up commit).
- **Active rules:** CLAUDE.md Rules 1–36 (Rules 3, 24, 25, 27, 33
  carry the `[TABULAR-ONLY]` tag and do not apply to Orbit Wars).
- **Loaded skills this session:** `postmortem` (this run);
  `kaggle-comp` referenced in process docs but not invoked as a
  Skill.
- **Submissions today:** 3/5 (baseline 52497828 → v1 52507539 μ=508.1
  → v1.1 52509319 PENDING). Rolling-last-2 = [v1, v1.1]; baseline
  evicted per Rule 12 caveat.
- **Calibration table:** v1 actual μ=508.1 vs predicted (TBD —
  prediction was not pre-registered before submission, friction
  candidate for next wrap-up); v1.1 PENDING.
- **Tests:** 111 green at HEAD (54 D.1 fixture + lib/agents/bundler).
- **Mechanism set in DEFAULT:** validate, arrival_size, lead_aim.
  EXCLUDED on negative ablation: comet_aim, sun_avoid.
