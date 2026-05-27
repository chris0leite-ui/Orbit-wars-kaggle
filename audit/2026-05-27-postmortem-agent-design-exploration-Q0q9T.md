# Postmortem — 2026-05-27 agent-design-exploration-Q0q9T

## What went wrong

Nothing flagged this session — the experiment was designed honestly,
executed cleanly, and the falsification produced a generalizable
lesson. The spearhead axis was closed at triage cost (~75 min A/B +
~3 hours implementation), which is well below the budget for closing
an axis cleanly. Cell D's 40 pp regression vs reference was unexpected
but the priors at design-time did not include the "favor leaf already
encodes direction implicitly" reasoning — that emerged from
interpreting the result, not from anything we should have known up
front. Not a bad decision.

## Frictions logged this session

Two entries appended to `audit/friction.md` under
`## 2026-05-27 (claude/agent-design-exploration-Q0q9T — spearhead A/B triage)`:

1. `chooser-bonus-double-counts-favor-direction` — chooser-side
   directional bonus (Cell D / E in the spearhead matrix) regressed
   40 pp / 60 pp below reference because the favor leaf already
   encodes direction implicitly via per-step ownership accrual.
   Generalizable rule candidate: audit a score's implicit dimensions
   before adding explicit bonuses for them.
2. `ab-quick-runtime-250-step-not-natural-win` — ab_quick.py per-
   cell runtime estimate (~4 min) was 3× under actual (13-15 min)
   because games against strong opps run to the 250-step cap, not
   the typical natural-end. Minor; one-off.

Cross-links:
- `audit/2026-05-27-spearhead-ab-matrix.md` — the 5-cell A/B
  results and interpretation.
- `/root/.claude/plans/do-it-but-do-bubbly-teacup.md` — the plan
  (spearhead directional rule) executed this session.
- Commits `85871cd` (implementation), `605d446` (audit),
  `de09238` (friction one-liners).

## Promotion candidates (PI ratified)

PI on 2026-05-27 wrap-up: "Nothing to add and nothing to promote."
No entries promoted to `.claude/skills/kaggle-comp/improvements.md`
this session. The chooser-bonus friction stays in `audit/friction.md`
under its own tag; promote on 3rd recurrence per the standing
discipline.

## PI additions

PI declined to add to the postmortem.

## Calibration snapshot (Rule 26)

No submissions this session. No new calibration row.

Sub-tracking from prior sessions (live μ values are transient per
PI directive 2026-05-27; query Kaggle live for current):
- Sub 53082192 (baseline relay v2 / defensive modeling) — reverted in
  prior session; rolled off Kaggle's rolling pair after sibling
  branch pushes.
- Sub 53067354 (baseline relay v1) — rolled off.
- Sub 53065150 (buildup_planner, no relay) — branch peak, rolled
  off; query live for current settled μ.

PI override count this session: 0 (mid-session clarifications were
design inputs, not corrections to in-flight work).

## Session output summary

- New module: `agents/baseline/spearhead.py` (~70 LOC,
  `SpearheadContext` + `build_spearhead_context` + `cos_alignment`).
- Edits: `agents/baseline/relay_forward.py`,
  `agents/baseline/chooser_trajectory.py`,
  `agents/baseline/main.py` — all env-gated, default OFF, behavior
  preserved when flags are off (verified via 128-step smoke parity).
- Audit: `audit/2026-05-27-spearhead-ab-matrix.md`.
- Friction: `audit/friction.md` + 2 entries.
- This postmortem.

## Next-session direction (carried forward via HANDOVER)

Discussed but not implemented this session:

1. **Aggression recalibration** — single-knob A/B sweep on
   `SAFETY_MARGIN` in `agents/baseline/proposer.py:641`
   (`_target_holdable_after_capture`). Replay EDA points to top-10
   garrison-at-launch ≈ 7.7 vs midpack ≈ 22; the 1.5× post-capture
   safety margin is the most likely candidate-acceptance-gate knob
   making us over-defensive. Single-axis, ~1 hour for the 4-cell
   sweep. Rule-40-aligned (modeling fix, not band-aid).
2. **Batched-chooser port** — `lib/game/batch_interpreter.py` exists
   (633 LOC, byte-exact parity vs scalar interpreter, tested) but
   no agent uses it. Porting `chooser_trajectory.choose_trajectory`
   to call it would broaden candidate evaluation per turn at the
   same wallclock. Higher-leverage / longer-effort move.
3. **Replay-mined opponent model** — replace `lib/opp_model.
   lite_greedy_policy` (heuristic) with a behavior-cloned model
   fit from the existing top-10 replay corpus in
   `audit/external/replays/`. Multi-session investment; addresses
   the saturation root cause directly (chooser optimizes against
   a fake opp). Highest EV-ceiling of the three.

PI signal at end of session: "we want to get on top," "simple moves
matter," "no rush on recovery." That argues for direction 1 next
session, with direction 3 queued behind it.

## Framework version at session-end

- Commit SHA: `de09238` (friction one-liners) — postmortem commit will
  add this artifact on top.
- Active rules: CLAUDE.md rules 1-48 (Rule 48 added 2026-05-24:
  Kaggle initial μ readings are not settled).
- Loaded skills this session: `postmortem` (this run); `Plan`
  agent invoked once during plan mode for the spearhead design;
  `Explore` agents invoked twice during plan mode and once during
  aggression-recalibration discussion.
