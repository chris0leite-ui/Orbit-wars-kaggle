# Postmortem — 2026-05-23 strategy-framework-design-OyoYR

## What went wrong

- **Verified the capture-fix via a 16-game panel instead of a
  unit-test on chooser argmax.** Priors at decision-time: 5/19
  verdict said leaf-side axis was exhausted at N=2 variants;
  the fix (28ce9f3) had regression tests + a worked example showing
  +196 V_diff on capture frames; Rule 38 required fix-verification
  reproducing the failure state. Given those priors, "verify the
  fix is real" was the right goal but "use a panel to do it" was
  the wrong instrument. A 2-min unit-test asserting
  `_per_seat_in_flight_credit` shifts the chooser's argmax on a
  synthesised capture frame answers the same question at near-zero
  cost. The panel only answers downstream "does the argmax shift
  propagate to placements?", which had the same prior as the 5/19
  tie regardless of the fix.

- **Started the session on v15 foundation without checking
  team-best.** Priors at decision-time: `state/current.md` says v15
  is champion as of 5/17, BUT `git log HEAD..origin/main` (run in
  the session-start hook) contained the sibling-branch orbitfix
  submission commits. The information was visible and not
  processed. Result: 4 hours optimising on a foundation 46 μ below
  the team's best (sub 52912707 from
  `claude/extract-physics-trajectory-Vjaz9`, μ=1165.4 vs v15's
  μ≈1119.6).

- **PI-override:** "we need a large lift, so 8 games suffice."
  I proposed 32 seeds / ~3 hr without first asking PI what
  statistical resolution they actually needed. A better-calibrated
  opening would have offered the small panel from the start with
  the "large lift" frame and saved a round-trip.

- **Rule-bypass:** Rule 16 (6-question pre-flight) was skipped
  before launching the panel. Q1 ("already explored?") would have
  forced me to justify revisiting an exhausted axis and the panel
  might have been replaced by the unit-test path.

## Frictions logged this session

Both appended to `audit/friction.md` under
`## 2026-05-23 (claude/strategy-framework-design-OyoYR — leaf-side
rerun w/ capture fix)`:

- `tag: axis-reopened-with-new-fix` — re-tested an axis the same
  branch declared exhausted 4 days earlier because a modeling fix
  landed; Rule 37 has no corollary for fix-induced re-litigation.
- `tag: spot-check-too-thin-first-pass` — initial 1-seed × 1-seat
  spot-check was misleadingly weak (2 turn divergences); bumping
  to 4 seeds × 1 seat showed the true signal (80-120 turn
  divergences + 2 winner flips). One seed is a sampling artifact,
  not a signal.

## Promotion candidates (PI ratified: see per-candidate)

- **Candidate 1 — Rule 37 corollary on fix-induced
  re-litigation.** Status: **NOT promoted**. PI declined; the
  axis-reopened-with-new-fix friction tag stays in friction.md
  for one cycle of grace before re-evaluation.
- **Candidate 2 — improvements.md entry on spot-check minimum
  sample size.** Status: **NOT promoted**. Weak (fired once), gets
  the one-cycle grace per friction.md convention.
- **Candidate 3 — state schema + session-start hook update for
  `team_best_submission`.** Status: **PROMOTED**. Drafted at
  `.claude/skills/kaggle-comp/improvements.md` under
  `## Pending — promotion needed`. Two coordinated changes:
  (a) `team_best_submission` block in `state/current.md`,
  (b) session-start hook parses
  `kaggle competitions submissions <comp>` and surfaces team-best
  when it differs from `current_submitted_agent`.

## PI additions (from step 4)

None — PI said "Nothing to add, write it."

## Framework version at session-end

- Commit SHA: `49c02dd` (audit) + this postmortem commit pending.
- Active rules: CLAUDE.md `## Operating rules — concise` 1..40 (Rule
  37 axis-cap and Rule 38 fix-verification both binding this session;
  Rule 40 modeling-correctness-over-restriction-tuning binding on
  the capture-fix design).
- Loaded skills this session: `postmortem` (this artifact).
- Branch: `claude/strategy-framework-design-OyoYR` (ahead 12 / behind
  48 of origin/main as of session-end, after the upcoming postmortem
  commit).
