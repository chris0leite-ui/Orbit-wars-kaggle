# Postmortem — 2026-06-03 (teamwork-seam reversal + refine submit)

Branch: `claude/kaggle-submission-strategy-JzIAr`

## What happened (one paragraph)

Re-opened the joint-coordination axis that was closed 2026-06-02 as "teamwork
doesn't arise." Identified the closure as a weak-opponent confound (the
0-coalitions measurement was only ever taken vs v7_0/v7_minimax). Re-measured the
generator on champion-vs-strong boards (~100+ coalitions/game), then A/B'd the
augment refiner on the real adaptive-K champion config: 70.2% h2h (n=57, Wilson-lo
0.573), paired +13/−4/+9 net. Submitted as `53336920` (PI explicit override of the
incomplete Rule-43 panel).

## What went well

- **Confound instinct paid off.** Suspecting the closure rested on a weak opponent
  (Rule 41) and testing the generator directly on strong-opponent boards was the
  whole unlock.
- **Measured the generator directly** once `env.run` was found to sandbox agent
  stderr — avoided a false null.
- **Matched-seed paired A/B + a 50% parity anchor** made the lift unambiguous and
  caught the two confounds below.

## What went wrong (→ friction.md)

1. **A/B on a non-champion base** (adaptive-K OFF) — nearly concluded "no lift" on
   a stripped-down agent. PI caught it. (`ab-on-non-champion-base`)
2. **Misread an env default as OFF** — claimed the kinematic table was off when it
   defaults ON. Gave the PI a wrong answer for a turn. (`env-default-misread-as-off`)
3. **Mid-run win-rate over-read** — called the static run a "wash" at 58%; it
   finished 68.8%. (Covered by Rule 45.)
4. **`/tmp` artifacts + the panel lost to a container restart.** (`tmp-artifacts-lost-on-restart`)

## Promotion candidates (for PI ratification → kaggle-comp/improvements.md)

- **"A/B on the live champion config, not the repo default; confirm a ~50% parity
  anchor before trusting any lift."** This burned ~an hour and a wrong conclusion
  this session; it generalizes to every lever A/B. Strong candidate.
- **"Re-test closed/null findings against the CURRENT opponent field before
  trusting them"** — a Rule-41 corollary; the closure was a stale weak-opponent
  artifact. Candidate.

(Not auto-promoting — flagged here for PI ratification next session per the
postmortem skill.)

## Result

Reversed a closed track, shipped the first coordination-using agent, and left a
localized highest-leverage follow-up (the 4-game regression tail). Net positive
session; outcome pending live μ.
