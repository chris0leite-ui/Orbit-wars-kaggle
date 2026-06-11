# Postmortem — 2026-06-12 elegant-dijkstra-uae6p0

Decision-quality-based review of the Producer-hunt session
(concentration rebuild → artifact disproof → revert to ledger_v1_4).

## What went wrong

- **Interpretation before measurement-verification.** The 8/10
  "fortress" battery result was load-bearing for half the session's
  work, while two warning signs were already in hand at the moment it
  appeared: the same seed flipping loss→win between solo and battery
  runs, and this campaign's prior dead-opponent scandal. Given those
  priors, hostile-testing the measurement (opponent compromised under
  load?) should have preceded doctrine-building. Nine freeze-gate
  encodings were engineered on the phantom; the disproof experiment,
  once run, took twenty minutes. This is the session's one clearly bad
  decision by decision-time standards, not hindsight.
- **Fourteen mechanisms committed as one block** with single-seed
  checks between. When the clean panel regressed (v7_0 5/12, bundle
  1/8 vs reference 8-9/12 and ~75-85%), per-mechanism attribution was
  impossible in-session. The whole block was reverted; genuinely good
  pieces (if any) are stranded in c42c9fc pending bisection.
- **One mid-battery agent edit** (reserve-release while the first
  battery ran). Caught immediately, batteries killed and restarted
  clean — the existing rule held, but the edit should not have
  happened.
- Mitigations that worked: Rule 1 kept everything off the ladder (no
  submissions); every headline claim was re-verified before any
  submission proposal; the artifact was disproven and documented the
  same session, with the audit correction at the top of the document
  it falsifies.
- PI overrides: none mid-session. Rule-bypass: none hard; the 8/10
  was reported to the PI at n=10 with contaminated methodology but
  with explicit caveats and verification queued — Rule 45's spirit
  bent, not broken. Rule-gap: total-launch liveness asserts cannot
  catch mid-game opponent degradation (now logged as friction with a
  drafted rule; PI declined promotion this session).

## Frictions logged this session

audit/friction.md `## 2026-06-12 (session 2: producer hunt)`:
`contention-degrades-torch-opponents`,
`wall-clock-budget-makes-strength-load-dependent`,
`mechanism-stack-without-bisection`, `crash-guard-hides-dead-agent`,
`snapshot-states-you-might-bisect`, `editing-agent-mid-battery`.

## Promotion candidates (PI ratified: no)

Three candidates were drafted and presented (A/B validity rule:
≤3 workers + torch thread caps + per-phase opponent liveness + solo
spot-checks; snapshot-before-battery; panel cadence of ≤2 mechanisms).
PI declined promotion — they remain as friction entries and as the
"NEW BINDING A/B RULE" note in HANDOVER.md for the next session to
apply informally.

## PI additions (from step 4)

None ("Nothing to add").

## Session decisions of record

- PI: **ledger_v1_4 will NOT be submitted.** Live pair unchanged
  (53556728 ledger_v1, 53558897 ledger_v1_2). 0/5 daily submissions
  used.
- Agent state at session end: agents/ledger/main.py byte-equal to
  submissions/ledger_v1_4.py, forecast parity green.
- Producer matchup closed as unresolved: 0 wins in ~45 honest games;
  the fortress win condition is disproven (see audit CORRECTION).

## Framework version at session-end

- Commit SHA: f8009fe (postmortem + wrap-up files commit follows)
- Active rules: CLAUDE.md operating rules 0, 1, 12, 32, 35, 36, 38,
  39, 40, 42, 45, 46 (strategy-lock era index)
- Loaded skills this session: postmortem
