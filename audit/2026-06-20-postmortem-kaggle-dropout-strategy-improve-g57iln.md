# Postmortem — 2026-06-20 kaggle-dropout-strategy-improve-g57iln

Decision-quality based (not outcome based). A clean session of negatives is a
good session if the decisions were sound given decision-time priors.

## What went wrong

- **Planned to ship deep-search on a sub-bar strength prior (Rule 45 bypass).**
  The session opened by planning to "harden + ship" depth-3 deep search on the
  strength of the prior session's "14/28 → 17/28 vs V2" claim. That claim was
  n=28 binary — *below* the Rule 45 n≥32 / Wilson-lo bar — yet it was used as the
  load-bearing premise for a ship plan. When the timing fix forced the rollout
  opponent to change (producer → lite_greedy), I finally measured strength at
  n=32 and the whole premise evaporated: depth-3 is **parity** with 2-ply even
  with the accurate opponent (Δmargin −0.062, p=1.00), and a **significant
  regression** with the wall-safe cheap opponent (−0.688, p=0.01). The plan's
  verification step caught it before any submit, so no LB cost — but the *plan*
  should have re-validated the n=28 claim at n≥32 **before** committing build
  effort, not as a side-effect of the timing change. Decision-quality: the
  hardening work (anytime guards) was partly wasted (~half a session) because the
  strength premise was never solid.

- **Mild over-autonomy before surfacing the strategic fork.** After the second
  clean negative (scorer-myopia refuted), the marginal/compounding picture was
  already visible; I launched a third investigation (step30→50 swing) before
  surfacing the "this is marginal, pick a direction" fork to the PI. The
  diagnostics were cheap and informative and the PI kept saying "go," so this is
  borderline, not clearly wrong — but the fork could have been raised one negative
  earlier.

- **Process frictions (minor, self-inflicted):** (a) launched background "waiter"
  loops whose own command line matched their `pgrep` pattern → infinite
  self-match, killed twice; (b) sized a diagnostic `timeout` at 540 s for a
  ~20-min job → killed mid-run, lost the result, relaunched; (c) AskUserQuestion
  called without the required `question` field (failed twice).

## What went right (decision-quality positives)

- **Never submitted during exploration.** Four ideas tested, all null/negative,
  zero submits — correct, since nothing cleared the bar and the concentration
  probe is still warming. The rolling pair was left intact.
- **Refused to spin the noise.** d3_prod looked 7/11 ahead at partial-n; I waited
  for n=32 (18/32, parity) instead of shipping the apparent lift. The sub-agent's
  "we lose the neutral race" spin was likewise overridden by the actual numbers
  (we grab *more* neutrals in losses).
- **Banked every negative** as a knowledge-base entry so the deep-search and
  scorer-objective lines are not re-walked.

## Frictions logged this session

No `audit/friction.md` one-liners were written during the session (work ran
through plan-mode investigation, not the normal WRAPUP flow). The three process
frictions above are recorded here in lieu.

## Promotion candidates (PI ratified: PENDING — PI said "wrap up", not ratified)

### [ ] .claude/skills/kaggle-comp/improvements.md — re-validate sub-bar priors before planning to ship

**Tag:** `revalidate-subbar-prior` (a strength claim established below the n≥32
Rule-45 bar must be re-measured at n≥32 *before* it becomes the basis of a build/
ship plan — not after sunk effort).

**Where to insert:** alongside the existing small-n-overconfidence guidance.

**What to add:** "When a prior session's lever looks promising on a sub-n-32
result, the FIRST action of any plan to ship it is to reproduce the lift at n≥32
(Rule 45). Do not invest build/hardening effort on a strength premise that never
met the bar — the deep-search line (2026-06-20) burned ~half a session hardening
depth-3 whose '17/28' premise was n=28 and collapsed to parity at n=32."

**Why:** ~half a session of hardening + eval effort spent on a premise that
evaporated; same small-n trap flagged twice this week.

## PI additions (from step 4)

PI responded "wrap up" to both the postmortem-additions prompt and the open
strategic-direction question — no additions; defer the direction call to next
session (recorded in HANDOVER).

## Framework version at session-end

- Commit SHA: (this commit)
- Active rules: CLAUDE.md Rules 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46.
- Loaded skills this session: postmortem (and kaggle-comp context).
