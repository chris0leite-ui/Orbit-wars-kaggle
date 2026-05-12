# Postmortem — 2026-05-12 PM fix-early-game-strategy-YSClQ

## What went wrong

**Bad decisions** (would not retake given same priors):

- **Spent ~2 h on three opener variants (A/B/C) before recognising the
  duplication.** v3.5.1's existing `aggressive=True` path already does
  the bowwowforeach trick at step 2-3 once `src.ships > 12`; a separate
  opener Mission class adds the same logic at a different layer. The
  v3.5 stack ablation had already shown `opening_only` regresses at
  40.6 % Wilson lo — that prior should have been more load-bearing.
- **Reported small-sample `8/8 (100 %)` as PASS** before the 32-seed
  confirmation collapsed it to 56.2 % / 44.1 % Wilson lo. The framing
  was technically correct on the lower bound at n=8 but misleading
  about confidence. Sample size belongs in every verdict line.
- **Did not pull live Kaggle μ at session start.** Iterated on the
  opener for an hour-plus thinking v3.5.1 was our strong baseline,
  while it had actually regressed to μ=943.1 (62μ below v3_snipe).
  Without PI's "look at kaggle for recent scores" nudge, my submission
  proposal would have been calibrated to the wrong baseline.
- **Retried `--workers 4` after the first silent crash** instead of
  switching to `--workers 2` or foreground immediately. Lost ~25 min
  on the second crash before diagnosing the pattern.

**PI overrides** (calibration data-points):

- "look at kaggle for recent scores" — corrected a missing
  baseline-recalibration step. Promoted to friction.
- "test all" — when I proposed only variant A, PI corrected to test
  all three opener variants + variant D. Variant D ended up the only
  deliverable that succeeded.
- "let's target to improve v7" — pivoted from v3 to v7 lineage. Before
  that I was implicitly assuming v3.5.1 was our best agent.
- "go with a" — branch decision (merge game-theory into current).
  No friction; the merge was straightforward.

**Rule-bypass failures:** none flagged.

**Rule-gap failures:**

- No rule says "pull live Kaggle μ at session start before iterating."
  The rolling-last-2 + ladder-μ-drift means yesterday's local results
  may be obsolete by today's session.
- No rule says "Wilson lo at n < 32 is screening, not verdict."
- No rule about bash redirect masking silent python crashes in
  background.

## Frictions logged this session

See `audit/friction.md` `## 2026-05-12 (PM — fix-early-game-strategy-YSClQ)`:

- `tag: local-ab-doesnt-transfer-to-ladder` — v3.5.1 68.8 % locally →
  μ=943.1 live (62μ regression).
- `tag: small-sample-wilson-lo-misleading` — 8/8 v7.1 wins at n=4 →
  56.2 % / 44.1 % at n=32.
- `tag: 32-seed-w4-silent-crash` — two consecutive 32-seed --workers 4
  A/Bs exited with empty log files after 23-24 min.
- `tag: empty-redirect-log-on-bg-failure` — `python ... > log.log 2>&1 &`
  doesn't always flush on abnormal exit.
- `tag: opening-mission-class-doesnt-help-v7-family` — three variants
  all FAIL 55 % gate.

## Promotion candidates (PI ratified)

**Promoted to `.claude/skills/kaggle-comp/improvements.md`:**

- ✅ (1) `live-mu-pull-at-session-start`
- ✅ (3) `diverse-panel-before-submit` (renamed
  `local-ab-doesnt-transfer-to-ladder` in the friction tag)

**Not promoted (PI declined):**

- ⛔ (2) `wilson-lo-min-sample-size` — PI declined; smoke→32-seed
  workflow stays as it is. Caller responsible for prominent sample
  size labelling.
- ⛔ (4) `trust-artifact-json-not-bg-log` — PI declined; staying with
  current bg-redirect pattern, agent caller responsibility to
  cross-check JSON artifact.

## PI additions (from step 4)

PI: "Nothing to add — my draft covers it." No frictions or
decisions added; postmortem stands as drafted.

## Framework version at session-end

- Commit SHA: `16b0c57`
- Active rules: 1..36 (per CLAUDE.md `## Operating rules`)
- Loaded skills this session: postmortem (this), claude-api (not used),
  kaggle-comp (improvements.md targeted by promotion candidates)
- Branch: `claude/fix-early-game-strategy-YSClQ`
- 5 commits today: `0a6b16d` → `7f2418d` → `393a710` → `7d3dbc6` → `16b0c57`
- Submissions used today: 0/5
- Live ladder: v7_minimax μ=1063.0 (team peak), top-10 cliff +369μ above us
