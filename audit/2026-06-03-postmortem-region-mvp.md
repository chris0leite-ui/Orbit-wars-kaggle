# Postmortem — 2026-06-03 region-mvp

## What went wrong
- **Under-estimated per-game cost before sizing the A/B sweep.** Launched the
  full 3-weight sweep with a 20-min timeout assuming ~30–90s/game; real cost was
  ~7–8 min/game (500 turns × ~1s). Couldn't finish one weight → had to stop and
  restructure. One timed game up front (Rule 2 "1-fold time-probe" analogue)
  would have sized it correctly.
- **Piped a live A/B through `grep` for monitoring** → block-buffering made a
  healthy run look stalled for ~14 min.
- **Parity baseline built without the champion config header** → false
  divergence at step 1 (config mismatch, not a code diff); ~8-min run wasted
  before catching it.
- **`timeout 540` on a ~21-min self-play test, twice** → two SIGTERMs that
  looked like test failures (the no-crash guarantee was already covered by 94
  completed A/B games).
- No PI overrides this session. The null itself was a good decision: the prior
  handover named the lever, it was cheap, and it's now cleanly falsified
  (decision-quality good regardless of the null outcome).

## Frictions logged this session
See `audit/friction.md` `## 2026-06-03`:
- `grep-pipe-buffers-ab-monitoring`
- `parity-baseline-missing-config-header`
- `ab-timeout-too-short-for-full-games`
- `pkill-sleep-sandbox-blocked-exit144`

## Promotion candidates (PI ratified: NO)
Drafted four (`time-probe-before-ab-sweep`, `no-grep-pipe-on-live-jobs`,
`parity-baseline-config-match`, `sandbox-no-pkill-no-sleep`). **PI: "nothing to
add or to promote."** None promoted to improvements.md.

## PI additions (from step 4)
None — PI declined additions.

## Framework version at session-end
- Commit SHA: c95c9aa (region-score null) + this wrap commit
- Active rules: CLAUDE.md Rules 0–47
- Loaded skills this session: loop, postmortem
