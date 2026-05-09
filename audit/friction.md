# audit/friction.md — current friction summary

> One entry per distinct friction event, grouped under a `## YYYY-MM-DD`
> heading. Format:
>
> ```
> - `tag: <kebab-slug>` — <session/day context>: <what happened>.
>   <Root cause>. **Fix:** <concrete action>.
> ```
>
> Reuse existing tags when possible. Persistent / cross-comp frictions
> get promoted to `.claude/skills/kaggle-comp/improvements.md` via the
> postmortem skill.

## 2026-05-09 (seed pre-population)

The three entries below were observed during the seed-build session
itself, before the day-1 agent runs. Logged here so the day-1 agent
does not re-discover them.

- `tag: kaggle-cli-no-competitions-view` — seed-build context: the
  Orbit Wars-era kaggle CLI dropped the `competitions view`
  subcommand in favour of `competitions pages`. Documentation snippets
  inherited from s6e5 prose still reference the old form.
  Root cause: kaggle CLI version drift; the new form supports
  `--page-name {description,evaluation,rules,...}` and `--content`.
  **Fix:** SETUP.md and the agent-handover prompt both reference the
  new form. Day-1 agent: use
  `kaggle competitions pages orbit-wars --content` and
  `kaggle competitions pages orbit-wars --content --page-name evaluation`.
- `tag: env-name-underscore-vs-hyphen` — seed-build context: the
  Kaggle competition slug is `orbit-wars` (hyphen) but the
  `kaggle-environments` env name is `orbit_wars` (underscore). Mixing
  these silently fails: `make("orbit-wars")` raises
  `Environment orbit-wars not found`. **Fix:** SETUP.md, bootstrap.sh,
  and comp-context.md all use the underscore form for the env. The
  hyphen form is for the Kaggle CLI only (slug).
- `tag: rolling-last-2-not-pi-selected` — seed-build context: in
  Orbit Wars, the platform automatically uses your **rolling last 2
  submissions** for final evaluation — there is no PI-selectable
  pair at the deadline. The s6e5 R2 default ("PRIMARY = best public,
  HEDGE = best OOF that regressed ≤30 bp") does not apply.
  Root cause: code-comp evaluation differs from tabular Playground.
  **Fix:** CLAUDE.md R-defaults block now flags R1, R2, R5, R7 as
  TABULAR-ONLY; R2 has a code-comp default inline. Submission cadence
  is the strategic lever, not endpoint selection.

## Anticipated frictions (likely first-week)

These have not yet fired but are predictable from the comp spec —
log them again under their actual date when they occur, with a
concrete example. Removing this section is fine once each has
fired once and been logged for-real.

- `tag: trueskill-noise-vs-signal` — TrueSkill σ is large early
  (μ₀=600 with default σ≈300). First-24h rank shifts are dominated
  by σ shrinkage, not μ change. **Anticipated fix:** wait ≥24 h
  after a submit before reading anything into the rank delta;
  budget at least one full day of ladder play per submit.
- `tag: validation-episode-blocking` — every kernel push triggers a
  self-vs-self validation episode (multi-minute wallclock) before
  joining the ladder. Slot accounting must wait for validation to
  pass. **Anticipated fix:** check
  `kaggle competitions submissions orbit-wars` after submit; do NOT
  treat the submit as "live on ladder" until the validation episode
  reports success.
- `tag: agent-1-second-act-timeout` — `actTimeout=1` per turn is
  tight. A naive Python agent with O(N²) scans over 40 planets +
  fleets can blow this on a slow worker. **Anticipated fix:**
  profile worst-case `agent(obs)` locally; any branch >500 ms wallclock
  needs optimisation before submit.
