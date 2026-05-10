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

## 2026-05-10 (day-1 agent — bootstrap branch)

- `tag: audit-date-must-track-system-currentdate` — wrap-up context: I
  stamped my Day-1 audit files and friction heading as `2026-05-09`
  because the kickoff began that local-time evening, but the system
  reminder's `# currentDate` and the submission's UTC timestamp were
  both `2026-05-10`. This forced a follow-up rename pass across
  audit/, HANDOVER.md, ISSUES.md, comp-context.md, scripts/, and
  state/current.md once the submission landed and the discrepancy
  surfaced. Root cause: I anchored on my internal narrative ("Day-1
  started yesterday") instead of the system-provided date, even though
  Rule 35-style "session-end second-brain" thinking should treat the
  system clock as canonical. **Fix:** at session start, capture the
  `# currentDate` value once and use it as the YYYY-MM-DD prefix for
  every artifact written that session — never the agent's recollection
  of when work began. If the session crosses 00:00 UTC, prefer the
  later date because it matches what `git log` and Kaggle CLI will
  stamp.

- `tag: kaggle-api-token-required-for-kgat-format` — day-1 bootstrap:
  `kaggle competitions list -s orbit` returned 401 with only
  `~/.kaggle/kaggle.json` containing `{"username": ..., "key": "KGAT_..."}`.
  Auth succeeded immediately after `export KAGGLE_API_TOKEN="$KAGGLE_KEY"`.
  Root cause: the new KGAT_… personal-access-token format is read by the
  CLI from `KAGGLE_API_TOKEN` (or `~/.kaggle/access_token`), not from the
  `key` field of `kaggle.json` — which expects the older 32-hex token.
  **Fix:** `bootstrap.sh` already sets `KAGGLE_API_TOKEN` from `KAGGLE_KEY`
  inside its own subshell (line 34-36), but the export does NOT propagate
  to subsequent shells; downstream `kaggle …` invocations must be prefixed
  `KAGGLE_API_TOKEN="$KAGGLE_KEY"` or this export must live in the parent
  session. Consider adding a `.kaggle/access_token` file (mode 600) so the
  CLI picks it up unconditionally — then no env-var dance is needed.
- `tag: pip-blinker-system-conflict` — day-1 bootstrap:
  `pip install -r requirements.txt` aborted with
  `ERROR: Cannot uninstall blinker 1.7.0, RECORD file not found. Hint:
  The package was installed by debian.` Root cause: `kaggle` pulls in
  `flask`/`requests` extras that bump `blinker`; the system-installed
  Debian `python3-blinker` lacks pip RECORD metadata, so pip refuses to
  upgrade in place. **Fix:** run `pip install --ignore-installed blinker`
  first, then retry `pip install -r requirements.txt`. Worth folding into
  `bootstrap.sh` step 2 as a pre-flight when running on Debian/Ubuntu base
  images.
- `tag: seed-repo-out-of-mcp-scope` — bootstrap context: the seed lives at
  `chris0leite-ui/Kaggle-playground-may-2026`, but this session's MCP
  scope and local git proxy are both allowlisted only for
  `chris0leite-ui/Orbit-wars-kaggle`. `mcp__github__get_file_contents` →
  `Access denied`; `git clone http://local_proxy@127.0.0.1:.../Kaggle-...` →
  proxy refused with 502 / `Couldn't connect`. Resolved by the PI
  pointing me at the direct GitHub URL — `git clone -b
  claude/orbit-war-setup-KbeKq https://github.com/chris0leite-ui/Kaggle-playground-may-2026.git`
  bypassed both gates because the proxy resolves direct HTTPS to GitHub.
  Root cause: the per-session repo allowlist is a deliberate sandbox,
  but cross-repo seeding workflows aren't called out in the kickoff
  prompt. **Fix:** the `agent-handover-prompt.md` should explicitly state
  "if you cannot reach the seed repo via the local proxy, fall back to
  `git clone https://github.com/<owner>/<seed-repo>.git` directly." (Open
  for next session to PR into the prompt.)

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
