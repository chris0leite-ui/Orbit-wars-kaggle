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
- `tag: env-not-fully-seed-deterministic` — Step 0 regression check
  (re-running `scripts/run_day1_rollouts.py` against the bootstrap-irewT
  audit JSON): per-seed **rewards / statuses are stable** (P0=6/6 vs
  random; baseline-vs-baseline P1=4/6, P0=2/6, no ties this run vs 1
  tie before), but **`final_ships` counts and `n_steps` differ** for
  every game (e.g. seed 13 went from `n_steps=227 / ships=3532` to
  `n_steps=399 / ships=3585`). Root cause not yet pinned — likely
  Python set/dict iteration order (hash randomisation, or
  unordered-set iteration inside the env) seeded independently of
  the configured seed. **Fix / implication for D.1:** the tournament
  fixture must treat **rewards** as the stable signal for winrate gates
  and treat ship-deltas / turn-counts as **noisy estimators** that
  need ≥N seeds before reading. Non-byte-equal artifacts are not a
  regression as long as the rewards counts match. Keep the original
  audit JSON as the canonical record; do not overwrite on re-run.
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

## 2026-05-10 (PM — simple-trading-strategies-QS0xV)

- `tag: requirements-not-installed-on-fresh-shell` — Phase 0 build:
  fresh container had no `kaggle_environments` / `pytest` /
  `scikit-learn` available; `pip install -r requirements.txt`
  failed once on `blinker` system-package conflict. **Fix:** use
  `pip install -r requirements.txt --break-system-packages
  --ignore-installed blinker --quiet`. Captured this in the
  bootstrap workflow rather than per-session friction; future
  agents can rerun bootstrap.sh to get the same result. The
  `kaggle_environments` install cascades a noisy 23-line OpenSpiel
  loader log on every Python import — annoying but harmless.
- `tag: data-main-py-not-fetched-by-bootstrap` — Phase 0 build:
  111 existing tests failed with "Could not find :
  /home/user/Orbit-wars-kaggle/data/main.py" because `bootstrap.sh`'s
  data-download step does NOT actually run on fresh containers
  here (the smoke-only path runs but the `kaggle competitions
  download -c orbit-wars` is gated by an unclear condition).
  **Fix:** ran `kaggle competitions download -c orbit-wars -p data/`
  manually + `unzip data/orbit-wars.zip -d data/`. Should be promoted
  to a session-start hook or to the bootstrap.sh fast-path.
- `tag: bundler-flat-file-agents-not-supported` — Phase 0 →
  submission staging: `scripts/bundle_agent.py` hardcoded
  `<agent_dir>/main.py`, but the simple-strategy panel uses
  flat-file agents at `agents/simple/<n>.py`. **Fix:** extended
  `bundle()` in `scripts/bundle_agent.py` to accept either a
  directory containing `main.py` OR a single `.py` file as the
  agent entry. Existing dir-mode behaviour preserved.
- `tag: importlib-spec-no-sys-modules-registration` — Phase 0:
  loading `submissions/roi.py` via `importlib.util.spec_from_file_location`
  + `exec_module` raised `AttributeError: 'NoneType' object has no
  attribute '__dict__'` inside `dataclasses.py::_process_class`.
  Root cause: dataclasses checks `sys.modules.get(cls.__module__)`
  for KW_ONLY sentinel, which returns None when the module isn't
  registered. **Fix:** `sys.modules['<spec_name>'] = mod` before
  `exec_module(mod)`. Should propagate this pattern into
  `scripts/tournament.py::_load_agent` if it ever loads bundled
  submissions directly.
- `tag: sklearn-1.8-dropped-multi_class-kw` — Phase 1: sklearn 1.8
  dropped the `multi_class="auto"` keyword on `LogisticRegression`
  (`TypeError: __init__() got an unexpected keyword argument
  'multi_class'`). **Fix:** removed the kwarg in
  `scripts/manifold_check.py`; the new default ("auto" inference
  by class count) is what we wanted anyway. Code that pinned
  `multi_class` for older sklearn would need a version guard;
  this comp doesn't have any other sklearn touchpoints today.
- `tag: nearest-equals-v1_orbitfix-by-construction` — Phase 1:
  `agents/simple/nearest.py` deliberately reproduces `agents/v1_orbitfix/`'s
  targeting + RNG seed as the strategy-panel control. The 7-class
  manifold check correctly classified them as mutually-confusable
  (29% mutual confusion) — but this is *correct* in-data behaviour,
  not a fingerprint failure. **Fix:** the gate target was always
  the 5-strategy zoo (excluding v1_orbitfix); `manifold_check.py`'s
  `--strategies` filter handles this. Surfaced in
  `audit/2026-05-10-phase1-manifold-verdict.md` so future agents
  reading the 7-class confusion don't re-derive the diagnosis.
- `tag: roi-family-shares-one-basin` — Phase 1: `nearest`,
  `production`, `roi` all share `DEFAULT_MECHANISMS` and a
  distance/production-aware score function, so their behavioural
  fingerprints overlap (12-17% mutual confusion at K ≤ 200 with
  the 15-feature design). The `_infer_target` ray-cast proxy is
  the load-bearing weakness — it loses the "which planet did they
  attack" signal that would actually separate them. **Fix not yet
  applied:** queued as H-coarsen-labels (merge ROI-family) or
  H-richer-fingerprint (target-distance/production distribution
  shape + early-vs-late split + target-id Shannon entropy) in
  state/hypothesis-board.md. PI to choose path.
