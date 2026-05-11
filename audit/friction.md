# audit/friction.md — current friction summary

## 2026-05-12 (analyze-leaderboard-strategies-sdZlE)

- `tag: multi-mission-stack-regresses-even-with-conditional-gates` —
  v3.5 build: I added four new Mission classes (opening / drain /
  gang_up / recapture) on top of v3_snipe + reinforce, each with
  what I believed were tight conditional gates (step-window,
  garrison threshold, eta-window, recently-lost-window). 32-seed
  full-stack A/B vs v3_snipe = 39.1% (Wilson lo 28.1%, FAIL).
  16-seed per-wave ablation: opening 40.6%, drain 46.9%, gangup 50.0%,
  recapture 53.1% — NONE clears the 55% Wilson lo gate individually.
  Same family as the v3.4 NEUTRAL_BONUS=1.5 regression (28.1% A/B) and
  the v3.3 blanket-(eta+1) regression (42.2% A/B). Root cause: the
  per-source-greedy planner is unexpectedly sensitive — adding ONE
  Mission class shifts the proposal distribution enough to displace
  higher-EV snipe picks at the same source. Mission classes are NOT
  independent through settle_plan. **Fix this session:** v3.5 NOT
  submitted; code retained on branch; debug hypotheses written to
  `audit/2026-05-12-v3.5-stack-results.md`. *Promotion candidate*:
  "New mission classes must clear a 16-seed Wilson lo ≥ 0.55 ABLATION
  gate (variant ⊕ v3_snipe vs v3_snipe alone) BEFORE being stacked
  into a multi-class agent." Three consecutive sessions have learned
  the same lesson; encode as a rule.

- `tag: ab-harness-not-reusable-for-arbitrary-pairs` — v3.5 ablation:
  `scripts/tournament.py` has only a `smoke` CLI subcommand (random vs
  baseline, 4 seeds). To run v3.5-vs-v3_snipe at 32 seeds and four
  per-wave ablations vs the same baseline I had to write THREE thin
  drivers (`scripts/run_v35_ab.py`, `scripts/run_ablation_panel.py`,
  `scripts/run_phys_ab.py`) — ~70-90 lines each, all nearly identical
  argparse + Wilson-lo + tournament-result-dump scaffolding. **Fix:**
  promote a generic `scripts/run_ab.py --agents A=path B=path
  [C=...] --seeds N --workers W --gate-baseline A` that prints the
  Wilson-lo verdict per pair. Should subsume `run_v35_ab.py`,
  `run_phys_ab.py`, `run_ablation_panel.py` (all deletable). Promotion
  candidate; deferred to a session that touches `scripts/tournament.py`
  proper.

- `tag: kaggle-env-var-case-confusion` — analysis-pull bootstrap:
  spent ~15 min before realising that `$KAGGLE_key` (lowercase, what
  I'd assumed) was empty while `$KAGGLE_KEY` (uppercase, what
  bash-lc-shells actually see) had the value. The
  `tag: kaggle-api-token-required-for-kgat-format` entry (2026-05-10)
  already covers the KGAT_ token-format issue but NOT the env-var
  case-sensitivity foot-gun. **Fix:** the bootstrap.sh comment block
  that documents `KAGGLE_API_TOKEN="$KAGGLE_KEY"` should NOT use
  variable substitution case-variants near each other; the kickoff
  agent-handover prompt could include `bash -lc 'env | grep -i kaggle'`
  as the canonical "what credentials do I have" check.

## 2026-05-11 (bootstrap-agentic-systems-lqnm6)

- `tag: guard-mechanisms-check-only-to-predicted-endpoint` — wrap-up
  context: live-replay analysis of 21 v2 episodes found 7.5% of our
  fleets flew OOB, 3.2% died in the sun. All three guards
  (`sun_avoid`, `path_clears_other_planets`, `oob_guard`) only
  simulated the fleet path up to the predicted target arrival step
  (`max_steps ≈ total_dist / speed`). When the lead-prediction missed
  (orbital drift, tangent shot), the fleet kept flying through empty
  space past the predicted spot until OOB/sun, but the guards stopped
  checking. **Fix:** new `lib/trajectory.predict_fleet_fate(src,
  target, angle, ships, world)` ray-casts the FULL trajectory until
  the first collision (target / planet / sun / OOB). All three guards
  delegate to it. Capture probe: OOB 7.5% → 2.6% → 0.3%; sun 3.2% →
  0.1% → 0.0%; reached 77.2% → 93.0% → 97.2% across the three fix
  waves of today.

- `tag: bundler-missing-block-e-modules` — submission flow: first
  `python -m scripts.bundle_agent agents/v3_snipe` produced a 53KB
  bundle that crashed 10/10 self-play with `NameError: 'propose_
  snipe_missions' is not defined`. Root cause: `DEFAULT_LIB_ORDER`
  in `scripts/bundle_agent.py` didn't include `lib/mission.py`,
  `lib/missions/snipe.py`, `lib/missions/reinforce.py`, or
  `lib/planner.py` — these Block E modules were added since the
  bundler was last touched, and `_INTRA_IMPORT_RE` silently strips
  intra-package imports so the symbols just go undefined. **Fix:**
  `DEFAULT_LIB_ORDER` now includes the four Block E modules
  (subpackage paths like `"missions/snipe"` resolve via pathlib
  transparently). Bundle E.2 gate cleared 0/10 after the fix.
  *Promotion candidate*: the bundler should auto-discover required
  modules via AST analysis of the agent's imports rather than
  relying on a hand-maintained list.

- `tag: 8-seed-mvp-result-is-noise` — Block E v3.1 lookahead MVP
  showed 11/16 = 68.8% lift over v2 in an 8-seed panel; PI excitement
  level was high. 32-seed retest collapsed to 50/50 parity (16 wins
  in either direction, all draws at step 500). Wilson 95% CI on 11/16
  is [44%, 86%] — already wide enough to encompass parity. **Fix:**
  do not publish lift headlines from n ≤ 16 seeds. The plan file's
  recommended gate was always 32 seeds × both seats; I jumped ahead.
  Same pattern likely to recur — promotion candidate to encode "≥32
  seeds for any lift claim" as a gate rule.

- `tag: until-loop-spurious-process-detection` — wrap-up context:
  `until [ -z "$(pgrep -f 'scripts.strategy_panel' 2>/dev/null)" ];
  do sleep 30; done` exited early during a 32-seed panel run because
  pgrep briefly returned empty during a worker-process transition.
  Reported "BOTH DONE" while the panel was actually still at game
  23/64; led me to a false "panel was killed" diagnosis and a
  redundant duplicate panel start (which then competed for CPU with
  the original). **Fix:** prefer the background-task `run_in_background`
  + auto-notification path over manual pgrep polling. If polling is
  unavoidable, require N consecutive empty pgrep checks before
  concluding the process has finished.

- `tag: rolling-last-2-tradeoff-needs-explicit-decision-record` —
  submission flow: v3_snipe push at 12:16 UTC evicted v1.2/roi
  (μ=1006.9, the OLDER and higher-rated of the two prior slots),
  leaving rolling pair = [v2 (974.3, BUGGY guards), v3_snipe
  (PENDING)]. The choice was rational (v3 ≫ v1.2 locally at 97%
  in 2P; v2's buggy guards are an active drag) but the trade-off
  wasn't recorded as a deliberate decision until the wrap-up. **Fix:**
  before every submit, write a one-line decision record citing the
  evicted submission and the expected-Δμ. Promotion candidate to
  formalise as a Rule 12 sub-clause.

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
- `tag: kaggle-oauth-503-on-submit-only` — submission flow:
  `kaggle competitions submit` returned `requests.exceptions.HTTPError:
  503 Server Error: Service Unavailable for url:
  https://api.kaggle.com/v1/security.OAuthService/IntrospectToken`
  on two consecutive attempts ~5 min apart, while
  `kaggle competitions submissions` (GET) worked cleanly between them.
  Root cause: Kaggle's OAuth introspection endpoint flapped
  specifically on the submit POST path; not our credentials. Verified
  no slot was consumed via the submissions list. **Fix:** PI-authorized
  third attempt; succeeded as ID 52518060. Promotion candidate: a
  small wrapper `scripts/safe_submit.sh` that pre-checks via GET
  then submits + verifies the new entry appears, surfacing 5xx as
  "retry, not loop" to keep us inside Rule 1's spirit.

## 2026-05-10 (evening — competition-strategy-brainstorm-ZK6XT)

- `tag: arrival-ledger-mechanism-without-planner-regresses` — Block C
  integration: putting `arrival_ledger` inside `DEFAULT_MECHANISMS`
  regressed v2 WR from 56% to 50% in A/B
  (audit/tournaments/20260510T215332Z.json). Root cause: per-source
  greedy strategies don't re-pick after a mechanism drops their
  intent; sources go idle for the turn. The "don't double-commit"
  filter is correct in principle but starves productive turns
  without a planner that re-allocates freed ships across mission
  classes. **Fix:** kept `arrival_ledger` implemented but EXCLUDED
  from `DEFAULT_MECHANISMS`. Moved the dedup logic to strategy level
  in `agents/v2/main.py` where each source can re-rank and re-pick.
  v3 planner will use the mechanism directly inside settle_plan
  where re-allocation IS available.
- `tag: strategy-level-affordability-filter-prefers-low-roi` — Block D
  first attempt: v2 bumped intent.ships per `WorldModel`-predicted
  defense AND filtered out targets whose bumped-ships exceeded
  src.ships. Resulted in 0/64 WR vs both roi and roi_baseline
  (audit/tournaments/20260510T215806Z.json) because the agent
  preferred LOW-ROI affordable targets over HIGH-ROI ones that the
  mechanism layer's `arrival_size` would have let through unchanged.
  Root cause: `propose_intents` should never filter on affordability
  — that's `validate`'s job. **Fix:** rolled back to "skip
  already-ours only, no bumping". Strategy-level WorldModel use is
  read-only (dedup); ship-sizing stays in the mechanism layer.
  Pattern lesson queued as a promotion candidate (see
  `audit/2026-05-10-postmortem-competition-strategy-brainstorm-ZK6XT.md`).
- `tag: bundle-output-clobbers-prior-bundles` — Block A bundling:
  `python -m scripts.bundle_agent agents/simple/roi.py --out
  submissions/` wrote to `submissions/roi.py` and silently
  overwrote the frozen v1.2 bundle that submission #52518060 was
  built from. The bundler computes the output filename from
  `<agent_dir>.stem` which collides for any re-bundle of the same
  agent. Recoverable here because v1.2 is on Kaggle; would have
  been a real loss if we'd needed a local A/B against the frozen
  prior. **Fix:** renamed the new bundle to
  `submissions/v1_3_roi_physics.py` manually; promotion candidate
  in postmortem to default the bundler's output path to a
  versioned filename (suggested `<agent>-<git-short-sha>.py`) or
  to refuse to overwrite an existing file without `--force`.
- `tag: handover-md-over-line-cap` — wrap-up: HANDOVER.md grew to
  439 lines this session — well over WRAPUP.md's 150-line cap —
  because each of three prior PM-branch session-end blocks
  appended without archive. Per WRAPUP step 5, oldest sections
  should be archived to `audit/archive-YYYY-MM-DD-<topic>.md`
  before commit. **Fix this session:** archive the three completed
  prior PM-branch blocks (review-competition-handover-0pGNc,
  simple-trading-strategies-QS0xV, improve-strategy-ab-testing-jYA2R)
  to `audit/archive-2026-05-10-handover-prior-pm-sessions.md`,
  retain only the bootstrap morning sections + tonight's
  competition-strategy-brainstorm-ZK6XT block.
- `tag: time-estimates-too-pessimistic-by-5x` — strategy-direction
  plan v1: original draft estimated Roman-equivalent v3 in **12-15
  days**. PI overrode: "your time estimates are off. tomorrow
  morning you will be done." Actual delivery from PI ratification
  to v2 ladder push: ~6 hours. Calibration: my budget for
  code-comp implementation work was ~5x too pessimistic. **Fix:**
  carry this calibration forward — when scoping subsequent Blocks
  (E1 mission framework, E2 gang_up, etc.), estimate in hours not
  days. The Block E1 plan budget of "4-6 hours" reflects the
  re-calibration.
