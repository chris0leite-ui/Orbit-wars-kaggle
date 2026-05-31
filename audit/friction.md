# audit/friction.md — current friction summary

> One entry per distinct friction event. Format:
>
> ```
> - `tag: <kebab-slug>` — <session/day context>: <what happened>.
>   <Root cause>. **Fix:** <concrete action>.
> ```
>
> Reuse existing tags where possible. **One line per event ideal;
> three lines max.** When this file exceeds 150 lines, rotate per
> self-improvement.md::Weekly distillation — archive to
> `audit/friction-archive-YYYY-MM-DD.md` and reset.
>
> Last rotation: 2026-05-14 (claude/audit-workflow-friction-XD56a).
> Full prior history at `audit/friction-archive-2026-05-14.md`.

## Just-landed fixes (claude/audit-workflow-friction-XD56a — 2026-05-14)

These patterns recurred multiple times; the fix is now in source. If
the next session sees them re-fire, the regression test is missing —
fix forward AND add a test.

- `[FIXED] data-main-py-missing-on-fresh-clone` — `bootstrap.sh` now
  guards on `[[ -f data/main.py ]]`, not on "any non-gitkeep file".
  Stops `data/shot_validator/` from masking the download skip.
  Recurred 3x (5/10, 5/12, 5/13).
- `[FIXED] kaggle-cli-401-wrong-auth-env-var` — `bootstrap.sh` now
  detects `KGAT_`-prefix tokens and routes them via
  `KAGGLE_API_TOKEN` instead of `kaggle.json`'s legacy 32-hex path.
  Also maps harness names (`KaggleUserName` / `KaggleAPIToke`) to
  canonical. Plus a `kaggle competitions list -s orbit` cred smoke
  surfaces 401s in minute one. Recurred 3x (5/10, 5/12, 5/13).
- `[FIXED] pip-blinker-system-conflict` — `bootstrap.sh` does a
  `pip install --ignore-installed blinker` preflight before
  `pip install -r requirements.txt`. Stops Debian's record-less
  `python3-blinker` from aborting the whole install.
- `[FIXED] bundler-missing-block-e-modules` (and 5 related tags) —
  `scripts/bundle_agent.py` now raises `RuntimeError` if any
  `from lib.X import ...` strips to an X not in the bundle order
  list. No more silent-NameError bundles. Bundle parity-gate alone
  was insufficient — the historical parity test used `v1_orbitfix`,
  which doesn't exercise the snipe-stack mission framework.
- `[FIXED] bundler-overwrites-tracked-submission` — bundler refuses
  to overwrite a git-tracked file in `submissions/` without
  `--force`. Caught one silent deletion of `v7_0_drop_one.py` on 5/13.
- `[CODIFIED] consecutive-falsification-cap` — Rule 37 in CLAUDE.md.
  3+ consecutive variants in the same design axis ⇒ pivot or
  escalate. Cost evidence: v7_X chooser axis sweep on 5/13-14 burned
  6 h on diminishing-EV variants past v7_3.
- `[CODIFIED] kaggle-kernel-mandatory-two-tier-smoke` — folded into
  Rule 2. Local CPU single-state smoke + small-scale GPU smoke
  before any production T4 push. Cost evidence: 90 min T4 quota on
  a JIT compile a 5-min CPU run would have flagged.
- `[FIXED] local-vs-v7_0-only-misses-ladder-distribution` —
  `fast.py eval --vs-panel` runs a 3-opponent calibration panel
  (`v7_0`, `v4_planner`, `v3.5.1`) by opt-in; PASS verdict requires
  every opponent's Wilson lower bound ≥ gate, not pooled. v3.5.1
  (5/12, -150μ vs prediction) and geo v3.1 (5/14, -80μ floor) both
  passed single-opponent A/Bs and regressed on the ladder.
  **Open follow-up:** make `--vs-panel` mandatory before submission
  (workflow rule, not yet hard-gated in source).

## Newly-fired patterns (this session)

- `tag: fix-not-validated-against-real-failing-state` — 2026-05-14
  audit-pass: I patched `bootstrap.sh` for `data-main-py-missing-on-
  fresh-clone`, ran the unit guards (syntax check, AST tests, --help
  output), saw 16 pytest failures with the exact `data/main.py` error
  string the patch was meant to neutralise, and categorised them as
  "pre-existing, not regression" instead of running the patched
  bootstrap. PI caught it: "you have not been able to bootstrap
  properly even though you noticed the friction." Same pattern as
  `agent-introspection-skipped-bootstrap` (2026-05-13). **Root cause:**
  fix-verification protocol was "unit-test the new code path" rather
  than "reproduce the failure state and confirm fix neutralises it."
  The rule was written in friction.md but never bound because friction
  notes don't gate behaviour. **Fix this session:** promoted to
  CLAUDE.md Rule 38; bumped SessionStart bootstrap hook to top
  pending in `improvements.md`.

- `tag: helper-reimplemented-inline-silently-wrong` — 2026-05-14
  game-strategy-eda-roatN: bundling-friendly inline rewrite of
  `lib.orbit.is_orbiting` checked `dist_from_sun > 0.5 AND radius > 0`
  instead of `orb_r + planet_radius < ROTATION_RADIUS_LIMIT`. Every
  self-play board reported `orbital_frac=1.00` (training corpus
  0.27-0.44); KMeans nearest-centroid put every board in cluster 3.
  v2 sweep ran at 67% vs v7_0 — looked encouraging — actually noise
  from the forced-C3 high-cadence template. **Root cause:** I
  paraphrased the library invariant instead of inlining the formula
  verbatim. **Fix this session:** corrected the proxy + added an
  outlier-distance threshold; the broader fix needed is a project
  rule like "when inlining for bundling, paste the source line, not
  a proxy" — promotion candidate for `improvements.md`.

- `tag: broken-mechanism-yields-fake-positive-signal` — 2026-05-14
  same session: the v2 sweep's 67% point estimate against v7_0 was
  taken as "directionally encouraging" support for the cluster-
  conditional overlay. After fixing the underlying classifier bug
  (above), v3 sweep collapsed to 53% with overlay-active games at
  46% and pure-v7-fallback at 80%. The "encouraging" result was the
  bug. **Root cause:** acted on a positive sweep result without
  verifying the upstream mechanism (the classifier) was actually
  classifying. **Fix forward:** any "we beat the panel by X%"
  reading requires a 30-second sanity print of what the agent
  actually does (cluster distribution, launch count distribution)
  before treating the number as signal.

- `tag: soft-clusters-need-confidence-fallback` — 2026-05-14 same
  session: Mine 1 had already flagged silhouette ≈0.17 ("clusters
  are real but not sharply separated") as a risk; I treated the
  k=4 KMeans output as a usable categorical anyway. With centroid
  distances spread training p25=2.05 → p95=3.30, marginal boards
  get force-routed into a wrong template. **Root cause:** soft
  clusters + hard nearest-centroid classification = templates
  applied where they don't fit. **Fix forward:** when silhouette
  < 0.20, the classifier ships with a confidence threshold (defer
  to a default policy beyond p90 distance) from day 1, not after
  a failed sweep.

## 2026-05-28 (claude/competition-objective-alignment-hqNVM — value-head Phase A)

- `tag: phase-a-bench-vs-evicted-not-rolling-pair` — Phase A A/B
  evaluated `baseline_learned` against `favor_hybrid` (the head behind
  the EVICTED μ=1149 team peak), not against the live rolling-last-2
  (μ=806 / μ=829). Verdict "wiring sound" is calibrated to a strong
  but absent opponent, not to what Kaggle will judge against. Risk: a
  future Phase B candidate could match favor_hybrid (good) yet still
  regress vs the actual rolling-pair baselines, because the rolling
  pair has its own quirks the head was never optimised against.
  **Root cause:** Phase A was scoped as a substrate-only diagnostic
  (wiring/feature-sufficiency); live-ladder calibration was correctly
  deferred to Phase B. **Fix forward:** before any Phase B submit,
  Rule 43 panel + Rule 45 n≥32 vs rolling champion are mandatory; no
  Phase A artifact gets pushed on its own.

- `tag: n32-inconclusive-still-a-pass-for-diagnostic` — Phase A
  result was 14/32, Wilson [0.282, 0.607], a Wilson-CI-crosses-50%
  INCONCLUSIVE verdict. We called it a Phase A "pass" anyway, on the
  strength of the 38pp jump from v1's 6/32. **Not a real friction**
  but worth tagging: for a SUBSTRATE diagnostic (does the wiring
  work?), a magnitude jump from "clearly broken" to "near-parity" IS
  the answer; statistical certainty isn't required because we're not
  shipping the artifact. For a SUBMISSION decision, n=32 + crosses-50%
  is a hard fail (Rule 45). The two thresholds are not the same.

- `tag: bundle-agent-doesnt-inline-from-baseline-main` — bundling
  `agents/baseline_joint_aggr_consolidated_orbitfix/main.py` (a wrapper
  whose body is `from agents.baseline.main import agent` + env-var
  setdefault block) produced a 350 KB output with **0 `def agent`**.
  The bundler stripped the import without inlining; kaggle_environments
  would fall back to the last callable in the module (wrong signature
  → game ERROR at step 0). Identical failure mode the sibling branch
  `claude/strategy-axis-decision-3437` (c25a329) documented two days
  ago — they noted their "n=8 vs LATEST 8W/0L" was Phase 4 beating an
  ERROR-on-step-0 file. **Fix landed this session (cherry-pick
  c25a329):** `scripts/bundle_agent.py` now exec-imports the bundle
  post-write, refuses if `agent` symbol is not callable, unlinks the
  output. Plus `tests/test_submissions_loadable.py` parametrised
  regression test catches the bug class for any bundle in tree.

- `tag: clean-ab-crashes-on-single-game-timeout` — `scripts/clean_ab.py`
  used `subprocess.run(..., timeout=600)`; one pathological game on
  seed 3 took >600s, raised `TimeoutExpired`, the `ProcessPoolExecutor`
  bubbled it via `fut.result()` and killed the whole script. Lost all
  in-flight results from sibling workers (seeds 5-15 were running,
  block-buffered stdout meant nothing flushed before unwind). 32-game
  A/B reduced to 8 visible results. **Fix landed (38372f4):** try/except
  `TimeoutExpired` per-worker; bump 600 → 900s; line-buffer stdout
  (`sys.stdout.reconfigure(line_buffering=True)`); wrap `fut.result()`
  in try/except.

- `tag: cpu-oversubscription-kills-ab-throughput` — ran two clean_ab
  A/Bs in parallel at `--workers 4` each (8 worker processes on a 4-CPU
  box). Load shot to 5–7×, individual games took 300–500s instead of
  150–250s typical, the 600s timeout (later 900s) became hot.
  Lesson: workers-per-AB × number-of-ABs ≤ nproc. **Fix forward:**
  run A/Bs sequentially when more than one is needed, or scale workers
  to `nproc / num_concurrent_abs`.

- `tag: bundler-rebuild-requires-prepend-recipe` — variant agent dirs
  (one-file shims via `setdefault` + `from agents.baseline.main import
  agent`) cannot be bundled directly even after c25a329's loadable
  guardrail — the guardrail now REFUSES the broken bundle but doesn't
  produce a working one. The friction.md recipe (bundle
  `agents/baseline`, then prepend env vars to `submissions/baseline.py`)
  remains the workaround. **Promotion candidate:** make
  `scripts/bundle_agent.py` recognise the wrapper pattern and either
  inline `agent` automatically OR bundle the underlying agent and
  apply the env-var prepend. Tracked in
  `knowledge-base/questions/2026-05-22-what-orbital-safety-doesnt-yet-capture.md`
  is the modelling follow-up; this bundler-UX work is separate.

## 2026-05-21 (claude/review-skills-improvements-moKOR — drain/sniper iteration)

- `tag: sniper-fleetfate-step-of-hit-attribute-bug` — first sniper
  variant (`emit_sniper_strikes`) accessed `fate_p.step_of_hit` but
  the FleetFate dataclass field is named `step`. The misleading
  docstring at `lib/trajectory.py:25` advertised `step_of_hit` as the
  return field name, so the bug was authored from the docstring not
  the source. AttributeError raised on first sniper trigger and
  cascaded ALL 4 agents to `status=ERROR` (kaggle_environments
  interpreter wraps the entire turn in a try/except that re-raises,
  collapsing all rewards to None). Lost ~25 min on a hung n=16 A/B
  and one full debug cycle. **Fix:** use `fate_p.step` and align the
  docstring (committed as 4f1fb2a). **Preventive:** when reading a
  docstring for a field name, also peek at the dataclass definition
  to confirm.

- `tag: ab-baseline-misread-as-regression` — read 4/16 = 25% for the
  orbital-fix variant as "regression" when in fact it's parity with
  the 4P self-play symmetric baseline (4 identical agents → focal
  wins exactly 25% by symmetry). Same misread for D1 drain at 6/16 =
  37.5% (POSSIBLE LIFT). Cost: nearly dismissed two viable features
  before noticing. **Fix:** for any 4P focal-vs-3xopp A/B, the
  null-hypothesis bar is **25% wins** (1/N seats), not 50%; use
  Wilson-lo vs 25% (not 50%) as the parity gate.

- `tag: pre-submit-bundle-shim-recipe-undocumented` — variant agent
  dirs (one-file shim with env-var `setdefault` + `from agents.baseline
  .main import agent`) cannot be passed directly to `bundle_agent.py`
  — the bundler only scans the agent dir and strips the import without
  inlining `baseline/main.py`. Recipe: bundle `agents/baseline` first
  to `submissions/baseline.py`, then awk-prepend the env-var
  `setdefault` block. This was rediscovered today; the previous session
  had also worked through it. **Fix:** document the recipe in
  `state/TOOLS.md` so the next variant doesn't re-discover it.

## 2026-05-20 (claude/review-skills-improvements-moKOR — cross-branch consolidation)

- `tag: inventory-as-categorical-summary-not-itemized` — first draft
  of the cross-branch tools registry (`state/TOOLS.md`) was
  category-grouped prose ("we have A/B harnesses", "we have
  diagnostics"). Required THREE PI nudges to surface the actual
  items: "have you mentioned ML competition branch?", "have you
  listed the A B testing tools and the single game diagnose tools?",
  "have you listed the validation and testing tools?". Each fix grew
  the doc by ~50 lines; net output is ~3× the first draft. **Root
  cause:** I summarize when PI wants enumeration. Categories aren't
  searchable; itemized tables are. **Fix (drafted, NOT promoted per
  PI):** when PI asks for an inventory / registry / catalog / tools
  list, default to itemized enumeration (one row per item, scannable
  table). Recorded in postmortem; not added to improvements.md.
- `tag: substrate-asset-discovery-filtered-to-recent-only` —
  initial 7-branch survey filtered to "recent" branches (commits in
  last ~5 days), excluded `claude/precision-physics-engine-ymJkA`
  (9 days old, holds the only guaranteed-landing inverse-intercept
  solver in the repo + live-published submission #52552139). Surfaced
  only when OyoYR-rebased's HANDOVER tier-split commit (84 min ago)
  cited it. **Root cause:** "recent" filter is appropriate for ACTIVE
  work, not for substrate-asset discovery. **Fix (drafted, NOT
  promoted per PI):** query all `claude/*` branches when discovering
  reusable substrate; do not filter by recency.
- `tag: rule-number-collision-without-cross-branch-grep` — authored
  Rule 41 ("pre-submit cross-branch coordination gate") without
  first running `git grep "Rule 4[1-9]"` across sibling branches.
  `claude/audit-workflow-performance-btjeK`'s knowledge-base already
  had a Rule 41 candidate ("confound-sweep before correlational
  conclusion"). Caught only when PI asked for latest updates on
  other branches; renumbered to 42-47 and adopted btjeK's as Rule 41.
  **Root cause:** authored new rule numbers without reading sibling
  branches' KB. The new Rule 44 (this session) would have caught
  this. **Fix (drafted, NOT promoted per PI):** grep all dev branches'
  KB + improvements.md for pre-existing proposals before authoring a
  new CLAUDE.md rule number.

## 2026-05-18 (claude/reverse-engineer-seat-geometry-BPJKs)

- `tag: wrong-file-recon-skipped-state-md` — recon for an audit-driven
  chooser change started at `data/main.py` (60-line Kaggle starter
  example, unchanged since 2026-05-01 kickoff) instead of `agents/
  baseline/` (the modular v15 re-baseline `state/current.md` names as
  our submission). Spent two rounds analysing the wrong agent and
  proposing constant-bump fixes that didn't map to anything we ship.
  PI caught: "is that really our submission? check again." **Root
  cause:** didn't read `state/current.md` before forming a code-change
  recommendation about modifying "our agent." Same shape as
  `agent-introspection-skipped-bootstrap` (2026-05-13) and
  `fix-not-validated-against-real-failing-state` (2026-05-14): jumped
  to a code mental-model without reading the state docs that index
  what's actually in tree. **Fix:** before proposing edits to "our
  agent," `cat state/current.md` and confirm the file path. Promotion
  candidate (sees 3+ recurrences).

- `tag: crn-symmetry-broken-without-reading-prior-audits` — designed
  asymmetric chooser change: `top_tier_mirror_policy` (aggressive
  Tier-1) in `build_idle_baseline`, kept `lite_greedy_policy` (passive)
  in `score_action`. Panel result: 0 wins / 32 games, Wilson 0.00-0.11,
  decisive FAIL. Burned ~30 min of compute + one full panel slot.
  Reverted (commit `f28c9fc`). **Root cause:** the chooser's Δ requires
  common-random-numbers symmetry — both legs of `leaf(action) -
  baseline` must use the SAME opp trajectory. The audit trail at
  `audit/2026-05-17-state-function-principled-fix-results.md` documents
  the v11 → v12 → v13 progression that fixed exactly this asymmetric-Δ
  failure mode. I diagnosed the audit signal correctly (lite_greedy
  too passive vs real top-10) but chose a remedy that reintroduced the
  v11 bug the team had paid to fix. **Fix:** before proposing chooser-
  internal edits, grep `audit/` for the last 30 days of chooser /
  opp_model / baseline notes (`grep -l "chooser\|opp_model\|baseline"
  audit/2026-05-*.md`). Methodologically correct fix here is symmetric
  stronger opp (vectorise `top_tier_mirror_policy` or train Tier-2
  logreg) — pending. Same shape as `wrong-file-recon-skipped-state-md`
  above: code-change before reading state. Promotion candidate.



- `tag: restriction-tuning-before-modeling-fix` — when a failure
  mode admits both a constant bump (MAX_*/MIN_*/threshold) and a
  modeling fix (better opp/leaf/prediction), my default was to
  propose the bump. PI re-articulated 3+ times this branch
  (MAX_WAIT, MAX_HORIZON, MIN_FLEET_SIZE). **Fix:** promoted to
  CLAUDE.md Rule 40 (prefer modeling-correctness over restriction-
  tuning).
- `tag: stop-hook-pressure-commits-speculative-WIP` — Stop-hook
  warned on every uncommitted-changes turn; pressed me to commit
  lite_greedy-neutral-fix before the v7_0 panel finished. Panel
  showed Wlo 0.700 → 0.483; reverted (1c5e059). **Fix:** when a
  change is being verified (panel/tests running), use `git stash`
  to silence the hook without committing speculative work. Added
  to kaggle-comp/improvements.md pending list.

## 2026-05-16 (claude/recover-main-foundations-MV0e2 — v20 session)

- `tag: panel-pass-without-h2h-vs-current` — v17 and v18 both
  panel-PASSED against the legacy panel (v7_0, v4_planner, v3.5.1)
  but FAILED head-to-head vs v15 (40.6%, 34.4%). I burned 30+ min
  per panel run before checking the h2h vs same-family current
  agent. **Fix:** make h2h vs current submitted agent the FIRST
  gate (~15 min n=16 triage), not the LAST. Panel is a smoke test
  for "not catastrophically broken," not a quality test. Promotion
  candidate for `.claude/skills/kaggle-comp/improvements.md`.
- `tag: value-function-change-without-calibration-baseline` — v15's
  F2 = `(my_prod − opp_prod) × pv(500)` over-credits both sides
  equally; the difference cancels the over-credit. Three F2-axis
  changes (v16/17/18) shrunk F2 magnitude per planet → F1 ship
  balance over-weighted → chooser became more conservative.
  Never measured v15's F1:F2 ratio before proposing changes.
  **Fix:** before any value-function change, run a single game
  with v15 and log per-turn F1, F2, and F2/F1 ratio. Establish
  the calibration baseline. Same applies to baseline rollout
  changes (v19: me-policy made baseline "too good", chooser
  emit-rate collapsed).
- `tag: explicit-rewrite-of-implicit-behavior` — PI's "asymmetric
  reach × defend" was ALREADY encoded in v15: rollout's reactive
  opp catches fragile captures → leaf shows opp owning → my F2
  drops via owner-flip. My explicit `_favor` rewrites were
  REDUNDANT with this implicit handling AND broke F2 calibration.
  **Fix:** before adding an explicit term, search the existing
  agent for whether the rollout, the cheap-rank, or some other
  pre-existing structure already encodes the behavior. Trust
  "the simulator IS the value function" — if the simulator is
  correct, value function changes are usually wrong.
- `tag: sequential-falsification-across-axes-no-stopping-rule` —
  Burned through v16 (F2 multiplier) → v17 (F2 hold-cap) → v18
  (F2 prop-split) → v19 (baseline me-policy) all in one session.
  Rule 37 caps at 3 on SAME axis but I jumped axes when 3 failed,
  rationalizing each pivot as principled. The actual signal was
  "stop and dig" not "try a different lever." **Fix:** when 3+
  variants on any axis fail head-to-head, the next move is
  instrumentation + write-up, NOT another axis. Promotion
  candidate.
- `tag: kaggle-cli-auth-needs-fresh-bootstrap-source` — `kaggle
  competitions submit` returned auth error in main Bash session
  even though session-start hook reported credentials OK.
  KAGGLE_USERNAME/KAGGLE_KEY env vars aren't inherited across
  Bash tool subprocesses; bootstrap.sh creates ~/.kaggle/kaggle.json
  but the file wasn't present (rwxr-xr-x but no kaggle.json
  inside). **Fix:** always wrap kaggle commands as
  `bash -c 'source bootstrap.sh > /dev/null 2>&1 && kaggle …'`
  to ensure credentials are mounted into the subprocess.


## 2026-05-15 (claude/bootstrap-read-handover-HjcdN — copycat branch)

- `tag: pv-broadpool-incompatible` — Phase 3 copycat with
  `PV_GAMMA=0.99` + broad-pool argmax (geo tilts + v7_0_drop_one)
  regressed to 12/32 = 37.5% vs v7_0_drop_one (FAIL Wlo=0.23) after
  the no-PV broad-pool was 50% n=8 and the prior σ-pair config was
  57.8% n=64. PV-aware proposers favour early captures; geo's
  concentrated/saturation tilts favour different shapes; the
  `delta_us_minus_them` judge can't reconcile. **Fix:** the PV lever
  belongs with a focused proposer (v7_pv = v7_0_drop_one + PV); do
  not stack it on a broad enumerator.
- `tag: same-process-pv-shared-state` — testing PV vs non-PV by
  running both agents in the SAME Python process is a false A/B:
  `lib.scoring.PV_GAMMA` is a module-level constant set once at
  import, so whichever agent triggers the import first wins for both.
  My in-process diagnostic of "PV-copycat vs vanilla v7_0_drop_one"
  was actually "PV-copycat vs PV-v7_0_drop_one." **Fix:** always
  run cross-config A/Bs through `fast.py eval` (separate workers, env
  inherited per process); never trust same-process numbers when
  agents need different env vars.
- `tag: wallclock-truncation-in-roster-wrappers` — wrapping
  `lib.v7_search.choose(K=10, wallclock_ms=350)` inside a copycat
  roster member to leave budget for outer scoring truncated v7's
  drop-one search badly enough to lose 8/32 = 25% (Panel #2). Bumped
  to 550 ms in commit `50a0a3e` and recovered to 57.8%. **Fix:** if
  you wrap a strong K=N chooser as a roster candidate, give it the
  FULL ladder budget (700 ms) and trim outer cost elsewhere; or skip
  re-scoring when there's only one candidate.
- `tag: small-n-ab-noise-misled-panel` — saw 5/8 = 62.5% on a
  Phase-3 PV smoke and immediately escalated to a 70-min full panel.
  Wilson 95% CI on 5/8 is roughly [0.30, 0.86]; the panel landed at
  12/32 = 37.5%. False confidence cost ~70 min. **Fix:** require
  n≥16 (or Wilson Whi-Wlo width < 0.40) before promoting a smoke
  to a full panel.
- `tag: worktree-signing-fails` — committing on a `git worktree`
  (used to isolate the behavioral-mimic branch from the running
  copycat panel) fails with `signing server returned 400 missing
  source`. The Anthropic commit signer expects the standard repo
  layout, not the worktree's pointer-`.git`. **Workaround:** after
  the panel finishes, switch the main checkout to the new branch
  and commit from there. **Fix forward:** investigate signer's
  source-discovery; configure it to accept worktrees.

## Still-open patterns (next-session priority)

- `tag: handover-stale-at-session-start-no-git-log-check` — Rule 32
  already requires session-start `git fetch + git log HEAD..origin/main`.
  Enforcement is aspirational. **Promotion candidate:** SessionStart
  hook (the `session-start-hook` skill exists in this environment).
  Cost evidence: 5/13 LATE wrote a full plan-mode design for work
  already completed on the same branch (`cb02fd9`, `4ba55f4`).
- `tag: jax-vmap-already-wired` — claimed integration was missing
  while it was actually live. **Pattern lesson:** before claiming a
  capability is unbuilt, grep `agents/*_v*_*/main.py` and
  `lib/*.py` for existing wrappers. Specifically applies to
  `score_candidate_jax_pure_jit` (6 ms after JIT).
- `tag: geo-v2-three-failed-wallclock-fixes` — three orthogonal
  attempts to bound the K=10 lookahead's max-wallclock all regressed
  strategy more than they saved time. **Promotion candidate:** when
  a single-knob change costs more than it saves in three orthogonal
  directions, the config IS the local optimum — stop tuning;
  submit if positive, find structurally different lever otherwise.
- `tag: env-clone-cost-grows-with-history` — `env.clone()` cost rises
  4× across an episode (5.6 ms cold → 22 ms warm) because
  `Environment.clone()` walks `self.steps`. Mid/end-game rollout
  cost is hosed; `lib/fast_sim.py` is the working bypass.
- `tag: trueskill-noise-vs-signal` — TrueSkill σ is large for the
  first ~24 h after submit (initial σ≈300, shrinks ∝ 1/√N). Wait
  ≥24 h before reading rank delta into strategy decisions.
- `tag: state-files-claim-current-champion-with-stale-mu` — 2026-05-17
  baseline session: `state/current.md` claimed v7_pv (μ=1064.4) was
  team peak while live Kaggle had v15 (μ=1115.5) as the rolling champion.
  Two days of intervening sessions (v8_scavenge → v9 → v12 → v13 → v15
  → v20) shipped without state/current.md being refreshed. **Root
  cause:** state files recorded μ values which drift, and the
  refresh-state-files step was skipped by the wrap-up of the
  intervening sessions. **Fix landed this session:** state/current.md
  no longer records μ values at all (top-of-file note plus
  rolling-last-2 entries hold only submission IDs + dates + statuses).
  Rule 32 already mandates session-start `kaggle competitions
  submissions orbit-wars`; with no μ in the file, there's nothing to
  go stale. Same convention applied to HANDOVER.md and
  state/mechanism-ledger.md.

## How to add an entry

```
- `tag: <kebab-slug>` — <session context>: <what happened>.
  <Root cause>. **Fix:** <concrete action>.
```

Reuse tags. New tags get one cycle of grace before promotion. If a
tag fires 3+ times, it goes to
`.claude/skills/kaggle-comp/improvements.md` and then into the
relevant skill file or source code, not back into friction.md.

## 2026-05-16 (claude/review-foundations-progress-14HXp — v13/v14/v15 chooser saturation)

- `tag: panel-misleads-head-to-head` (4th recurrence) — v13's
  hybrid-policy panel showed 75 → 94% vs v3.5.1; head-to-head vs
  v12 was 47%. v14 maximin panel similar, h2h 50%. v15 Iter 3
  reactive opp panel preserved, h2h 45%. **Root cause:** panel
  measures vs ONE opponent class at a time; ladder is a mixture
  AND a same-family agent (v12) plays moves the panel doesn't.
  When opp model in opp_traj matches a panel opponent's pipeline,
  panel gain is panel-specific overfitting that doesn't transfer.
  **Fix:** require head-to-head Wlo>0.50 vs the same-family agent
  (v12) at n≥32 as a hard gate before submission. Panel is
  necessary but NOT sufficient. 4× fired ⇒ promotion candidate.
  Promote to `.claude/skills/kaggle-comp/improvements.md` as a
  rule: panel without v12-h2h gate is incomplete.
- `tag: crn-cancellation-blunts-leaf-scorer-features` — Adding F4
  (vulnerability penalty) to `_favor` regressed Felipe / Naoism
  / head-to-head across 3 threat-formulation variants. **Root
  cause:** opp_traj is replayed identically in baseline + every
  candidate (CRN variance reduction). Same threatened planets
  appear in both leaves → F4 discount applies equally → cancels
  in Δ. F4 only differentiates via second-order effects (our
  launch depletes a source → source vulnerable in candidate but
  not baseline) — which is net-NEGATIVE (punishes aggressive
  plays). **Fix:** leaf-scorer modifications need to evaluate
  via h2h vs v12, not via lift on panel. Hand-crafted features
  on top of v9 `_favor` are unlikely to lift — CRN invariance
  is a structural barrier. Path forward: learned value head
  (replaces scorer entirely) or empirical loss-pattern analysis.
- `tag: agent-exception-swallowed-by-kaggle-env` — v15 Iter 3
  maximin code referenced `t_agent_start` without setting it
  at agent() entry. Silently NameError'd in 2P games where
  short_list >= 2 → agent returned [] → games lost. Diagnostic
  signal: turn-ms p95 = 13ms (way below normal 50-150ms) =
  agent crashing early. **Root cause:** kaggle_environments
  catches all agent exceptions and treats them as the agent
  returning nothing. Errors don't surface in fast.py output.
  **Fix:** when adding new code paths into agent(), smoke-test
  by inspecting turn-ms — anything < 30ms p95 means the agent
  is short-circuiting (crashing OR returning [] for a non-
  trivial reason). Inspect for silent exception swallowing.
- `tag: dogpile-overestimates-without-reactive-opp` — v15 Iter 2
  joint candidates (multi-source → single target) regressed
  head-to-head 28-31% vs v12, both raw and with opp-cost filter.
  **Root cause:** joint Δ at horizon K assumes opp_traj built
  once at turn start — opp doesn't react to our dogpile, so
  leaf state shows us "owning" a far-away hard-to-defend capture
  without accounting for opp's counter-attack. Δ over-estimates
  joint value. **Fix:** action-space expansion (dogpile,
  coordinated multi-target) needs reactive opp model FIRST.
  Cross-iteration learning: the three diagnosed root causes
  (scorer/action/opp) are NOT independent — they compose via
  the K-step fixed-opp-rollout invariance. Order of fix
  attempts matters: opp reactivity must come before action-
  space expansion.
- `tag: chooser-family-structural-saturation` — Three iterations
  (F4×3, dogpile×2, reactive-step-0×1) across all three
  diagnosed root cause axes. Best result: parity (45%). None
  lifted head-to-head vs v12. **Empirical conclusion:**
  v9-family chooser (candidate enumeration + _favor + opp_traj
  + K-step rollout) is structurally saturated at μ~1120. Surface
  modifications cannot break the ceiling. **Fix:** future
  sessions should not iterate on v9-family components without
  first running an empirical loss-pattern analysis (path A) or
  pivoting to a learned value head (path B) or different
  chooser family (path C). Promote: add to standard practice —
  any new chooser variant must beat v12 h2h at n≥32 before
  expecting ladder lift. The 7 iterations this session are a
  cautionary tale.
- `tag: early-trueskill-mu-unreliable` — v12's ladder score
  settled from 1217.7 → 1099.3 as more games accumulated.
  The +97μ "huge gain" was an early-window low-sample
  artifact. Caused us to over-estimate v12 → over-estimate
  v13/v14/v15 expected lift. **Fix:** wait 6h+ post-submit
  before basing strategic decisions on a new submission's μ.
  TrueSkill needs ~50+ games to converge; first 10 games can
  be off by ±80μ. Document in WRAPUP that the team-floor
  calculation uses SETTLED μ, not first-read μ.

## 2026-05-17 (claude/audit-workflow-performance-btjeK)

- `tag: kaggle-cli-401-in-followup-shells` — diagnostic
  session: bootstrap.sh exports `KAGGLE_API_TOKEN` inside
  its own shell only; every subsequent Bash tool call comes
  up with no token and the real CLI 401s. KGAT_-prefix
  tokens cannot live in `~/.kaggle/kaggle.json` (legacy
  32-hex auth path returns 401 — verified empirically this
  session). Bootstrap's "skip kaggle.json for KGAT_" branch
  is correct but leaves no env-persistence path. **Fix:**
  SessionStart hook installs `$HOME/.local/bin/kaggle` shim
  that re-derives `KAGGLE_API_TOKEN` from harness var
  `$KaggleAPIToke` on every invocation. Verified in fresh
  shell; survives across Bash tool calls.
- `tag: vanished-in-space-dominates-trajectory-waste` —
  replay-mine of v15's 92 live games shows `vanished_in_space`
  at 838 fleets (8.8%, 18% ships-weighted) vs sun-death at
  13 (0.14%). Sun was the salient PI hypothesis; the real
  ship-leak is whatever causes mid-flight vanish (comet
  collision suspected). **Fix:** before sun-fix (pivot #5)
  is worthwhile, identify the vanish mechanism — it's the
  dominant ship-waste category by ~6x.
- `tag: composite-head-2p-only-no-4p-opp-aggregation` —
  wiring `lib.value_heads.composite_capture_value` into
  `agents/baseline/value.favor_composite`: composite
  collapses all non-me planets into one "enemy" bucket.
  In 4P this loses the `favor()` sum-of-opps signal that
  "capturing a weak opp is full credit." **Fix:** opt-in
  via `BASELINE_VALUE_HEAD=composite` env var (default =
  `favor`). Flag filed at `knowledge-base/flags/
  2026-05-17-composite-value-head-2p-only.md`. Do NOT
  submit composite-default agent without 4P-aware variant.

## 2026-05-17 (claude/improve-fleet-efficiency-cQXg4 — 7 variants falsified)

- `tag: pattern-overlay-on-tuned-baseline-doesnt-lift` (3rd recurrence)
  — built 7 variants across 2 axes (chooser filters v21/v21_a/v21_ae/
  v21_solo, rollout opp v22, opening overlay v23 at two windows). All
  fail at n=32 vs v15 (range 15.6 % – 31.2 %). v15's chooser, leaf,
  opp model, and emit dedup are co-tuned end-to-end; any single-component
  modification breaks calibration in some other dimension. **Fix:**
  promote to kaggle-comp/improvements.md — refuse to plan single-
  component modifications on v15. Either wholesale chooser-family
  replacement (different value head AND different proposer AND different
  chooser) or no change. Full archaeology on the branch plus
  `audit/2026-05-17-fleet-efficiency-negative-result.md`.
- `tag: launch-rate-is-symptom-not-cause` — replay analysis showed v15
  launches 2 fleets in turns 0-15 vs top-10's 7-10. v23 tried to close
  the gap with `propose_opening_missions` short-circuit. Live: 15.6 %
  vs v15 — worse. Transplanting a behavioural pattern from top-10 replays
  without their surrounding stack regresses 25-35 pp. **Fix:** before
  injecting a behavioural pattern from replays, construct a controlled
  test (would v15 with that pattern alone lift?); if you can't, the
  pattern can't be transplanted.
- `tag: n16-falsely-shows-parity` (recurrence of `small-n-ab-noise-
  misled-panel` from 2026-05-15). v21 at n=16 = 8/16 = 50.0 % (Wlo=0.28)
  read as parity; same agent at n=32 = 10/32 = 31.2 % Wlo=0.18 = clear
  FAIL. Wilson CI width at n=16 is ≈ 0.45 — cannot distinguish parity
  from a 20 pp regression. Burned 4 single-axis ablations at n=16
  before the n=32 reveal. **Fix:** for any submission-gating decision,
  n=32 minimum. n=16 is for smoke only ("agent doesn't crash"), not
  for verdicts. Promotion candidate: bump `fast.py eval` default
  `--max-seeds` from 8 (= n=16 with 2-seat balance) to 16 (= n=32).

## 2026-05-17 (claude/space-fleet-physics-engine-lrLE6 — v8_analytic value-head pivot to fast_sim)

- `tag: K-shorter-than-launch-eta-makes-value-head-blind` — JAX K=8
  rollout couldn't distinguish 38 of 40 candidate atoms from no-op
  (delta = 0.0000 exactly) at seed 1 turn 80. Root cause: in-flight
  fleets count as `my_ships` in the leaf, so any launch with ETA > K
  produces a bit-identical leaf state. Median launch ETA = 10-30
  turns; K=8 catches almost nothing. All prior tuning operated on the
  candidate pool BEFORE the leaf or on opp simulation INSIDE the K
  window — never touched leaf representation. **Fix:** new value
  head must either (a) extend K past the median launch ETA, or (b)
  credit in-flight aimed fleets with their expected production gain
  (cheap-rank's pv_horizon). Diagnosed via /tmp/micro_trace.py;
  fixed via fast_sim+lite_greedy pivot at branch commit `7e511a0`
  (K=15). Architecture concept doc:
  `knowledge-base/concepts/v8-analytic-architecture-state.md`.
- `tag: copy-K-from-jax-budget-to-fastsim` — initial fast_sim port
  inherited K=8 from the JAX-budget-constrained config. Predictable
  consequence: same horizon-too-short failure mode as JAX, manifested
  as 2/8 vs nearest (REGRESSED from JAX baseline 4/8). Per-step cost
  on fast_sim is ~10-30× cheaper than JAX, so K=15-25 is actually
  affordable. **Fix:** when porting an algorithm between cost
  regimes, re-derive budget-derived constants (K, batch size, cap)
  from the new regime's per-call timing; never copy from the old one.
- `tag: rule-37-strict-kill-vs-pi-override` — plan-mode WRAPUP gate
  was "Wilson 95% LB < 40% vs nearest → kill". 4/8 wins gives LB=21.5%
  which triggers kill, but n=8 is too small for the LB to clear 40%
  unless ≥7/8 wins. PI overrode the strict-read gate with "we don't
  need to win, we need to know if architecture is buildable-on" —
  reframed verdict from outcome-based to substrate-viability-based.
  **Fix (promotion candidate):** dual-gate kill-or-keep at small n
  with both Wilson LB and substrate-viability checks (knob
  responsiveness, predicted-outcome-matched, timing headroom).


## 2026-05-31 (claude/competition-objective-alignment-hqNVM — Tier 2 opp-model falsified for chooser-budget reasons)

- `tag: chooser-probe-fix-overcorrected` — `affordable_validate_cap`
  probe upgrade (commit 05aa624) captured Tier-2 opp cost in
  `per_step_ms` correctly, but the chooser's downstream formula
  `per_cand_ms = (per_step × avg_K=32.5 + per_leaf) × 1.5_safety`
  multiplies by *max* horizon, not the actual `prop_horizon` (~5-15)
  used in `score_candidate_v4`. ~5× over-estimate of per-candidate
  cost → `safe_deadline = deadline - 344 ms` pre-bailed validation
  after ~3-5 candidates → `scored` list empty → focal emitted nothing
  most turns → 0/32 vs launch_rules_universal. **Fix:** reverted
  the probe in commit 7e8c5dc; the proper fix needs `avg_K` to use
  median prerank `prop_horizon` (or per-tier safety multiplier).
  Deferred until a chooser-quality session takes this on.

- `tag: chooser-tuned-for-tier0-only` — heavier opp models (Tier 1
  `top_tier_mirror`, Tier 2 `trained_logreg`) cost ~5-6 ms/call vs
  Tier 0 `lite_greedy`'s ~0.5 ms/call. In a 1000 ms chooser budget
  per turn, Tier 0 validates ~1200 candidates per turn, Tier 1/2
  ~150-200 — an 8× drop. Win rate vs `launch_rules` collapses:
  pv_eta (Tier 0) = 56% (n=16); pv_eta + Tier 2 = 6% (n=16). Any
  future RL/IL/distilled opp model > ~1 ms/call hits the same wall.
  **Fix:** structural (PI ideas captured in
  `knowledge-base/thoughts/2026-05-31-tier2-root-cause-and-pi-ideas.md`)
  — event-driven rollout horizon (3 strategic events instead of 10
  ticks) OR a tiny distilled opp model that runs at lite_greedy
  speed (~0.5 ms) trained from top-leaderboard Kaggle replays.

- `tag: monitor-spam-on-killed-task` — my Monitor poll script
  `until both A/Bs have wins=` loop kept re-printing the COMPLETED
  Tier 0 result every 30 s for the full 30-min timeout, because the
  Tier 1 A/B was killed by the harness's own 25-min timeout and
  never wrote a "wins=" line. The exit predicate required both
  logs to have terminal lines; one was killed silently. Spammed
  ~10 system-reminder notifications into the chat after the result
  was already shared. **Fix:** monitor exit predicates must include
  "process is gone AND file is stable for N seconds", not just
  "matching line appears". Logged once-only outputs per file via
  a sentinel-file pattern, not re-emit-on-poll.
- `tag: <kebab-slug>` — <session context>: <what happened>.
  <Root cause>. **Fix:** <concrete action>.
```

Reuse tags. New tags get one cycle of grace before promotion. If a
tag fires 3+ times, it goes to
`.claude/skills/kaggle-comp/improvements.md` and then into the
relevant skill file or source code, not back into friction.md.

## Anti-spam — what does NOT belong here

- Successful experiments → `audit/YYYY-MM-DD-*.md`.
- LB / rank results → `state/calibration-ladder.md`.
- Hypothesis churn → `state/hypothesis-board.md`.
- Multi-paragraph reasoning → audit postmortem or
  `knowledge-base/thoughts/`.

If something is worth a paragraph, it's not friction. It's a real
postmortem.
