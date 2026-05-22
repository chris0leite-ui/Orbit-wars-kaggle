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

- `tag: bundle-agent-doesnt-inline-from-baseline-main` — 2026-05-21
  PM (claude/strategy-axis-decision-3437): `scripts/bundle_agent.py`
  comments out `from agents.baseline.main import agent` inside thin
  wrapper `main.py` files but never inlines the target function body.
  Result: bundles built from any `agents/<name>/main.py` whose only
  effective code is env-var-set + `from agents.baseline.main import
  agent` have NO `agent` symbol in the output. kaggle_environments'
  loader (`agent.py:64`) falls back to "last callable in module
  namespace" — for the consolidated bundle that's `opening_plan(world,
  model, my_id, num_seats)`, signature mismatch, game ERRORs at step
  0. Submitted file 52882014 worked on Kaggle (μ=1124) so an older
  bundler version did inline it; the local re-bundle silently broke.
  Cost evidence this cycle: last session's "n=8 vs LATEST: 8W/0L,
  Wilson [0.658, 1.000]" was Phase 4 beating an ERROR-on-step-0
  bundle, NOT vs ladder-leader-strength. Decision to submit FND was
  taken on falsified evidence. **Detection runtime:** `python3 -c
  "from kaggle_environments import make; env=make('orbit_wars',
  debug=True); env.run(['submissions/X.py', lambda o,c=None: {'actions':[]}])"`
  → `RAISED: opening_plan() missing 2 required positional arguments`.
  **Fix immediate:** rebuilt `submissions/baseline_joint_aggr_consolidated_orbitfix.py`
  by prepending env-var block to `submissions/baseline.py` (which has
  the `agent` body inlined correctly). **Fix forward:** add a
  post-bundle assertion `hasattr(module, 'agent')` to the parity gate
  in `scripts/bundle_agent.py`; refuse to leave bundles without
  `agent`. Companion improvement: when the entry main.py contains
  `from agents.X.main import agent`, inline X's `agent` function body
  rather than commenting the line out.

- `tag: kaggle-mu-treated-as-final-not-snapshot` — 2026-05-21 (Rule
  43 already codified end of last session; firing again early this
  session). Said "two submissions at μ=1063.9 and μ=1059.0" without
  re-pulling — values had drifted to 1058.2 and 1055.0 by the time I
  needed them. Same pattern as the three drifts logged in Rule 43's
  origin note. **Fix:** Rule 43 already enforces re-pull at session
  start AND before any push decision; the gap is muscle-memory.
  Promoted from candidate to enforced rule end of last session;
  re-firing means the read-from-doc habit is sticky. Mitigation
  this session: re-pulled live μ before drafting this entry.

- `tag: header-comment-misled-static-analysis` — 2026-05-21 PM
  (claude/strategy-axis-decision-3437): asserted "FND bundle's agent
  is from agents/baseline, doesn't call Phase 4" based on the bundle's
  HEADER line `# Bundled by scripts/bundle_agent.py from agents/baseline
  + lib/...`. Header was a stale bundler artifact; the actual `agent()`
  at line 18236 of the bundle is a post-bundle-appended analytical
  pipeline (`_AGENT = compose(decision=decision_outcome_aware_milp,
  ...)`) that DOES call `solve_outcome_aware` → `_endgame_bonus` every
  turn. PI caught the wrong conclusion ("review carefully that it is
  actually true what you say"); runtime trap (insert `raise
  RuntimeError("PHASE4_*_CALLED")` at Phase 4 entry points and play
  one game) proved Phase 4 IS live. **Root cause:** static reasoning
  from header text instead of from the function body. **Fix:** when
  claiming a code path is dead/live, the verification step is a
  RUNTIME trap (raise on entry, run one game, observe), not a grep
  for imports. Generalises Rule 38 (fix-verification reproduces
  failure state) to code-path claims (dead-code claims reproduce
  via runtime).

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

## 2026-05-16 (claude/recover-main-foundations-MV0e2 — v13 session)

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

## 2026-05-18 (claude/audit-workflow-performance-btjeK — submission 52784853 ladder regression)

- `tag: kaggle-mu-does-not-settle-stop-saying-it-does` — repeated
  PI correction across multiple sessions: Kaggle TrueSkill μ
  DRIFTS continuously as ladder games are played. I keep writing
  "settled μ=X" in wrap-up docs (HANDOVER.md, state/current.md,
  audit notes, even chat responses). Every μ value reported is a
  SNAPSHOT at the moment of `kaggle competitions submissions
  orbit-wars` query, not a final value. PI has corrected this in
  prior sessions; the pattern recurred this session in 6+ places.
  **Fix:** scrub "settled" / "settle" wording from all Orbit Wars
  artifacts. Use "snapshot" or "current" with explicit timestamp.
  Promotion candidate if it fires again: add a session-start hook
  warning, OR a pre-commit hook that rejects "settled" near a
  μ value in Orbit Wars files.



- `tag: local-ab-vs-ladder-calibration-miss-30mu` — submission
  `52784853` (PV off + bug #3/#4/#12) local A/B vs prior bundle was
  26/32 = 81.2% (Wlo=0.647) PASS — predicted settled μ 1130-1160.
  **Actual settled: 1083.1**, ≈30μ BELOW prediction and ≈30μ below
  the 1113.4 it replaced. Repeats the recurring local-vs-live
  calibration miss documented in `state/current.md::Calibration
  WARNING` (multiple -20 to -30pp gaps on recent submissions). The
  bundle-as-baseline doesn't represent the LB opponent distribution
  the new submission actually fights. **Fix:** before any future
  submission push, run a 3-opponent panel (`fast.py eval --vs-panel
  --require-h2h <current-floor>`) AND a 4P sub-panel; only push if
  ALL FOUR gates clear (oracles + bench + h2h + 4P). The
  bundle-only A/B is sufficient for "did we structurally regress"
  but NOT for "will this lift the ladder."

## 2026-05-18 (claude/audit-workflow-performance-btjeK — bug #15 v1/v2 + #14 option 5)

- `tag: wrong-root-cause-from-symptom-similarity` — bug #15 v1 (commit
  466fc98) added BOTH a production-PV term in composite's base AND a
  per-fleet counterfactual capture credit. After the A/B regressed
  (40.6% vs bundle 50%), I told PI the root cause was "double-counting
  per capture." The "drop the per-fleet credit, keep PV" v2 (b285882)
  was supposed to fix this — but A/B still failed at exactly the same
  39.6%. Then bug #14 option 5 (smart reactive defense) was supposed
  to cure the v2 regression — and also failed at 39.6%. The actual
  root cause: the PV term ITSELF over-credits because the chooser
  was calibrated WITHOUT it. Adding any positive capture signal of
  that magnitude inflates all candidate scores → over-emission → drained
  sources. **Fix:** before proposing a structural fix for a regression,
  run the kill-switch FIRST — `COMPOSITE_PRODUCTION_PV=0` would have
  isolated the PV term's contribution in one A/B, before the 3-cycle
  v1→v2→v3 investigation.

- `tag: stateless-policy-in-rollout-cannot-converge` — bug #14 option 5
  v1 was stateless per rollout tick. It ignored already-in-flight
  friendly reinforces when computing `garrison_at_eta`, so every tick
  re-emitted a fresh reinforce against the same threat. By tick 5 we
  had stacked 5 redundant reinforces, draining the sister and bloating
  the fleet count. A/B: 15.6% (max wallclock 8252ms). **Fix:** include
  in-flight friendly ships in the garrison math (idempotency contract);
  test pinned at `test_idempotency_inbound_friendly_counts_toward_garrison`.

- `tag: per-tick-rollout-policy-blows-wallclock` — even with the
  idempotency bug fixed, calling the defensive policy per-candidate
  per-rollout-tick (5000 calls/turn for N_VALIDATE=200, horizon=25)
  added ~1.5s to median turn cost — bench max 1492ms with 10
  >1000ms outliers. **Fix:** precompute the policy emits ONCE per
  candidate (tick-0 obs) and merge with the candidate's launch at
  `wait_N`. Models "all this turn's chooser moves emit together."
  Bench dropped to 685ms max, zero outliers.

- `tag: planet-collision-in-test-setup` — `test_oracle_cleanup_capture_last_opp_planet`
  builds a circle of 23 OUR planets around the sun, then adds the
  opp planet at angle 0 — coincidentally the SAME position as P0.
  Ray-cast picks one or the other, proposer emits 0 candidates,
  test fails. Discovered while tracing why option 5 didn't flip
  the xfail. **Fix:** offset the opp planet by a few units (still
  TODO — task parked for next session).

- `tag: rollout-auto-defense-in-baseline-zeros-delta` — first cut of
  option 5 applied auto-defense in BOTH baseline and candidate
  rollouts. Baseline became too capable: it auto-defended threats,
  so candidates that ALSO defended (real reinforce launches) added
  no marginal value → Δ ≈ 0 → chooser refused to emit real defense.
  Working defense oracles regressed to FAIL. **Fix:** asymmetric on
  purpose — auto-defense in CANDIDATE rollouts only, baseline stays
  idle. Restored working oracles to PASS.

## 2026-05-17 (claude/audit-workflow-performance-btjeK)

- `tag: validate-cap-too-tight-cost-winrate-not-just-wallclock` —
  trajectory chooser wallclock fix session: mirrored composite's
  `affordable_validate_cap` blindly (N_VALIDATE=60 + min-with-n_aff
  cap). When per_cand_ms probed high, n_aff floored to 8 — chooser
  scored only 8 of ~200 candidates on heavy turns. Winrate dropped
  from 42/64 (65.6%) to 37/64 (57.8%). **Fix:** N_VALIDATE=200 +
  rely on `safe_deadline` pre-bail (already in the loop) for the
  real budget. Cap becomes a safety ceiling; deadline is the binder.
  Rule 40 applies: restriction-tuning (cap-the-cap) was the wrong
  lever; the modeling fix is "let the real budget bind, don't pre-
  estimate it conservatively."
- `tag: env-var-shared-process-breaks-ab-isolation` — spatial-leaf
  A/B session: tried to A/B `BASELINE_VALUE_HEAD=hybrid_spatial` vs
  reference clone with `os.environ["BASELINE_VALUE_HEAD"]="hybrid"`.
  fast.py loads both agents into the SAME worker process via
  `importlib.util.spec_from_file_location`. The two top-level
  modules are distinct, but the inner `agents.baseline.value`
  module is cached in `sys.modules` by name and shared. `os.environ`
  is process-wide. Whichever main.py loads SECOND overwrites the
  env, then both agents' per-turn `select_favor_fn()` reads the
  same value → both play with the same head. The A/B result is
  uninformative (~50%, not a real comparison). **Fix:** for A/B
  isolation, patch the BUNDLE's `select_favor_fn` to hard-code the
  head choice (bypass env entirely). Loaded bundles via
  `importlib.util.spec_from_file_location` ARE module-level isolated
  (unique module names). Env-based dispatch is fine for production
  (single agent per process) but breaks within-process A/B. Rule 38
  cousin: verify your A/B harness ACTUALLY tests two different
  configurations before trusting the winrate.
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
- `tag: vanished-in-space-was-classifier-bug-not-comets` —
  replay-mine of v15's 92 live games initially showed
  `vanished_in_space` at 838 fleets (8.8%). PI hypothesised
  comet collision; investigation revealed the existing
  `attribute_fleets:290` check used static `best_d < 5.0`
  from fleet OLD position to planet NEW position — orbital
  planets that moved >5 units between obs_prev and obs_vanish
  were misclassified as vanishes. **Fix:** replaced single-
  point distance with engine's `swept_pair_hit` primitive
  against every planet in obs_prev (with planet new positions
  from obs_vanish, and comet new positions from the comet
  path for expired-same-tick comets). Result: win 42.8 → 47.4%,
  defense 32.8 → 35.2%, waste_traj 9.0 → 0.9%, waste_comet
  = 0.1% (12/9507). PI's comet hypothesis falsified
  quantitatively; v15 was already cleaner than measured.
  Permanent measurement-honesty improvement.
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
- `tag: bundler-modular-agent-namespace-access-breaks-bundle` —
  submission 52744234 ERRORED in Kaggle validation with
  ImportError: 'attempted relative import with no known parent
  package'. Root cause: `agents/baseline/main.py` used
  `from agents.baseline import chooser, proposer` (modular pattern,
  new this session). The bundler's `_INTRA_IMPORT_RE` only stripped
  `from lib.X` and `from .X` lines — left the absolute
  `from agents.baseline.X` line in the bundle. Locally `agents/`
  was importable from cwd, so parity gate passed; Kaggle's flat
  filesystem has no `agents/` package and import resolution
  trampled through `kaggle_environments/.../lux_ai_s3/agents.py`,
  surfacing the misleading "relative import" error. **Fix:** extend
  bundler regex to match `from agents.<name>.X import …`, and add
  `_topo_sort_agent_submodules` to inline sibling .py files (value,
  proposer, chooser) before main.py. Switch main.py to single-line
  explicit-name imports (`from X.chooser import build_idle_baseline,
  …`) — multi-line parenthesised imports leak continuation lines
  past the per-line strip regex. Verified: parity 712 turns + cold-
  load test (repo dir stripped from sys.path) on the re-bundle.
  Re-submit 52744856.
- `tag: composite-head-wallclock-over-1000ms-on-heavy-turns`
  — composite A/B vs panel/peaks hit max turn-ms 1183/1292
  (env budget 1000ms). Root cause:
  `agents/baseline/chooser.affordable_validate_cap`
  (lines 78-90) probes per-step `fs_step` cost only — fine
  for `favor` (100 µs leaf) but undercounts
  `composite_capture_value` (~2-5 ms leaf — builds World
  + ray-casts every fleet) by ~95%. `N_VALIDATE` cap stays
  large, candidate budget overruns. **Fix:** probe one
  leaf eval too; per-candidate cost becomes
  `per_step_ms × avg_K + per_leaf_ms`. Caller signature
  picks up `me` and `gamma` (both already in scope at
  `chooser.choose:100`). Verified by re-running A/B after.
- `tag: predict-fleet-fate-sun-safety-cushion-false-rejects` —
  `lib/trajectory.predict_fleet_fate` checked
  `sun_d < SUN_RADIUS + 0.5` (the engine uses bare
  `sun_d < SUN_RADIUS`). The 0.5 cushion was filed
  2026-05-11 as "float drift" insurance but caused
  systematic false rejections in production code:
  `lib.mechanism.sun_avoid` dropped legal snipe/reinforce
  intents in `v3_snipe`; `agents/baseline/proposer.
  PROPOSER_TRAJECTORY_FILTER` A/B vs v15 stuck at 56.2pct
  (n=64). **Fix:** `SUN_SAFETY = 0` in `lib/trajectory.py`.
  Same A/B post-fix: 65.6pct (n=64), +9.4pp. Trajectory
  chooser v4 also showed the same pattern (filter ON
  vs OFF). Same friction class as `helper-reimplemented-
  inline-silently-wrong` (2026-05-14) — a near-correct
  re-implementation of an engine primitive diverges
  silently. **Promotion candidate (3rd recurrence of
  the helper-divergence pattern this comp):** when
  inlining or wrapping an engine primitive, MUST
  reproduce the engine's exact predicate, including
  any inequality direction or strict/non-strict bound,
  and add a parity test that exercises the boundary.

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

## 2026-05-19 (claude/strategy-framework-design-OyoYR-rebased — value-head axis falsified)

- `tag: full-panel-AB-before-single-game-evidence` — ran two 256-game
  4P FFA panels (favor vs projected, then favor vs projected_sum) at
  ~20 min each before checking whether the new value head changed the
  agent's behaviour on a single game. Single-game inspection
  afterwards showed favor and projected_sum produced **bit-identical
  trajectories** on the seeds I checked (seed 42 seat 0) for the
  stale baseline. **Root cause:** treating the panel as the first
  validation step instead of a confirmatory one. Cheap diagnostic
  (`python /tmp/inspect_one_4p.py --seed S --seat I --focal <bundle>`
  for 2-4 different seeds, diff the per-turn action streams) is ~3 min
  and reveals zero-behaviour-change cases before burning 40 min of
  panel compute. **Fix:** before any 256-game A/B, inspect 2-4 single
  games with `inspect_one_*.py` and confirm the two bundles produce
  different action streams. If trajectories are bit-identical on the
  sample, abort the panel — the change isn't expressive in this
  architecture and the panel is noise. PI-ratified 2026-05-19
  mid-session.

- `tag: value-head-permissiveness-greenlights-bad-proposer-candidates` —
  2026-05-19 (rebased): projected_rank_diff_sum on btjeK lost 22.6 pp
  to favor in 4P (42.2% vs 64.8%, no CI overlap). Deep-dive on
  seed=7 seat=1: projected_sum launched 2.3x more than favor (93 vs
  41) but only +3 captures; the extra 52 launches were 26 reinforces,
  18 OOB/sun, 4 bounces. **Root cause:** NEW failure mode (NOT CRN
  symmetry — that's the F4 / dogpile family). The per-seat
  `P_p × (T-step)` projection inflated the value of marginal
  candidate actions enough to clear the chooser's Δ-threshold; the
  proposer's bad candidates (fleets to OOB / fleets to our own planet)
  were no longer filtered by the value head. Favor's tighter
  F1+F2+A2 hybrid was doing real work rejecting them. Bonus:
  p95=1534 ms over the 1000 ms budget. **Fix:** stop iterating on
  leaf-side value heads under this proposer; investigate proposer-
  side candidate filtering (a7f9383 hold-feasibility filter is the
  reference direction). Documented in
  `audit/2026-05-19-value-head-axis-exhausted.md`.

- `tag: stale-branch-base-invalidates-local-AB` — 2026-05-19: ran 2P
  and 4P A/B tests of projected_rank_diff_sum vs `favor` on a branch
  `claude/strategy-framework-design-OyoYR` based on `origin/main
  d25e9d3` — but the LIVE submission is on `claude/audit-workflow-
  performance-btjeK`, which is ~80 commits ahead with bug fixes
  #3/#4/#11/#12, PV-off, A2-4P hybrid in `favor`, and the
  hold-feasibility filter. Local A/B was projected_sum_stale vs
  favor_stale and showed TIE (68% vs 66%). On the rebased branch
  (btjeK base + my variant), the same variant lost 22.6 pp.
  **Root cause:** no session-start check that the branch is current
  with where the live submission actually came from — Rule 32
  fetches origin/main but the relevant branch is the live-submission
  branch, not main. **Fix:** at session start when intending to
  improve a live submission, check `state/current.md` for the source
  branch of the live bundle and rebase / branch from THAT, not from
  origin/main. The branch-tip might be origin/main, but the live
  submission's parent is often a feature branch.

## 2026-05-20 PM (claude/strategy-framework-design-OyoYR-rebased — joint_solver Phases 1→5E shipped; 0/16 locally; μ=711.5 live)

- `tag: bundle-multi-line-imports-broken` — phase-5 submission prep:
  scripts/bundle_agent.py's regex import-stripper operates per-line;
  multi-line `from X import (a, b, c)` left orphan continuation
  lines, producing `SyntaxError: unmatched ')'` in the bundle. Found
  by smoke-testing the bundle. Root cause: per-line regex can't see
  multi-line parenthesised imports. **Fix:** all
  bundle-deployable lib/agents modules must use single-line imports
  (`from X import a, b, c`). Did the rewrite for lib/joint_solver/
  this session. Promotion candidate.
- `tag: bundle-aliased-imports-broken` — phase-5 submission prep:
  `from X import Y as Z` stripped by bundler leaves `Z` undefined
  at bundle runtime. Hit "name 'opening_plan' is not defined" and
  "name '_greedy_assign' is not defined" and "name 'fleet_speed'
  is not defined." Root cause: bundler strips the import line
  entirely; the alias rename is lost. **Fix:** avoid `as` aliases
  in bundle-path code; if needed, do `Z = Y` after the import.
  Promotion candidate.
- `tag: cross-agent-imports-not-bundled` — phase-5 submission:
  lib/joint_solver/* imports `from agents.baseline.*` (proposer,
  predicates, strategic_lp, migration_solver, chooser_trajectory).
  The bundler inlines THIS agent's siblings (`agents/<this>/*.py`)
  but not OTHER agents' files. Bundle had stripped imports
  without inlining helpers → "name '_greedy_assignment' is not
  defined". Workaround: based the analytical bundle on
  `submissions/baseline.py` (helpers already inlined) and
  appended joint_solver content. Root cause: bundler design only
  considers within-agent deps. **Fix:** move shared helpers to
  `lib/` OR extend bundler to inline cross-agent deps explicitly.
  Promotion candidate.
- `tag: tests-pass-bundle-broken` — phase-5 submission: 50/50
  unit tests pass; bundle had multiple silent failures (above
  three tags). Tests use full PYTHONPATH; bundle uses flat exec
  — they don't exercise the deployment path. Root cause: no CI
  step exercises the bundle. **Fix:** every PR that changes
  bundle-path code must run a bundle-loads-and-emits smoke.
- `tag: source-bundle-behavior-diverges` — phase-5 submission:
  same agent, same seed, source vs bundle launch counts differ by
  ±20 across 5 seeds (avg src=64, bundle=76 vs trajectory
  baseline). Both still lose 0-5, but the agents are
  demonstrably different. Likely cause: LP tie-breaking through
  different float paths when classes are inlined into one
  namespace vs imported from modules. **Fix:** add a
  bundle-parity check (fixed seed N-turn action equality) BEFORE
  any submission; bail if any divergence. Promotion candidate.
- `tag: local-AB-not-calibrated-to-live-ladder` — phase-4
  through 5E: 6 consecutive 0/16 n=8 A/Bs vs ONE opponent
  (trajectory baseline). Submitted at session end → live
  μ=711.5 (baseline was 1122). Local A/B told us the BINARY
  outcome (lose) but not the SEVERITY — we expected μ ≈ 800-1000;
  reality was near-random. Root cause: single-opponent local panel
  is not representative of live ladder distribution. **Fix:**
  every A/B harness must test vs ≥3 distinct opponent classes
  (random + lite_greedy + a strong baseline at minimum).
  Promotion candidate.
- `tag: env-var-pollution-across-mp-workers` — phase-5E session:
  `agents/analytical/main.py` had module-level
  `os.environ.setdefault("PROPOSER_DRAIN_FILTER", "off")`. The
  tournament harness uses `multiprocessing.Pool` workers that
  import BOTH agents → setdefault leaks → baseline runs in same
  worker reads the polluted env. Discovered by code-review agent
  + AB symmetry inspection. Fixing the pollution (per-call
  save/restore in agent()) made baseline stronger; single-game
  game length 160 → 114, exposing that earlier 0/16 A/Bs were
  measuring an artificially-weakened baseline. **Fix:** never
  modify os.environ at module load in agent or lib modules;
  scope all overrides per agent() call with try/finally restore.
  Promotion candidate.
- `tag: arbitrary-parameter-tuning-vs-principled-values` —
  phases 4 through 5E: 5 phases spent tuning parameters
  (ALPHA_OPP_PENALTY, SHIP_COST, T_END, DEFENDER_GUARD,
  HORIZON, MAX_CONTESTERS_PER_PLANET, OPP_BONUS, ROI_THRESHOLD,
  HOLD_WINDOW, etc.) without changing the binary A/B outcome.
  PI question "why do we even have arbitrary parameter choices?"
  reframed: the game's win condition defines them (T_END=500,
  α=1.0, SC=1.0, GUARD=0). Cleanup didn't change the result
  (0/16) but exposed the architectural bind clearly. **Fix:**
  when tuning ≥3 knobs in succession yields no win-rate change,
  audit whether the OBJECTIVE itself encodes arbitrary choices.
  Promotion candidate.
- `tag: oversized-ship-counts-via-dedup-tiebreak` — phase-5G
  diagnosis: agents/baseline/proposer.py:enumerate_ship_counts
  emits 3 ship-count variants per (src, tgt) — [capture_size,
  2×capture_size, full_budget]. All variants have identical
  cheap_marginal_value for captures (formula doesn't depend on
  ship count once `ships > pred_ships`). The proposer's
  wait_band dedup is order-sensitive and tied → arbitrary winner
  (often the largest). With low SHIP_COST in the LP, larger
  variants weren't penalised. Result: source drains in ONE
  oversized fire then idles for many turns. Discovered via
  direct LP introspection of step 50 seed 42. **Fix:** raise
  SHIP_COST (now 1.0) OR fix the proposer's dedup to prefer
  smaller ship counts when cheap_delta ties.
- `tag: rule-37-axis-cap-vs-deeper-diagnosis` — phase-5C/5D/5E
  cross-cycle: agent proposed STOP at Rule 37 (3 consecutive
  axis falsifications) after Phase 4-5A losses. PI overrode:
  "don't stop on this axis; find root causes." The deeper
  diagnosis yielded real bugs (env-var pollution, oversized
  ship counts, double-counted opp arrivals) that the
  "give-up-cleanly" path would have missed. Root cause: Rule
  37 reads as a hard stop, but the PI's heuristic allows
  reframing-within-axis when concrete bugs surface. **Fix:**
  refine Rule 37 to: "STOP iterating on the same KNOB after N
  falsifications, BUT continue if root-cause diagnosis surfaces
  a NEW bug class (not just a new tuning value)."



- `tag: analytical-zero-not-bug` — Slice 8c session: agent
  proposed "relax the Δ > 0 emit gate" when differential chooser
  produced long late-game idle stretches in the introspect. PI
  override mid-session: "Δ=+0.0 is a feature, not a bug. The
  candidate space is incomplete." Root cause: agent conflated
  "no positive-Δ candidate this turn" (correct math) with
  "broken chooser." **Fix:** promoted to improvements.md as
  Rule 42 candidate ("closed-form zero is honest, not broken;
  investigate the candidate space, not the gate").
- `tag: stack-not-replace-analytical-on-rollout` — slices 4-10
  (cross-session pattern). Across 7 architectural attempts
  (predicates-as-priors backstop, bounded-interval dominance,
  LP commit-as-hint, differential leaf eval, wait_N filter,
  migration solver, joint LP chooser), every "add analytical
  layer on top of existing rollout chooser" produced same-
  or-worse win rate vs trajectory. Replacing the substrate
  was directionally right but value-calibration / candidate-
  space gaps still bit. Root cause: rollout was doing implicit
  planning via leaf-favor encoding whole-turn consequences;
  analytical commits added on top override decisions instead
  of augmenting. **Fix:** promoted to improvements.md as
  Rule 43 candidate ("analytical work replaces substrate or
  stays in input layer — never stacks on top").
- `tag: per-source-distribution-vs-class-filter-misread` —
  Slice 8c session: agent rushed wait_N>0 filter as the
  "under-emit fix" based on top-3 candidate listings showing
  wait_N>0 dominance. Introspect had already shown all 7
  positive candidates from src=8 — that's a per-source-
  allocation observation, not a wait_N one. The filter
  changed nothing meaningful (emit rate 0.75 → 0.72) and
  made outcome worse (37.5% → 18.8%). Root cause: jumping
  to one-line fix on a partial reading of the trace. **Fix:**
  before any candidate-class filter, check per-source
  positive-Δ distribution in introspect; per-source dedup
  caps emits regardless of which class dominates.
- `tag: introspect-script-stale-after-architecture-change` —
  scripts/differential_introspect.py wrapped
  `score_candidate_differential` which returns 0 for own→own
  candidates. After Slice 9 added migration candidates with
  the special-case scoring path, the introspect's "positive-
  Δ count" undercounted migration emits. Misleading trace
  output suggested migrations weren't firing when they were.
  Root cause: introspect wrappers tightly coupled to the
  exact scoring function rather than the actual choose_*
  output. **Fix:** update introspect to also wrap the final
  choose_* call and count its outputs, not just its
  internal scoring helpers. (Not done this session; logged
  for next.)

```
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

## 2026-05-23 (claude/strategy-axis-decision-3437 — items 1+3+4+5 + live A/B reveal)

- `tag: synthetic-baseline-doesnt-predict-live` — 7 A/Bs of α+β
  stacked vs derived `*_off` bundles ranged 50-62.5% (looked
  positive at 56-62% directional). Two A/Bs vs actual live submissions
  (`_phase4_step1_FND` μ=1101, `orbitfix` μ=1165) BOTH gave 3/8 =
  37.5%. Root cause: synthetic baselines isolate "code change effect"
  not "vs deployed agent." **Fix:** filed as candidate Rule at
  `knowledge-base/flags/2026-05-23-synthetic-baseline-misleading.md` —
  any A/B intended to gate a submission must include the current live
  rolling-pair leader as one opponent.
- `tag: 4p-harness-self-play-artifact` — `scripts/ab_4p_focal.py`
  gives focal 5/8 = 62.5% in self-play of identical files (vs
  expected 25% baseline). Root cause: `kaggle_environments.env.run`
  shares `sys.modules` across 4 in-process agents; when 3 of 4
  share `bg_path`, they coordinate via shared singletons
  (`_PS_DEFAULT`, `_TM_DEFAULT`, `_KT_TABLE`) while isolated focal
  gets independent state. **Fix:** new `scripts/clean_ab_4p.py`
  subprocess-per-game (mirror of `clean_ab.py` 2P). Self-play
  parity Wilson CI now includes 0.25.
- `tag: tie-handling-counts-ties-as-wins` — first version of
  `clean_ab_4p.py` returned focal_won=True when focal's reward
  equaled the max (including ties). 4 self-play games at step=500
  cap all reported "WIN" instead of "TIE." Root cause: copied
  `sorted_rs.index()` pattern from `ab_4p_focal.py:_focal_rank`
  which has the same bug. **Fix:** count strict inequality;
  `tied_at_top` recorded separately; focal_won requires UNIQUE
  rank 1.
- `tag: maximin-router-read-env-not-lazy-fn` — sister session fix
  (commit `b436e05` not mine): `agents/analytical_phase_c/main.py`
  `_decision_router` was reading `os.environ.get("LP_MAXIMIN_SEARCH")`
  directly, bypassing the lazy `_maximin_enabled()` function used
  by hardcoded variants. Result: maximin_on/maximin_off bundles
  both routed to depth2_search regardless of gate. **Fix:** router
  imports + calls `_maximin_enabled()`. This invalidated the prior
  4W/4L maximin A/B result, which actually measured LP-vs-LP.
- `tag: bundler-double-inline-joint-solver` — `bundle_analytical_phase_c.py`
  re-inlined `joint_solver/*` after `bundle_agent.py` already
  inlined them via the upgraded `DEFAULT_LIB_ORDER`. Bundle size
  854KB → 1.06MB, dual `_topology_features_enabled` definitions,
  variant builder's regex matched 8 blocks instead of 4. **Fix:**
  comment out the redundant inline pass in `bundle_analytical_phase_c.py`.
- `tag: per_planet_topology_score-kwarg-only-swallowed` — call
  site at `lp_outcome.py:852` passed `my_id` positionally; signature
  uses keyword-only (`*, my_id`); TypeError swallowed by surrounding
  `try/except`; topology_scores stayed None for every turn. The
  Phase 0 "topology-active assertion" caught this silently dead
  code (1156 calls to `_per_planet_topology_score` post-fix in an
  80-step game). **Fix:** `my_id=int(my_id)` keyword in call.
- `tag: lp-pending-not-deducted-during-opening-fallthrough` — when
  `opening_planner.opening_plan` returns committed=None during the
  opening phase (step < 30), the LP runs as fallthrough. Bundle
  defaults to `LP_PENDING_AWARE_BUDGET=0`, so the LP re-issues the
  same wait_N>0 column each turn (src/tgt have ample apparent
  ships). Across 11 turns, 11 duplicate ScheduledFires accumulated
  into commit_persistent for fire_step=12; at step 12 they all
  decanted into one action list (12 × (src=15, ships=21)). Env
  rejected most but the bug surfaced as a flood-fire. Surfaced by
  the Phase ζ.v2-opening hold-aware fix making opening_planner
  reject more candidates → empty schedule → LP fallthrough → bug
  triggered. Pre-existing latent issue masked by opening_planner
  always emitting a non-empty schedule. **Fix:** scoped
  `_predict_opp_counter` in opening_planner to OPP_RESPONSE_LAG=4
  window (mirror legacy gate) so opening_planner doesn't over-reject;
  longer-term, enable LP_PENDING_AWARE_BUDGET=1 or de-dup in
  commit_persistent. **Discovered during** the 9a19306-then-reverted
  investigation (Phase ζ.v2-opening falsified at Gate 5 = 2/16);
  the bug remains latent in tree, not yet fixed.

