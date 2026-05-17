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

## 2026-05-17 (claude/improve-fleet-efficiency-cQXg4 — v21 patches falsified)

- `tag: explicit-rewrite-of-implicit-behavior` (2nd recurrence) — built
  v21 = v20 + cheap target-quality prefilter (E1) + joint-commitment
  emit (A) + rollout-based capture-and-hold filter (E2). All three
  patches address the empirical lost-back failure mode (60-70% of
  captures lost back within 50 turns per replay analysis). All three
  layer FILTERS on top of v15's chooser pipeline. h2h vs v15 at n=32:
  10/32 = 31.2% Wlo=0.180 Whi=0.486 — clean regression. **Root cause:**
  v15's reactive-opp rollout already encodes the lost-back signal —
  opp counter-recaptures inside the rollout → my F2 leaf-favor drops →
  candidate Δ collapses. Adding explicit filters double-counts that
  signal and trims productive aggression along with the waste. **Fix:**
  add the signal *inside* the rollout (stronger opp policy), not on
  top of it. New plan at `audit/2026-05-17-v21-pivot-plan.md`. 2nd
  recurrence → promotion candidate for kaggle-comp improvements.md:
  pre-flight check before any explicit filter on the chooser output is
  "does the rollout's reactive opp already catch this? if yes, do not
  add the filter — modify the opp instead."
- `tag: n16-falsely-shows-parity` — v21 h2h vs v15 at n=16 returned
  8/16 = 50.0% (Wlo=0.28, Whi=0.72), read as "INCONCLUSIVE parity";
  the SAME variant at n=32 returned 10/32 = 31.2% (Wlo=0.18,
  Whi=0.486) — a clean FAIL. Wilson CI width at n=16 is ~0.45,
  literally cannot distinguish parity from a 20pp regression. Burned
  4 variants × ~7 min each (28 min compute) on the n=16 panel before
  the n=32 reveal. **Fix:** when stake is a submission decision, n=16
  is for SMOKE only ("agent doesn't crash"). Gating decisions require
  n=32 minimum (Wilson width ~0.25). Already related to
  `small-n-ab-noise-misled-panel` (2026-05-15) — that one was 5/8;
  this one is 8/16; the lesson is the same. Promotion candidate:
  refuse to call a 50%-at-n=16 result "parity"; either bump n or
  declare "untested." Specifically promote to fast.py eval default:
  `--max-seeds 16` (= n=32 with 2-seat balance) should be the default,
  not n=8 (= n=16). Two-line code change.
- `tag: diagnostic-sample-size-overfit` — wrote `scripts/diag_v21_vs_v15.py`
  on 4 seeds (1000-1003) to find the cause of v21's n=16 parity result.
  Roll-up showed "Δemits=-23.2, Δcaps=-5.8, Δlost_back=+1.2" — concluded
  "Patch A over-commits." Built v21_solo (MAX_COMMIT_ROUNDS=0) to test;
  result = same 43.8% as A-only and A+E1. Diagnostic was over-fit to
  seed 1002 (the only divergent seed in the 4; +62 lost_back single-
  handedly drove the +1.2 mean). **Root cause:** 4-seed diagnostics
  cannot identify a dominant failure mode; one bad seed dominates the
  mean. **Fix:** per-game diagnostics need n ≥ 16 on the SAME seed set
  as the h2h panel — otherwise the diagnostic's mean is a tail, not the
  bulk. Or use median + IQR instead of mean.


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
