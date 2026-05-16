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

## 2026-05-15 (claude/fix-weak-game-starts-NhDQ3 — capture-and-secure)

- `tag: detached-bg-killed-on-session-resume` — Long-running panel
  suite died across two session boundaries despite `nohup setsid …
  & disown`. The bash-tool's own `run_in_background` survived ~3 h
  on the first run, but the next session's harness reset killed
  the resume-script. **Root cause:** harness reaps process groups
  at session-start regardless of detachment. **Fix:** for jobs
  that must cross sessions, prefer the bash-tool `run_in_background`
  + idempotent resume scripts that detect interrupted logs and
  pick up missing variants. Logged on branch wrap-doc.
- `tag: local-overpredict-2x` (3rd recurrence) — geo_recap panel
  mean 60.9 % across 192 games. Pattern: v3.5.1 5/12 −15 pp,
  geo v3.1 5/14 −7 pp, recap held from submit because expected
  live ~54 % is not decisively above v7_pv's 1062.2 μ. **3× fired
  ⇒ promotion candidate.** **Fix:** add a hard rule — local A/B
  mean must clear (gate + discount), not gate alone. Promote to
  `.claude/skills/kaggle-comp/improvements.md`.

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
