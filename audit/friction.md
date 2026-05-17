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

- `tag: ab-strong-opp-before-smoke-against-floor` — 2026-05-15
  v8_analytic_phase_c session. Spent ~5 hours on three sequential
  A/B compute jobs (32-game Phase C vs B.1 = 2h26m; two 4-seed
  ablations restarted twice each after container reclaims) without
  first checking whether Phase C even beats simple floor opponents
  (`random` / `nearest` / `roi` — 5-10 s/game each). The 43.8%
  point-estimate against B.1 (a strong opp) gave no information about
  whether Phase C was beating ANYTHING; for all we knew the agent
  was randomly worse than every opponent. PI directive after the
  third failed iteration: "next time first smoke test by playing
  against simple opponents." Rule: before any A/B vs a strong opp
  (B.1, v7_0, iter_v2, etc.), run `fast.py smoke <focal>` which
  pits against `random` + `nearest` (cheap floors). Only proceed to
  strong-opp A/B if smoke passes. Lift to CLAUDE.md if it survives a
  session (currently audit-only).

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
  fixed via fast_sim+lite_greedy pivot at `7e511a0` (K=15).
- `tag: copy-K-from-jax-budget-to-fastsim` — initial fast_sim port
  inherited K=8 from the JAX-budget-constrained config. Predictable
  consequence: same horizon-too-short failure mode as JAX, manifested
  as 2/8 vs nearest (REGRESSED from JAX baseline 4/8). Per-step cost
  on fast_sim is ~10-30× cheaper than JAX, so K=15-25 is actually
  affordable. **Fix:** when porting between different cost regimes,
  re-derive horizon-vs-budget rather than copying the prior value.
- `tag: rule-37-strict-kill-vs-pi-override` — plan-mode WRAPUP gate
  was "Wilson 95% LB < 40% vs nearest → kill". 4/8 wins gives LB=21.5%
  which triggers kill, but n=8 is too small for the LB to clear 40%
  unless ≥7/8 wins. PI overrode the strict-read gate with "we don't
  need to win, we need to know if architecture is buildable-on" —
  reframed verdict from outcome-based to substrate-viability-based.
  **Lesson:** Wilson-LB thresholds at n=8 are very conservative;
  use them as one signal, not the sole verdict. Promotion candidate:
  encode "use Wilson LB AND a substrate-viability check (knob
  responsiveness, timing headroom) for kill-or-keep decisions."
- `tag: parallel-bench-cpu-contention` — earlier same-day parallel
  bench (nearest + v7_0 simultaneously) inflated per-turn timing
  30% via CPU contention and produced unreliable max-turn-ms
  numbers (some seeds at 1100ms). Sequential bench reproduced
  timing of 200-700 ms p95. **Fix:** never run two
  v8_analytic-as-focal benches in parallel; sequence them. Already
  fixed in `/tmp/probe1_bench.py` invocation pattern this session.
- `tag: bench-light-on-seeds-needs-n-balance` — n=8 side-balanced
  (4 seeds × 2 side assignments) gave honest signal where n=4
  was variance-noisy, but Wilson LB still uninformative at this
  scale. Per-seed pattern more informative than aggregate at n=8:
  K-sweep predicted seed 1 win flip from 0/2 → 2/2, and bench
  confirmed it. **Promotion candidate (low priority):** track
  per-seed outcome timeline in `state/calibration-ladder.md` so
  prior-bench seed effects are visible at decision time.

## 2026-05-16 (claude/space-fleet-physics-engine-lrLE6 — v8_analytic structural recovery)

- `tag: focal-agent-never-smoke-tested-against-floor` — opened the
  branch's focal agent `v8_analytic` (Phase B.1) and discovered it
  loses 32/0 to the trivial `nearest` baseline in the smoke floor.
  The agent had never been submitted, never been smoke-tested
  against random+nearest before; multiple ablations layered on top
  (v8_phase_c, v8_phase_c_h1, v8_phase_c_no_panel) without anyone
  noticing the underlying agent was below the smoke floor. Root
  cause: same pattern as the `ab-strong-opp-before-smoke-against-
  floor` friction logged 2026-05-14 — every new branch jumps to A/B
  vs `iter_v2` / `v7_0` and never runs the cheap-floor probe. **Fix:**
  on first session of any new branch developing a focal agent, the
  *very first* probe must be `python fast.py smoke <focal>`, BEFORE
  any A/B against strong opponents. The friction is now bound in
  CLAUDE.md Rule 2 (Kaggle GPU kernel two-tier smoke); same
  discipline should apply to any focal agent on any branch.
- `tag: state-current-md-stale-vs-leaderboard` — `state/current.md`
  last updated 2026-05-14 by a different branch, listed `geo` as
  the current submission. Live Kaggle showed the actual leaderboard
  agent is `v8_scavenge.py` (sub #52687411, score 1089.0, submitted
  2026-05-15 17:41 UTC on branch `claude/recover-main-foundations-
  MV0e2`). Two sessions of v8_analytic iteration on a different
  branch missed this entirely. Root cause: state/current.md is owned
  by whatever branch last submitted; parallel-development branches
  have no signal that the leaderboard has moved. **Fix:** every
  session-start hook should pull `kaggle competitions submissions
  -c orbit-wars` and diff the latest entry against
  state/current.md's `last_submission_id`. If mismatched, surface a
  warning before any compute.
- `tag: lax-cond-inside-vmap-evaluates-both-branches` — attempted
  to short-circuit `comet_spawn` on non-boundary steps (99% of
  steps) via `jax.lax.cond(any_spawn, _do_spawn, identity, state)`,
  expecting the fast path to skip the spawn body. Per-chunk K=8
  cost REGRESSED 40% (88 ms → 125 ms) because under `vmap`, JAX
  lowers `lax.cond` to `lax.select` — both branches are evaluated
  unconditionally and one result is chosen. The "skip work"
  intuition is wrong inside vmap. **Fix:** to actually skip work,
  the predicate must be evaluated at the Python orchestration level
  *outside* the vmap, dispatching to one of two pre-JIT'd kernels.
  Reverted via `git checkout`.
- `tag: python-unroll-inside-jit-quadratic-cost` — `fleet_launch`
  unrolled a `MAX_AGENTS × MAX_LAUNCH_PER_AGENT = 16` action loop
  in Python at trace time, producing a straight-line HLO graph
  with 16 separate `jnp.cumsum(~fleets_alive)` over MAX_FLEETS=256
  + 16 × 7 `at[].set` scatters. Phase profile showed fleet_launch
  at 55-58% of `jax_step` cost even on sentinel-only actions.
  Refactoring to a single `lax.scan` body (traced once) dropped
  fleet_launch from 2.78 ms → 0.46 ms (6×) and per-chunk K=8
  scoring from 167 ms → 88 ms (33%). **Fix established as
  pattern:** any per-step JAX function with a Python `for` loop
  over a small dimension should be `lax.scan` instead — the scan
  body is traced once, the compiled binary is much smaller, and
  XLA on CPU runs it faster. Audit other `jax_*.py` modules for
  similar unrolls.
- `tag: idle-step-runs-expensive-launch-phase` — the K-step rollout
  in `score_one` calls full `jax_step` for K-1 strict-idle turns,
  each running `fleet_launch` on sentinel-only actions. With
  vmap C=128 K=8, that's 896 wasted `fleet_launch` invocations per
  chunk (out of 1024 total). **Fix:** introduced `jax_step_no_launch`
  variant that omits the fleet_launch phase. Used as scan body for
  idle steps; turn 0 still uses full `jax_step` with the candidate
  action. Cut per-chunk cost 22%. Generalisable pattern: in any
  multi-step rollout, separate the "applies action" step from the
  "no-action" steps and route them to different JAX kernels.
- `tag: multiprocess-smoke-killed-in-sandbox` — `python fast.py
  smoke` consistently SIGTERM'd within seconds in the current
  remote-execution session, regardless of `--workers` setting
  (tried 4, 2, 1). Background tasks via Bash tool got killed
  almost immediately. Foreground play loops (`for s in 0 1 2 3;
  do python fast.py play ...`) worked fine. Root cause unclear —
  possibly multiprocess spawn pattern, possibly per-process resource
  limits in the sandbox. **Fix:** for now, use serial play-loop
  evaluations instead of multiprocess smoke when working from this
  session type. n=8 single-game loop is the practical max within
  one Bash call's 10-minute timeout.
- `tag: defense-port-regresses-vs-aggressive-strong-opp` — the
  v8_scavenge mechanism port (defensive reinforce + γ-discounted
  favor + K-step idle rollout) recovered the smoke floor (0/4 → 4/7
  vs nearest) but regressed 0/4 vs v7_0. Three suspected culprits:
  (1) γ=0.99 production discount weights prod ~5× less than the
  prior linear `prod × remaining`; (2) strict-idle K-rollout ignores
  v7_0's 7 turns of aggressive actions; (3) nearest-K=8 atom cap
  drops far-target opportunities v7_0 exploits. Not yet ablated.
  **Fix:** next session, run a 3-variant γ sweep + a Tier-1-mirror
  K-rollout variant. Phase B.1's original justification was beating
  v7_0; the port can't lose that.



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

## How to add an entry

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
