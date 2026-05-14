# audit/friction.md — current friction summary

## 2026-05-14 (claude/simplify-fast-setup-azW8T — geo v3.1 live regression + lessons)

- `tag: local-vs-v7_0-only-misses-ladder-distribution` — geo v3.1 was
  tested locally vs v7_0 only (n=192: 57.3% / Wlo 0.50; +7pp 2P)
  and vs 3x v7_0 in 4P (n=128: 56.3% first-place / +31pp over baseline).
  Live ladder result: μ=984.0 σ-discounted floor (-80μ from v7_pv's 1064.4).
  Same pattern as v3.5.1 on 2026-05-12 (-150μ vs local +56.6% Wlo). The
  v7_0 self-panel doesn't reflect the live ladder distribution
  (v3.5.1, v7_pv, v7_0_drop_one_rebuilt, top-10 archetypes).
  **Promotion candidate**: future local A/B must span ≥3 opponent
  classes before any submission. Add `--vs-panel` flag to `fast.py
  eval` that runs a 3-opponent panel by default. Cost: 3× current
  eval wallclock but eliminates the local-overpredict bias.

- `tag: geo-v2-iteration-trajectory-downward-not-individually-regressing` —
  v3.2 batch (gang_up + empty_out + tap_capture): each addition was
  within the 5pp drop threshold individually (v3.1 57.3% → v3.2a 59.4%
  → v3.2b 56.2% → v3.2c 53.1%), but cumulative -4pp from v3.1 baseline.
  Adding more candidates of similar score makes the K=10 lookahead's
  ranking noisier, even when each tilt is "fine" in isolation.
  **Fix**: test each batch component vs the FINAL baseline (not
  step-by-step) before stacking. Or: cap candidate count so adding
  one drops another (rather than appending).

- `tag: jax-vmap-already-wired` — I claimed "B: JAX vmap deferred,
  4-6 hour integration" earlier in the session, citing missing
  obs→GameState converter. WRONG. `agents/jax_v7_0/main.py` and
  `lib/game/jax/conversions.py:scalar_to_jax` already implement the
  path. Measured: JIT compile 1.8s (one-time, pre-warmable at import),
  cached score 6ms (30-70× faster than scalar score_candidate's
  ~200-400ms). **Fix**: when claiming integration is missing, READ
  agents/*_v*_*/main.py to check for prior wrappers. The "OFFLINE-ONLY"
  flag on jax_v7_0 is a parity concern, not an integration block.

- `tag: geo-v2-three-failed-wallclock-fixes` — geo v2.3 (K=10 lookahead
  + sense tilts + archetypes + 4P branch) gives ~+5pp 2P lift over v7_0
  (n=128) and +25pp 4P first-place lift (n=64). But max=1500-2900ms in
  5% of turns risks ladder forfeit. Three iterations to "fix" the
  wallclock all regressed strategy more than they bounded max:
  v2.4 lite_greedy follow-up (-17pp), v2.5 WALLCLOCK 350 (-20pp),
  v2.7 K=8 (-20pp). v2.3/v2.6/v2.8 = same code = local optimum.
  **Promotion candidate**: when a single-knob change costs more than
  it saves in three orthogonal directions, the config IS the local
  optimum — stop tuning, submit if positive, find structurally
  different lever otherwise. Detailed bisect in
  `knowledge-base/thoughts/2026-05-14-geo-v2-iteration-results.md`.

## 2026-05-13 (claude/simplify-fast-setup-azW8T — geo v1 bisect: parity ceiling)

- `tag: geo-v1-substrate-correct-heuristics-regress` — built geo agent
  (lib/geo/{sense,posture,allocator}.py + agents/geo/main.py) per the
  approved plan. ALL "obvious" value-add heuristics regressed vs v3.5.1
  bundle. Final state at parity (48.4% / Wilson [0.366, 0.604] at n=64);
  no beat. Detailed bisect table and lessons in
  `knowledge-base/thoughts/2026-05-13-geo-v1-bisect-lessons.md`.
  **Key takeaways:**
  (1) Replacing settle_plan with global score-sort multi-launch
      regresses -31pp because it concentrates force at strong sources
      instead of spreading.
  (2) Cross-class score multipliers >= 2× regress -37pp because they
      crush settle_plan's per-source best-mission selection.
  (3) Non-aggressive snipe sizing in any posture regresses -22pp
      because v3_snipe loses to v3.5.1 by 56.6%.
  **Promotion candidate**: before bolting "obviously helpful" heuristics
  onto a tuned baseline, test each one in ISOLATION against the baseline.
  bisect-2 (v3.5.1 source-pipeline) confirmed the substrate is correct at
  46.9%; without that anchor I'd have spent more cycles chasing a
  source-vs-bundle drift hypothesis.

## 2026-05-13 (claude/simplify-fast-setup-azW8T — fast.py landed)

- `tag: fast-py-is-now-canonical-iteration-entry-point` — single-file
  `fast.py` at repo root replaces the diffuse iteration harnesses with
  one CLI: `python fast.py {smoke,eval,play,bench,baselines} <agent>`.
  Adaptive Wilson-gated A/B (16 → 32 → 64 tiers, early-stop on
  Wlo≥gate or Whi<gate). Plain-function agents anywhere
  (`def agent(obs, configuration=None)`) — no directory ceremony.
  **Superseded** for the iteration loop (NOT deleted, will rot):
  `scripts/run_ablations.py`, `scripts/run_v7_wide_deep_smoke.py`,
  `scripts/run_v7_wide_deep_ab.py`, `scripts/ab_variants.py`,
  `scripts/eval_v1.py`, `scripts/strategy_panel.py`. `scripts/tournament.py`
  and `scripts/ffa_tournament.py` are kept for 4P FFA panels (fast.py
  is 2P-only in v1). `scripts/bundle_agent.py` is unchanged — fast.py
  evaluates source-tree or bundled files, but submission still goes
  through the bundler. Verified: smoke / eval / play / bench all run
  green against `random`, `nearest`, `v7_0_drop_one`. Deferred v1.5:
  full-episode play through `lib/fast_sim.py` (skips `env.run` overhead)
  and a `--jax` vmap-over-games fast path.

## 2026-05-12 EVE (game-ai-lookahead-3ucqH — v9 super-version + v10 + submit attempt)

- `tag: kaggle-cli-401-was-wrong-auth-env-var` — `kaggle competitions
  submit` returned HTTP 401 throughout the day; I incorrectly
  concluded the token was sanitized/expired. **Root cause:** the
  `$KaggleAPIToke` token has format `KGAT_<32 hex>` — that's a NEW
  Kaggle auth format. The legacy `KAGGLE_KEY` env var (used by
  `bootstrap.sh`'s default path) treats this as a plain key string
  and the API rejects it. The CORRECT env var for `KGAT_*` tokens is
  **`KAGGLE_API_TOKEN`**. Tested: `unset KAGGLE_USERNAME KAGGLE_KEY;
  export KAGGLE_API_TOKEN="$KaggleAPIToke"; kaggle competitions
  list -s orbit` → works. Same env unlocked the actual submit
  (v7_0_drop_one pushed as #52588156). **Fix to land before next
  session:** `bootstrap.sh` step 1 should detect the `KGAT_` prefix
  on the harness token and export `KAGGLE_API_TOKEN` instead of
  writing `~/.kaggle/kaggle.json` (which uses the legacy key path
  internally). Cost ~30 min of session time on diagnosis.
  **Promotion candidate:** update bootstrap auth-detection logic;
  add a runtime smoke `kaggle competitions list` immediately after
  cred resolution so failures surface in the first 5 minutes, not at
  submit time.

- `tag: sigma-equiv-helps-v7minimax-hurts-v7_0` — the σ-equiv layer
  (sym_hypot + planner _tb + SCORE_ROUND=6) lifted v7_minimax to μ=1063
  (per parallel-branch audit: ~+45μ over v3.4 baseline alone) but
  REGRESSES v7_0's drop-one architecture by ~54pp at n=24. v7.6 bisect
  confirmed this directly. Root cause hypothesis: σ-equiv's deterministic
  tie-break removes some diversity from settle_plan's candidate ordering;
  v7_minimax's K=3 maximin partially compensates because it explicitly
  scores both σ-paired actions, while v7_0's K=10 ship-delta relies on
  variance in the candidate set to differentiate launches. **Fix:**
  σ-equiv reverted in lib/missions/snipe.py + lib/planner.py for v9+.
  Promotion candidate: the v3 family agent comparisons need σ-equiv;
  the v7-family agents (drop-one + K=10) don't. Library-level changes
  that benefit one architecture can harm another.

- `tag: K=15-regresses-with-ship-delta-head` — bumping K from 10 to 15
  with the default ship-delta scoring head regressed v9_k15 to 2/24 = 8.3%
  vs v7_0 (Wilson lo 2.3%, catastrophic). p95 = 734ms over the 700ms
  watchdog → watchdog truncates → conservative-incumbent-fallback on
  many turns. The inflight_value head RESCUED this from total collapse
  in v9_combined (58.3%) but couldn't gate-clear. **Lesson:** K=10 is a
  sweet spot, not a blind spot, in the v7_0 regime. Don't extend K
  without first reducing per-step rollout cost or upgrading the head.

- `tag: value-head-lift-pattern-consistent-but-not-significant-at-n=24`
  — Five super-version variants we tested all point to evaluate_value
  / inflight_value lifting v7_0 by ~+12-25pp at point-estimate, but
  none cleared Wilson 55% at n=24. n=96+ would be needed to crystallize.
  The lift is real (smoke confirms v9_combined launches when v7_0 returns
  []) but small enough that the existing budget can't gate it cleanly.
  **Promotion candidate:** local A/B at n=64 should be the standard
  gate for "directional but-not-yet-Wilson-significant" candidates.

## 2026-05-12 PM (game-ai-lookahead-3ucqH — v7.1-v7.6 stack iteration)

- `tag: maximin-budget-blow-up` — v7.1's 2×N maximin matrix × symmetric
  scoring (2× cost) = 4× rollouts per turn. With K=10 and recapture
  in the incumbent, p95 turn ms hit 1105 — over the 700 ms watchdog
  → conservative fallback to incumbent → -54pp regression vs v7_0
  (25% Wilson lo 12%). **Fix:** dropped the maximin overlay entirely;
  `choose_simple_with_4p` runs drop-one argmax (same as v7_0) with
  σ-equiv + recapture + 4P-aware. The maximin theoretical guarantee
  isn't worth its compute cost at K=10.

- `tag: bundle-default-lib-order-stale-when-new-modules-added` —
  v7.5 A/B run 1 returned 0/24 silently because the bundle inlined
  references to `propose_recapture_missions` and `evaluate_value`
  but their source files weren't in `scripts/bundle_agent.py::
  DEFAULT_LIB_ORDER`. NameError at runtime → empty action → loss by
  elimination, no log. **Fix:** appended `missions/recapture` +
  `lookahead_planner` to `DEFAULT_LIB_ORDER`. **Promotion candidate:**
  add a pre-bundle check that greps every `agents/*/main.py` for
  `from lib.<name>` imports and warns if not in lib order. Stops
  silent bundle bugs at bundle time, not at A/B time.

- `tag: type-checking-import-survives-bundler-strip` —
  `lib/lookahead_planner.py` had `if TYPE_CHECKING:` then `from lib.intent
  import World`. The bundler strips the inner `from lib.*` line but
  leaves the `if TYPE_CHECKING:` block → empty body → IndentationError
  at exec. **Fix:** removed the TYPE_CHECKING guard; inline-quoted
  the `World` forward-ref to plain `world` in `adaptive_K`.
  **Promotion candidate:** bundler should drop empty conditional
  blocks left behind by `_INTRA_IMPORT_RE` stripping.

- `tag: recapture-still-regresses-even-after-score-scale-fix` —
  Ported `propose_recapture_missions` with the audit's hypotheses #1
  (snipe-scale denominator) and #2 (top-K=5 cap) fixed. v7.5
  (σ-equiv + recapture + 4P) still regressed -8.3pp vs v7_0 in 2P
  A/B. Hypothesis #3 (premature commitment on infeasible
  recaptures) is the likely remaining bug — recapture missions
  fire on recently-lost planets whose new owner is fortifying,
  burning ships that should snipe/reinforce. **Fix flagged for
  next session:** add a feasibility check that requires
  `model.ships_at(target, eta) < base_ships - 1` (target stays
  capturable at our arrival). Otherwise drop the recapture
  proposal.

## 2026-05-12 (game-ai-lookahead-3ucqH)

- `tag: bootstrap-env-var-typo-KaggleAPIToke` — the harness exposes
  Kaggle credentials as `$KaggleUserName` and `$KaggleAPIToke` (note
  the truncation: `Toke` not `Token`), but `bootstrap.sh` looks for
  `KAGGLE_USERNAME` / `KAGGLE_KEY`. First bootstrap run hit
  `ERROR: no Kaggle credentials found`. Fix this session: invoke
  bootstrap with `export KAGGLE_USERNAME="$KaggleUserName" KAGGLE_KEY="$KaggleAPIToke"`
  prefix. **Promotion candidate:** add an explicit name-translation
  block at the top of `bootstrap.sh` (or a `.envrc` shim) that maps
  the harness-style names to the documented `KAGGLE_*` names so the
  next session doesn't re-hit this.

- `tag: bootstrap-skip-data-when-shotvalidator-present` — `bootstrap.sh`
  step 3 uses `compgen -G "data/*"` + `grep -v '^\.gitkeep$' | wc -l`
  to decide whether to download comp data. Because the repo already
  has `data/shot_validator/`, this evaluates non-empty and the
  download is skipped — but `data/main.py` and `data/README.md` are
  *not* present, and several existing tests (`test_fixture_smoke.py`,
  `test_v1_parity.py`, `test_bundle.py`) require them. 17 tests fail
  with `FileNotFoundError: data/main.py` in a fresh sandbox until
  the comp data is pulled (and this sandbox's harness creds return
  401 on the comp endpoint, blocking a workaround). **Fix:** narrow
  the bootstrap "is data present?" check to specifically look for
  `data/main.py` (the deciding artifact), not "any non-gitkeep
  file."

- `tag: env-clone-cost-grows-with-history` — the Phase 2 audit quoted
  `env.clone()+step()` at 5.6 ms/step on a cold env. After 20 warmup
  steps the cost rises to ~22 ms (`Environment.clone()` references
  `self.steps`, which grows linearly through the episode). Mid/end-
  game per-turn cost is therefore ~4× worse than the audit number
  suggests. `scripts/bench_fast_sim.py` records both numbers in the
  audit doc (`audit/2026-05-12-fast-sim-bench.md`) so the cost
  trajectory across the episode is explicit.

## 2026-05-11/12 (optimize-ship-strategy-tDPXx)

- `tag: idle-bucket-reduction-is-misleading-proxy` — methodology:
  Phase-0 idle-source decomposition surfaced MECHANISM_DROP at ~96%
  of all idle classifications and motivated four consecutive
  scoring/filter knobs (airtime, endgame, affordability filter,
  gang-up). Every variant *reduced* the bucket — and *tied or
  regressed* at 64-seed Wilson vs v3.4 baseline. Root cause: each
  "drop" represents an attempted capture at a high-value target;
  some succeed via WorldModel adversary stacking and the ones that
  bounce still preserve home-garrison defensive value. The proxy
  measures attempts-not-launched, while what matters is
  expected-value-captured. **Fix this session:** documented the
  inversion in `audit/2026-05-11-v3.5-airtime-and-endgame-burn.md`
  and `audit/2026-05-12-gang-up-v1.md`. Promotion candidate: codify
  "validate the proxy via correlation with the actual outcome metric
  BEFORE running ≥1 variant against it." Multiple cycles of fix→fail
  cost the entire overnight session.

- `tag: gang-up-substrate-bug-arrival-size-reinflates` — gang_up_size
  ran BEFORE validate (correct) but BEFORE arrival_size too;
  arrival_size's `intent.ships = max(intent.ships, needed)` silently
  re-inflates every throttled share to the full target garrison, then
  drops if the re-inflated value exceeds `src.ships`. Phase-0 data:
  validate drops -39% (gang-up's intended effect) but arrival_size
  drops +31% (the re-inflation backfire). Net total drops -2%, so the
  gang-up mechanism wasn't even mechanically working end-to-end.
  **Fix:** documented; future Option A redesign needs arrival_size to
  be sibling-aware (track sum of co-target intents, deduct from
  per-intent needed). Filed in `audit/2026-05-12-gang-up-v1.md`.

- `tag: ab-variants-regex-rejected-inline-comments` — scripts/ab_variants.py
  initially matched `^NAME\s*=\s*[-+0-9.eE]+\s*$` which excluded
  declarations with trailing inline comments like `GANG_UP_ENABLED = 0
  # default OFF`. First gang-up A/B failed at the bundling step with
  "variant override not found." Patched regex to tolerate
  `(?:#.*)?$` and preserved the trailing comment in the substitution.
  **Fix:** committed in `a8ae69a` alongside the multi-file
  auto-discovery. Promotion candidate: harness scripts that patch
  source files should accept the project's actual coding style, not
  a stricter subset; add a smoke unit test for known constants in
  each declared `PATCHABLE_PATHS` file.

- `tag: ab-variants-hardcoded-snipe-only` — scripts/ab_variants.py
  originally hardcoded `SNIPE_PATH = lib/missions/snipe.py`. The
  first gang-up A/B attempt tried to patch GANG_UP_ENABLED (defined
  in `lib/mechanism.py`) against the snipe file and failed loudly.
  **Fix:** extended to scan `PATCHABLE_PATHS = [snipe, reinforce,
  mechanism, planner]`, auto-discover the owning file per constant,
  and error on multi-file collisions. The discovery + collision
  check is cheap and prevents a class of future bugs as more
  ablation knobs are added across files.

- `tag: bool-vs-int-constant-typing-for-ab-regex` — added
  PROPOSER_AFFORDABILITY_FILTER as `False` (bool); ab_variants regex
  only matches numeric literals so couldn't patch it. Fixed by
  switching the constant to int (`0` / `1`) — Python truthiness
  preserves the `if FLAG:` check. Costs us one type signature
  precision in exchange for cleaner harness compatibility. **Fix:**
  documented in the constant docstring; future opt-in flags will
  default to numeric-literal style.

- `tag: claude-bash-pipe-buffers-progress-output` — first big sweep
  (6-variant 32-seed) launched as `python ... 2>&1 | tail -15`
  produced no output until completion. The trailing pipe buffers
  stdout, so I couldn't see progress and assumed the bundling was
  hanging. Killed it; re-ran. **Fix:** use `python -u -m ... >
  /tmp/<name>.log 2>&1 &` for background sweeps (no pipe → real-time
  flush) and arm a Monitor on the log file. Pattern locked in for
  the remaining 4 sweeps tonight; cost was ~30 minutes of waiting +
  one wasted compute window.

- `tag: 32-seed-point-estimate-noise-at-128-game-level` — AIRTIME=0.5
  variant looked good at 32-seed pair-level (54.7%, 35/29/0) but
  converged to 52.3% (67/61/0) at 64-seed. The extra 32 seeds were
  precisely 50/50 = 32/32 wins/losses. Wilson_lo dropped from 42.6%
  to 43.7% — small absolute change, but the "tied" verdict only
  materialised at 64-seed. **Fix:** raise the "ship" gate to require
  64-seed pair-level Wilson_lo > 50%. 32-seed point estimates above
  50% can be confidently noise. Promotion candidate: "32-seed
  Wilson_lo < 50% → require 64-seed retest before ship."

## 2026-05-11 PM (analyze-submission-logs-dFHeS)

- `tag: stale-rolling-last-2-pre-submit` — submission flow: pushed
  v3.4 (52556866) at 21:19 UTC believing the rolling-last-2 was
  `[v2 (965.3), v3_snipe (1055.5)]`. Actual state from a fresh
  `kaggle competitions submissions orbit-wars` query was
  `[v3_snipe (1005.7), precision_v3 (984.6)]` — the parallel branch
  `precision-physics-engine-ymJkA` had pushed `precision_v3.py` at
  17:00 UTC, 4h before my push, silently evicting v2. My push then
  evicted v3_snipe (the strongest live entry), not v2. **Fix:** every
  pre-submit decision record MUST include a fresh
  `kaggle competitions submissions orbit-wars` query taken within the
  last 10 minutes. Branch-local state and HANDOVER files are
  unreliable indicators when multiple agents push in parallel. Add
  this as a Rule 12 sub-clause. Net impact this time: v3.4 should
  still settle ≥ v3_snipe (it's a strict extension), so the worst
  case is parity. But the principle is load-bearing.

- `tag: postmortem-fleet-schema-off-by-one` — analysis context:
  `scripts/episode_postmortem.py:211` unpacked `ships_launch =
  init_entry[5]` but the Fleet schema is `[id, owner, x, y, angle,
  from_planet_id, ships]` — index 5 is `from_planet_id`. Per-episode
  postmortem `fleet['ships']` was therefore meaningless. Fleet OUTCOME
  categorization (captured/bounced) was unaffected since it's based
  on planet ownership flips. **Fix:** changed to `init_entry[6]`;
  added a comment citing the env source. Promotion candidate: add a
  smoke test that asserts a known launch's recorded `ships` equals
  the action's emitted ship count.

- `tag: blanket-formula-fix-without-targeted-evidence` — investigation
  flow: confirmed real "one ship too little" near-miss bounces (38 of
  518 bounces at margin {+0, +1}), built a blanket fix
  `needed = G + production * (eta + 1) + 1`. 32-seed A/B regressed
  to 42.2% Wilson [30.9%, 54.4%]. Root cause: env's center-to-center
  distance over-estimates eta by `(r_src + r_target)/v` for static
  targets; the blanket fix wastes ships there. **Fix:** when a finding
  identifies a small minority of cases (here 7% of bounces), do NOT
  apply a fix that affects 100% of cases. Targeted heuristic only.
  Same lesson re-fired with the flat `NEUTRAL_BONUS=1.5 /
  COMET_BONUS=1.3` fix (also 28.1% regression). Promotion candidate:
  "if the finding is local, the fix must be local."

## 2026-05-13 LATE (claude/read-handover-iLWTq — stale handover read)

- `tag: handover-stale-at-session-start-no-git-log-check` — session
  start: I read HANDOVER.md ("Last written: 2026-05-13 EVE by
  consolidate-fast-simulation-ysd9M") and built a full plan-mode
  design for diagnostic + cheap wins + brute-force search, including
  writing the plan file at
  `/root/.claude/plans/go-diagnostic-cheap-wins-woolly-rose.md`. After
  ExitPlanMode, `git log --oneline` revealed the branch tip already
  carried `cb02fd9 diagnostic + cheap wins + brute-force search` and
  `4ba55f4 A/B result: v7_1 (H11+H15) ... Wilson lo 36.4%` — i.e. the
  entire plan had been executed earlier on the same branch, with the
  Track B A/B already returning a below-gate verdict. A newer
  HANDOVER.md ("Last written: 2026-05-13 LATE by
  claude/read-handover-iLWTq") was on disk by the second read pass
  and reflected the real state. Root cause: I treated HANDOVER.md as
  canonical without cross-checking against `git log` to verify the
  handover matches the branch tip's commits. Rule 32 mandates
  session-start git fetch + log diff, which I skipped. **Fix this
  session:** noted here and pivoting to the actions in the current
  handover (JAX 64-game A/B for v7_1, parity-gate divergence
  investigation). **Promotion candidate:** add a pre-handover-read
  step in the kaggle-comp skill (or CLAUDE.md prelude) that runs
  `git log -5 --oneline HEAD` and reconciles its subjects against
  the handover's "This session" section before reading anything
  else. Time cost: ~30 min designing work that was already done.

## 2026-05-13 (consolidate-fast-simulation-ysd9M — JAX sprint wrap)

- `tag: silent-engine-capacity-loss` — JAX `fleet_launch` slot
  allocator used `target_count = launches_so_far + 1` against a
  cumsum that was recomputed each iter from already-updated
  fleets_alive. After launch 0 lands at slot 0, the second launch
  steered to cum_free==2 (slot 2), wasting slot 1. Half the fleet
  capacity silently lost; tests passed because they compare by
  fleet id, not slot density. **Root cause:** off-by-one in a
  Python-unrolled inner loop that mutated state in flight. **Fix:**
  set `target_count = jnp.int32(1)`, add `slot_mask.any()` guard so
  a full fleet array doesn't overwrite slot 0. The new
  `test_fleet_launch_packs_slots_contiguously` would have caught
  this. Promotion candidate: any in-loop-mutating state needs a
  contiguous-packing test, not just a per-element parity test.
- `tag: parity-tests-vs-mirror-not-source-of-truth` — three places
  where "JAX matches scalar" parity tests actually compared
  JAX-vmap form against the numpy-mirror form (which inherits the
  same constants). The 0.5-vs-0.3 lead-aim tolerance bug (8f-C2)
  passed tests for that reason. **Fix:** added a single-state
  scalar-vs-JAX rollout test (`test_jax_rollout_pipeline_matches_
  scalar_realize`). Promotion candidate: when porting algorithm
  X to platform Y, the parity test must compare Y-output against
  X-output, not Y-output against a-second-port-of-X.
- `tag: harness-knob-without-plumbing` — `run_jax_ab.py` exposed
  `A_AGGRESSIVE` env var, but `rollout_step_jax_pure` accepted
  only `opp_aggressive`. The var was silently ignored. The kernel
  ran 4 versions before this surfaced. **Fix:** plumbed through
  `my_aggressive` parameter end-to-end. Promotion candidate: every
  exposed CLI/env knob should be exercised by a smoke test that
  flips it and asserts the rollout output diverges.
- `tag: kaggle-cli-kgat-auth-2hr-detour` — `kaggle kernels push`
  failed with 401 for ~30 min before I found the new KGAT token
  needs `KAGGLE_API_TOKEN` env var, not `kaggle.json`. The CLI's
  `auth_method: LEGACY_API_KEY` config was misleading. **Fix:**
  documented in `scripts/kaggle_ab_kernel/run_ab.py` doc; mental
  model: KGAT tokens need bearer-style env vars. Promotion
  candidate: when an auth path fails, grep the CLI source for env
  vars before fighting the credentials file.

## 2026-05-12 (analyze-leaderboard-strategies-sdZlE)

- `tag: stack-first-ablate-later-is-the-wrong-order` — iter-1 v3.5
  build: I composed all four new Mission classes (opening, drain,
  gang_up, recapture) into v3.5's `agents/v3.5/main.py` and ran a
  32-seed A/B before any individual ablation. Stack failed at 39.1%
  (Wilson lo 28.1%). The PER-MISSION ablation (run AFTER the fail)
  showed each class individually failed too. Had I run the per-mission
  ablation FIRST I'd have saved the full-stack tournament (~64 games)
  AND identified that NONE of the four would have lifted, redirecting
  to the surgical-edits approach (iter-2) hours earlier. The priors
  were available: v3.3 blanket-eta-fix regressed (42.2%) and v3.4
  NEUTRAL_BONUS=1.5 regressed (28.1%) — same pattern, both in main's
  audit. **Fix:** when adding ≥2 new Mission classes or proposers
  simultaneously, ALWAYS run per-class ablation panel BEFORE the
  stacked full-agent A/B. Time saved: 1× tournament = ~5-10 min in
  this case, but the strategic redirect is the bigger win. *Promotion
  candidate (in-comp; PI declined cross-comp earlier today).*

- `tag: module-mutation-patching-has-worker-reuse-race` — iter-2
  parameter sweep build: first cut of the agg_06 / agg_08 / agg_09
  variant agents set `base.SHIP_FRACTION = X` at module-import time
  (the `aggressive_sizing` base module was imported once per worker,
  then mutated). In multiprocessing's worker-reuse mode, a worker
  loaded with `aggressive_sizing_06` (sets fraction=0.6), then
  reused for `aggressive_sizing_08`, would observe the LATER
  value because both modules mutated the shared base. Caught at
  design time before launching the sweep; fixed by moving the
  assignment inside `agent(obs)`. **Fix already applied this
  session:** variant agents set the constant inside agent() so every
  call resets it; safe within a single-threaded worker process.
  Promotion candidate: avoid module-level constant mutation as a
  parameter-sweep mechanism; prefer either (a) function parameters
  threaded through, or (b) full per-variant copies of the proposer.

- `tag: data-main-py-not-fetched-by-bootstrap-recurrence` —
  iter-2 4P FFA: `scripts/run_ffa_agg.py` failed with
  `FileNotFoundError: agent file not found: data/main.py` because
  bootstrap.sh's data download path didn't run on this fresh
  container. SAME issue as the 2026-05-10 PM friction
  `data-main-py-not-fetched-by-bootstrap`. Fix-this-session: ran
  `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle competitions download -c
  orbit-wars -p data/ && unzip` manually. **Recurrence count: 2 this
  comp.** Bootstrap.sh fast-path remains broken; the data step is
  gated behind an unclear condition. *Promotion candidate (in-comp,
  re-promote): bootstrap.sh should unconditionally run the
  competitions-download step on a missing data/main.py — no
  conditional gates.*

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
