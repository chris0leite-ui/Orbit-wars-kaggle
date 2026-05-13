# HANDOVER.md — next-session brief

> Last written: 2026-05-12 EVE by `claude/consolidate-fast-simulation-ysd9M`.
> Format budget ≤ 160 lines. Prior wraps archived under
> `audit/archive-2026-05-1*-handover-*.md`.

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC →
  **42 days remaining.**
- **Live submitted agent:** `v7_0_drop_one`, **submission #52588156**,
  status COMPLETE with **publicScore 1094.9 — team peak.**
- **Rolling-last-2 (Kaggle's auto-evaluation pair):**
  `[v4_planner #52579863 (1038.6), v7_0_drop_one #52588156 (1094.9)]`.
- **Public-LB rank:** **109 / 2587 teams** (top 4.2%).
- **Gap to top-10 prize:** +336 to 3Comets (#10) at 1430.9. #1 =
  bowwowforeach 1675.9.
- **Daily submission budget:** 2/5 used today already (#52565976 v3.5.1
  + #52588156 v7_0_drop_one).
- **Test suite:** 373 pass / 2 skipped / 1 xfailed (v3_snipe replay-
  parity drift from σ-equiv lib patches — documented in
  `tests/test_replay_parity.py`).
- **σ-awareness:** Kaggle's published Score = μ − κσ. v7_0_drop_one is
  at 64 evaluation episodes (≈ 4 h since submit at the snapshot),
  σ-band ~6 Score points. The +56 lead over v4_planner is real, not
  σ-shimmer.

## This session — consolidation only

Branch `claude/consolidate-fast-simulation-ysd9M`. One thick merge plus
prune and state rewrite. No new submission.

1. **Merged `claude/game-theory-strategy-analysis-0oH4N`** (which itself
   merged `claude/game-ai-lookahead-3ucqH`) onto this branch. Brings:
   - **Fast brain core:** `lib/fast_sim.py` (183× faster than env.clone +
     step; bit-exact Snapshot wrapper around
     `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter()`),
     `lib/opp_model.py` (Tier 0 v3_snipe / Tier 1 v3.5.1 opponent
     policies), `lib/v7_search.py` (chooser: enumerate candidates,
     score each via fast_sim rollout, pick best),
     `lib/lookahead_planner.py` (value head + adaptive K + comet-
     boundary truncation), `lib/value_heads.py`,
     `lib/candidate_portfolios.py`, `lib/mirror.py`.
   - **σ-equivariance patches:** score rounding to 6 decimals + symmetric
     tie-break in `lib/planner.py`, `sym_hypot` in `lib/orbit.py` for
     bit-exact paired distances.
   - **Anchor agent:** `agents/v7_ablations/v7_0_drop_one/main.py` +
     `submissions/v7_0_drop_one.py` (the live μ=1094.9 bundle,
     sha256 `bb7ab23a75bc5865`).
   - **Knowledge:** `knowledge-base/concepts/lookahead-simulator-
     architecture.md` (permanent reference for the fast-brain stack).
2. **Pruned dead and intermediate variants:** failed v7 sweeps
   (`v7_1_minimax`..`v7_6_no_recapture`, `v7_combined`, plus
   `v7_ablations/v7_1`..`v7_4`), parked v8 PSRO stack
   (`v8_fastbrain`, `v8_minimal`, `v8_psro_meta`), failed v9 super-
   versions (`v9_combined`, `v9_inflight`, `v9_k15`, `v9_opening`),
   v10 variants (`v10_evaluate` FAIL, `v10_opening` not validated),
   intermediate v3.5 scaffolding (`v3.5`, `v35_ablations`,
   `v35_iter2`), intermediate v4 sweeps (`v4_endgame`, `v4_hybrid`,
   `v4_mirror`, `v4_mirror_t1`, `v4_mirror_t2`), and dead A/B harness
   scripts (`psro_*`, `run_v7_ablation`, `run_iter2_ablation`,
   `run_v35_ab`, `run_sizing_sweep`, `run_aggressive_sizing_32`,
   `run_ablation_panel`, `run_ffa_agg`, `opening_probe`, `bench_v7`).
3. **State docs rewritten** to true live scores. `state/current.md`
   now carries the full live submission ladder (13 entries from
   day-1 baseline through v7_0_drop_one), σ-proxy episode counts,
   and the corrected top-10 cliff (1430.9, not 1447.6).

## What's kept (the minimum that matters)

`agents/`: `simple`, `v1_orbitfix`, `v2`, `v3_lookahead`, `v3_snipe`,
`v3.5.1`, `v7_ablations/v7_0_drop_one`, `v7_minimax`.

`submissions/`: `v3.5.1.py`, `v3_snipe_frozen.py` (parity test
fixture, regenerated post-merge), `v4_planner.py`, `v7_0_drop_one.py`
(live anchor), `v7_minimax.py`.

`lib/`: full mission framework (`missions/{snipe, reinforce, gang_up,
recapture, drain, opening}.py`), full physics
(`{aim, combat, fleet, geometry, mechanism, orbit, trajectory,
world_model}.py`), the fast-brain stack listed above, plus
`{intent, mission, planner, scoring, fingerprint}.py`.

## Phase 3a — scalar-rollout speedups (DONE this session)

Driver: profile showed `generate_comet_paths` costs ~85 ms on each of
the 5 spawn boundaries (steps 50/150/250/350/450). An agent doing
K=10 lookahead × 50 candidates that crosses the next spawn would
spend ≈5 s on comet generation alone — silently over the 1 s/turn
budget. Killer fix: cache the result.

What landed:

1. **Comet-path cache.** `_FakeEnv` gains `comet_path_cache: dict`
   keyed by `(episode_seed, spawn_step)`. Shared across clones inside
   `lib/fast_sim.clone()`. The interpreter consults it before calling
   `generate_comet_paths`; cache miss → store; cache hit → skip the
   computation entirely. New parity test
   `test_comet_cache_hit_parity` (10 cases) validates two
   cache-sharing branches produce byte-identical state.
2. **AABB prune in `swept_pair_hit`.** Cheap bounding-box rejection
   before the discriminant math; eliminates ~70-90 % of the fleet ×
   planet pairs.
3. **Local hoists** in the per-step hot loops (planet path comp,
   comet update, fleet movement) + **module-level math aliases**
   `_cos`/`_sin`/`_sqrt`/`_log`/`_atan2`.
4. **Set-based fleet removal** — `id()` membership replaces
   `f not in list` element-wise list-equality.

Skipped: lighter `clone()`. Standalone bench: clone() is 12 µs/call,
not 241 µs — the earlier number was env.reset() noise.

**Impact:**

| Workload | Before | After | Speedup |
|---|---|---|---|
| Interpreter per-step (steady state) | 1224 µs | 983 µs | **1.24×** |
| Mid-game lookahead per-step (K=10) | — | 190 µs | — |
| **Spawn-crossing turn, K=10 × 50 candidates** | **~5 s** | **155 ms** | **~32×** |
| Non-spawn turn, K=10 × 50 candidates | — | 97 ms | — |

The cliff is gone. Agents can now do wide+deep lookahead on
spawn-boundary turns inside the 1 s budget.

Gates: 62/62 game-parity tests; full suite + 150-episode parity sweep
both pending confirm at wrap-up time.

Critical files: `lib/game/interpreter.py` (math aliases, AABB,
hoists, set-remove, cache lookup); `lib/fast_sim.py`
(`_FakeEnv.comet_path_cache` + clone shares it);
`tests/test_game_parity.py` (cache-HIT test);
`scripts/profile_step.py` (the harness).

## Phase 2 — pure-Python game-engine rebuild (previous session)

Done in one pass: `lib/game/interpreter.py` is a verbatim port of
`kaggle_environments.envs.orbit_wars.orbit_wars.interpreter` (812 src
lines → 569 LOC port; same RNG path, same combat, same termination).
`lib/fast_sim.py` now imports from `lib.game.interpreter` instead of
`kaggle_environments`. The bundler inlines `lib/game/interpreter.py`
into every submission bundle (added to DEFAULT_LIB_ORDER ahead of
`fast_sim`; +12 KB / bundle).

Parity gates (all green):

- `tests/test_game_parity.py` — 32 tests: init (8 seeds × 2/4 agents)
  + 60-step shadow (5 × 2/4) + 500-step shadow (3 × 2/4).
- `scripts/full_episode_parity_sweep.py` — 100 × 2P + 50 × 4P × 500
  step random-policy episodes, byte-exact parity.
- Existing gates: `test_fast_sim_parity`, `test_v1_parity`, all bundle
  tests (10/10), full suite 405 passed.

Microbenchmark: ours 1088 µs/step vs Kaggle 1103 µs/step (1.01×).
**Parity is the win this phase; speed is not.** Optimisation is
Phase 3 (vectorise the fleet × planet sweep collision in
`interpreter()` — the O(F·P) hot loop).

Knowledge ref: `knowledge-base/concepts/pure-python-game-rebuild.md`.

## Next phase — Phase 3 candidates

Three orthogonal directions, pick one per session:

1. **Vectorise the hot loop.** Profile `lib/game/interpreter.py`,
   replace the `for fleet in fleets: for planet in planets:` swept-pair
   loop with a numpy batch. Target: ≥ 10× per-step speedup at fleet
   counts ≥ 20. Parity gate stays green or revert.
2. **Batched simulator.** Build `lib/game/batch_interpreter.py` that
   runs N games in parallel via a `(N, P, 7)` ndarray for planets and
   a `(N, F, 7)` ndarray for fleets. Unlocks RL training and large-N
   tournaments. Parity test: per-game outputs match the scalar
   interpreter byte-for-byte.
3. **Drop the kaggle_environments init handshake.** Today we still use
   `make("orbit_wars",...).reset()` to bootstrap. Replace with a
   thin local helper that calls `lib.game.interpreter` in init mode.
   Removes the last kaggle_environments dependency from offline play.

## Out of scope for any session before that work lands

- No new submission from this branch. v7_0_drop_one #52588156 is
  the live anchor; the rolling-last-2 holds.
- No PR to main from this branch yet. Open a PR only when the PI
  authorises promotion.

## Pointers — Phase 2 additions

- `lib/game/interpreter.py` — pure-Python interpreter port
- `lib/game/__init__.py` — package re-exports
- `tests/test_game_parity.py` — shadow-step parity harness
- `scripts/full_episode_parity_sweep.py` — ad-hoc N-seed parity sweep
- `lib/fast_sim.py` — flipped to import from `lib.game.interpreter`
- `scripts/bundle_agent.py` — `game/interpreter` added to
  DEFAULT_LIB_ORDER before `fast_sim`
- `knowledge-base/concepts/pure-python-game-rebuild.md` — concept doc

## Pointers — Phase 1 (consolidation, prior in this session)

- `lib/fast_sim.py`, `lib/opp_model.py`, `lib/v7_search.py`,
  `lib/lookahead_planner.py`, `lib/value_heads.py`,
  `lib/candidate_portfolios.py`, `lib/mirror.py` — fast-brain stack.
- `lib/planner.py`, `lib/orbit.py` — σ-equivariance patches.
- `agents/v7_ablations/v7_0_drop_one/main.py` — anchor agent source.
- `submissions/v7_0_drop_one.py` — anchor bundle sha
  `bb7ab23a75bc5865`.
- `submissions/v4_planner.py` — rolling-last-2 pair.
- `knowledge-base/concepts/lookahead-simulator-architecture.md` —
  fast-brain reference doc (read this first next session).
- `tests/test_replay_parity.py` — xfail note on v3_snipe drift.
- `audit/2026-05-12-v4-planner-receding-horizon-pathology.md` —
  why v4_planner under-performs v7_0 in drop-one regime.
