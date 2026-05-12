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

## Next phase — research the fast-simulation rebuild

The user's stated next step is a **100%-accurate pure-Python rebuild
of the orbit-wars game itself**, so we can iterate without the
`kaggle_environments` dependency or its per-step framework overhead.
`lib/fast_sim.py` already bypasses ~99% of the Environment wrapper
but still calls `orbit_wars.interpreter()` from the kaggle package.
The next step replaces THAT.

Research questions for the next session — answer before writing code:

1. **What does `orbit_wars.interpreter()` actually do?** Already on
   disk at `/usr/local/lib/python3.11/dist-packages/kaggle_environments/
   envs/orbit_wars/orbit_wars.py` (~750 lines). Map the entry points
   (`interpreter`, `random_agent`, `starter_agent`,
   `generate_planets`, `generate_comet_paths`, `swept_pair_hit`,
   `renderer`) and the physics constants
   (`COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]`,
   `BOARD_SIZE = 100`, `SUN_RADIUS = 10`, `ROTATION_RADIUS_LIMIT = 50`,
   `PLANET_CLEARANCE = 7`, etc.).
2. **What parity test design do we need?** Run our re-impl side-by-
   side with `interpreter()` on the same `(state, env.info[seed])`
   stream, assert state-for-state equality through full 500-step
   episodes. Already have a fixture corpus
   (`audit/live-episodes/52544634/`, `52532938/`, `SELFPLAY/`)
   that can serve as the gate.
3. **What's the performance target?** `fast_sim.step()` is 0.12 ms
   per simulated step today (Environment overhead removed). A pure-
   Python re-impl that drops the `Struct` boxing and the in-package
   import overhead could plausibly hit 0.05 ms; a numpy-vectorised
   batch could push 10-100× on parallel rollouts.
4. **Vectorise or stay scalar?** A scalar pure-Python re-impl is
   the smallest scope and the simplest parity-test target.
   Vectorising over planets and fleets is a separate, later step
   once the scalar version is locked.

Output for that session: a short design doc + a parity-test rig
+ a stub `lib/sim.py` that imports the real interpreter for now.
Don't write the replacement on day one — research first.

## Out of scope for any session before that work lands

- No new submission from this branch. v7_0_drop_one #52588156 is
  the live anchor; the rolling-last-2 holds.
- No PR to main from this branch yet. Open a PR only when the PI
  authorises promotion.

## Pointers — new or updated this session

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
