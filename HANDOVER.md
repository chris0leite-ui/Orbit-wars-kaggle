# HANDOVER.md — next-session brief

> Last written: 2026-05-13 LATE-2 by `claude/read-handover-iLWTq`.
> Format budget ≤ 160 lines. Prior wraps archived under
> `audit/archive-2026-05-1*-handover-*.md`.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC → **41 days remaining.**
- **Live submitted agent:** `v7_0_drop_one`, **submission #52588156**,
  publicScore **μ=1094.9** (team peak). Public-LB rank **109 / 2587**
  (top 4.2 %). Gap to top-10 prize: +336 μ.
- **Rolling-last-2:** `[v4_planner #52579863, v7_0_drop_one #52588156]`.
- **No new submission this session.** All work is local; ladder anchor
  preserved.

## This session — diagnostic + cheap wins + brute-force search

Plan: `/root/.claude/plans/go-diagnostic-cheap-wins-woolly-rose.md`.

### Track A — loss-mode diagnostic

`audit/2026-05-13-v7-0-loss-modes.md`. 97 live replays pulled via
`scripts/live_episode_summary.py`. 50 losses classified with
`scripts/classify_losses.py` (new, 5 buckets).

| Bucket | Count | Cross-tab (2P / 4P) |
|---|---:|---|
| `opening_lost` | 34 (68 %) | 8 / 26 |
| `mid_economy_lost` | 16 (32 %) | 13 / 3 |
| other 3 buckets | 0 | — |

**Headline:** 4P games are 90 % opening-determined (26/29). H11 is
exactly the fix; ship it.

### Track B — cheap wins (H11 + H15)

- **H11 opening grab.** `propose_opening_missions` is now wired into
  `lib/v7_search.py::_build_incumbent_intents` (1-line addition, plus
  import). The proposer was already built and tested
  (`tests/test_mission_opening.py`); only the wire was missing.
- **H15 comet reject.** `lib/missions/snipe.py:236-243` now hard-rejects
  comet targets where `remaining_lifetime <= eta`. Previously the
  mission was emitted with score≈0, consuming a per-source slot in
  `settle_plan`.
- New tests: `tests/test_mission_opening_wireup.py` (4 cases),
  `tests/test_snipe_comet_reject.py` (5 cases). All green.
- Bundles: `submissions/v7_1_open_drop_comets.py` (209 KB).
- **Scalar A/B (16 seeds × 2 seats = 32 games, 13 min wallclock):**
  v7_1 17W-13L-2D vs v7_0 = **53.1 %, Wilson lo 36.4 %.** Directional
  signal positive but **BELOW the 55 % gate.** No submission justified
  on this sample. **A 64-seed A/B is required** before any push — on
  the Kaggle T4 vmap kernel with the new JAX port that's ~1.5 min.

### Track C — brute-force game-theory search

- **C1 runtime depth-2 maximin** (`lib/v7_search.py::choose_depth2`,
  ~200 LOC). Outer = our drop-one set (≤ 8); inner = opp drop-one set
  (≤ 4) recomputed from post-turn-1 state. Payoff matrix → maximin pick.
  Budget shape 32 cells × ~15 ms = ~500 ms wall. Tests in
  `tests/test_v7_depth2.py` (5 pass).
- **C2 JAX candidate-axis vmap brute scorer**
  (`lib/game/jax/jax_brute_search.py`, ~180 LOC). vmaps
  `score_candidate_jax_pure` across C candidates on a fixed state.
  Used as the leaf engine for offline brute search. Parity test
  `tests/test_jax_brute_search.py` 3 pass (serial vs vmap argmax
  matches, max-diff < 1e-2 float32).
- **C3 N-step path-search oracle**
  (`scripts/jax_path_search.py`, ~270 LOC). 32-seed N=2 oracle ran in
  9 s; **v7_0 single-ply matches the depth-2 maximin choice 85.2 %
  of the time (4/27 disagreements; 3 of 4 say "drop one more launch").**
  Modest but real head-room for depth-2; CSV at
  `audit/2026-05-13-depth2-oracle-32seeds.csv`.
- **JAX port for H11/H15** (`compute_opening_score_matrix` in
  `lib/game/jax/jax_missions.py`; `use_opening` knob on
  `policy_emit_jax_pure` / `rollout_step_jax_pure` /
  `score_candidate_jax_pure`). Smoke parity vs scalar matches
  bit-exactly (seed 42, step 0: src=0 → tgt=8, ships=6, score=1226.79).
  Knob plumbed through `scripts/kaggle_ab_kernel/run_jax_ab.py` via
  `A_USE_OPENING` / `B_USE_OPENING` env vars.
- Bundle: `submissions/v7_2_depth2.py` (209 KB; depth-2 chooser).

## Falsified / risks

- ~~Bundler parity gate fails on v7_1 / v7_2 with non-deterministic
  divergence on 1 / 528 turns. Suspected module-level mutable state.~~
  **FIXED this session (commit `07ef918`).** Real root cause was the
  `time.perf_counter()`-based wallclock watchdog in `lib/v7_search`
  choosers: when the 700 ms budget bailed mid-candidate-list under CPU
  jitter, argmax ranged over different subsets and picked differently.
  Same `obs`, same process — different output. Fix: new
  `_effective_wallclock_ms` helper consults an `ORBIT_WARS_PARITY_
  WALLCLOCK_MS` env var; `_parity_gate` in `scripts/bundle_agent.py`
  sets it to 60000 ms via try/finally. Production-path wallclock is
  unchanged. Verified: end-to-end gate clean on seed 0; multi-seed
  probe sweep (find_parity_bug) clean across 2480 turns on seeds 0–2
  (seed 3 hit a probe timeout under the now-slower unbounded budget;
  no mismatches emitted before cutoff).
- A prior commit (`5ae24a1`) added an exception-safe context manager
  for the `_shared_world_model` attach/detach. That fix is correct on
  engineering grounds (replaces a bare-`del` with try/finally) but
  was a misdiagnosis of the parity bug — kept for the defensive
  cleanup; not load-bearing for parity. See postmortem
  `audit/2026-05-13-postmortem-read-handover-iLWTq.md`.

## Next-session first-actions

1. **Run the JAX A/B for v7_1 vs v7_0 at 64 games on Kaggle T4.**
   `scripts/kaggle_ab_kernel/run_jax_ab.py` now has `A_USE_OPENING=1
   B_USE_OPENING=0` knobs. ~1.5 min wallclock on T4 (vs the 13 min CPU
   run we did). The scalar 32-game result (53.1 %, Wilson lo 36.4 %)
   is underpowered; the 64-game JAX result is the gate decision.
   Bundler parity is now clean — submissions can drop the
   `--skip-parity-gate` workaround.
2. **If JAX A/B clears Wilson lo ≥ 55 %, submit
   `v7_1_open_drop_comets`** (#3rd push, evicts v4_planner from
   rolling-last-2). Decision must be PI-signed; the diagnostic
   confirms H11 is targeting the right loss bucket (68 % opening-lost).
3. **Port `choose_depth2` to JAX** for fast A/B testing of v7_2.
   Nested vmap over (our_cand × opp_cand). ~250 LOC; deferred this
   session.
4. **If v7_1 ships and lifts μ, queue v7_2 (depth-2 chooser) gated by
   a 64-seed A/B vs the new v7_1 incumbent.** The 14.8 % oracle
   disagreement suggests +5-10 pp expected lift if the depth-2 picks
   are systematically better.
5. **Decide on a "reproduce-before-fix" rule for non-deterministic
   bugs.** PI declined to promote this session, but the misdiagnosis
   cost ~1 h. See postmortem
   `audit/2026-05-13-postmortem-read-handover-iLWTq.md` if you want
   to revisit.

## Out of scope for next session

- The five-experiment platform refactor from the previous handover.
  Two of the five (opening, comet) just shipped via Track B; depth-2
  shipped via Track C1. Re-prioritise the remaining three (multi-launch,
  opening classifier, strategy library) only after v7_1 ladder result.

## Pointers — this-session deliverables

- `lib/v7_search.py` — opening wire (line 50, 142); `choose_depth2` +
  `choose_depth2_with_4p` (~200 LOC); `_bind_shared_world_model`
  context manager + `_effective_wallclock_ms` env-var override
  helper.
- `scripts/bundle_agent.py` — `_parity_gate` now sets
  `ORBIT_WARS_PARITY_WALLCLOCK_MS=60000` for the parity comparison
  via try/finally.
- `tests/test_v7_search_shared_model_cleanup.py` — 9 unit tests
  pinning both invariants (5 cleanup, 4 wallclock override).
- `audit/2026-05-13-postmortem-read-handover-iLWTq.md` — this
  session's postmortem.
- `lib/missions/snipe.py:236-243` — H15 hard reject.
- `lib/missions/opening.py` — unchanged (built in prior session;
  unit-tested).
- `lib/game/jax/jax_missions.py` — `compute_opening_score_matrix` +
  H15 reject in `compute_snipe_score_matrix`.
- `lib/game/jax/jax_score.py` — `use_opening` knobs plumbed through.
- `lib/game/jax/jax_brute_search.py` — candidate-axis vmap scorer.
- `scripts/classify_losses.py` — 5-bucket loss-mode classifier.
- `scripts/jax_path_search.py` — offline depth-2 oracle.
- `scripts/bundle_agent.py` — opening added to DEFAULT_LIB_ORDER.
- `scripts/kaggle_ab_kernel/run_jax_ab.py` — `A_USE_OPENING` /
  `B_USE_OPENING` env knobs.
- `agents/v7_1_open_drop_comets/main.py`, `agents/v7_2_depth2/main.py`.
- `audit/2026-05-13-v7-0-loss-modes.md` — loss-mode audit.
- `audit/2026-05-13-depth2-oracle-32seeds.csv` — oracle output.
- `audit/2026-05-13-loss-modes-52588156.csv` — per-game classifier output.
- Tests: `test_v7_depth2`, `test_jax_brute_search`,
  `test_mission_opening_wireup`, `test_snipe_comet_reject`.
