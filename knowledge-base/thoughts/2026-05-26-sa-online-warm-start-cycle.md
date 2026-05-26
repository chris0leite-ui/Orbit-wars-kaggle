# 2026-05-26 PM — sa_online warm-start cycle (observations)

Branch: `claude/competitive-programming-strategy-ESwSv`. Separate
session from the morning's strategic-head iteration (own postmortem at
`audit/2026-05-26-postmortem-competitive-programming-strategy-ESwSv.md`).

This file is observations only. No interpretation.

## Sequence of events

1. Sub **53059642** (cascade-aware SA, commit `ccff115`, pushed earlier
   in the day): TIMEOUT.

2. Sub **53061384** (`7d1ab2f` — tightened per-turn budget so Kaggle
   stays inside overage pool): TIMEOUT step 357/500.

3. Per-turn timing trace pulled from `kaggle competitions logs`. Showed
   per-turn duration growing 0.63s (turns 1-25) → 1.13s (turn 125+,
   plateau). The plateau sits 0.13s above the 1s actTimeout, draining
   overage steadily.

4. I declared the SA architecture "structurally too slow" and
   recommended pivoting. PI: "No, there is an option. You simply have
   to find it. Run a code review agent."

5. Code review skill ran 5 finder angles in parallel. Angle E flagged
   `_populate_admissible_set` work that scales with game complexity.

6. `cProfile` on a single agent() call (warm cache, default settings)
   showed:
   - 90% of per-turn budget inside `_populate_admissible_set` →
     `_compute_capture_emission_from_edge` → `predict_fleet_fate`
   - 148,000 calls per turn to `lib.orbit.predict_relative`
   - Re-validating ~184 cascade-graph edges through full-trajectory
     collision checks each turn

7. Two-line fix:
   - `lib/sa_core.py` `_FATE_CACHE` keyed by `(src_id, tgt_id, t_dep,
     ships_bucket)`. First touch validates; subsequent touches
     dict-lookup. Reset hook for between-game state.
   - `agents/sa_online/main.py` `_CACHED_SNAP`. First call: `fs_from_obs`
     populates planet_position_cache (30×501 trig). Subsequent calls:
     mutate planets/fleets/comets/step in place. Caches stay.

8. Host confirmed in Kaggle discussion **700191**: "1 and 3/5 of a CPU,
   ~8GB RAM, wall-clock actTimeout=1s, overrageTime=60s, model-level
   caches will work within an episode." 1/3 to 1/5 of a Ryzen 1700.

9. Sub **53062327** (v4, commit `6147dcb`). Local 500-step smoke vs
   `random` reported survived with 57s/60s overage in pocket. Live
   sub failed.

10. PI asked "what would an expert do?" → "do 1 (inspect what agent
    plays) and explain numba." Local inspection vs the peak baseline
    on seed 7542: sa emitted 5 actions over 79 turns, eliminated step
    79, 0 planets, 0 ships. Numba not installed locally; per host
    discussion it is preloaded on Kaggle.

11. PI direction: "if we focus on closest biggest planets at ETA only
    as warm start?"

12. Plan mode → wrote
    `/root/.claude/plans/wiggly-singing-elephant.md`. Three Explore
    agents found that `_capture_value` (lib/sa_core.py:861) already
    implements `production × (t_end - t_arr) - ship_cost` for the
    ruin-recreate operator. PathGraph already has lookup. Source-
    budget pattern is at `lib/planner.py:settle_plan`. Plan
    approved.

13. Implementation (commit `c419045`): added
    `_warm_start_from_admissible(ctx, current_plan, max_emissions=8)`
    in `lib/sa_core.py`. Reorders ctx-build before initial-score in
    `simulated_anneal_online`, then calls warm-start when
    `len(current_plan) < SA_WARM_START_THRESHOLD` (default 3).

14. First version: source-affordability checked against
    `ownership_cache[t_start]`. At turn 1 home has 10 ships; warm-
    start rejected nearly every candidate. Smoke result was
    numerically indistinguishable from v4 (5 emits → 7 emits).

15. Fixed: check at the emission's own `t_dep` using
    `ownership_cache[t_dep][src].ships - cumulative_spent[src]`. The
    forward-sim'd ownership_cache already includes accrued
    production. Tests rewritten to match new semantics. 25/25 green.

16. Second smoke still showed 7 emits / eliminated step 80 / mean
    1342ms. Discovered `_co_evolve` runs at module load locally
    (turn 0 = 35s) and pre-fills `_PLAN_BY_TURN`, so by turn 1 the
    carryover plan is non-empty and warm-start doesn't fire.

17. Set `SA_COEVOLVE_CYCLES=0` for the smoke to mirror Kaggle's
    blocked-co_evolve behavior. Result: 16 emits / 13 active turns
    / eliminated step 118.

18. Tried `SA_REFINE_OPP_POLICY=noop`: 62 emits / 40 active turns /
    eliminated step 121. The agent now actually plays.

19. Set `noop` and `SA_WARM_START_MAX=16` as production defaults
    (commit `147d6aa`).

20. Local 200-step smoke vs peak baseline (co_evolve enabled, since
    that's how the bundle ran when bundled): 69 emits / 50 active
    turns / eliminated step 134 / mean 284ms / max 593ms.

21. Sub **53063161** (v5) submitted. PI direction was "Ship v5 as-is
    now."

## End-of-session state

- Rolling pair: `{53062327 sa_v4, 53063161 sa_v5}`. Both PENDING at
  session close. v4 → expected ERROR or very low μ. v5 → expected
  μ unknown, smoke showed agent plays but loses to peak by
  production deficit.
- Submission slots used today (this session): 4. Daily budget 5.
- Test suite: 25/25 green including 6 new warm-start tests + a
  module-level fate-cache reset autouse fixture.
- Plan file `/root/.claude/plans/wiggly-singing-elephant.md` was
  rewritten this session, overwriting the prior plan for the
  already-merged cascade-aware-ALNS work.

## What was committed

- `4165227` perf(sa_online): fate cache + snap cache
- `6147dcb` fix(sa_online): include ships in fate cache key (log2 bucket)
- `c419045` feat(sa): warm-start from cascade-aware admissible set
- `147d6aa` config(sa): opp_policy=noop + warm-start max=16

## What I assumed without measuring

- That the per-turn cost growth in sub 53061384's log was driven by
  forward-sim cost scaling with game-state complexity. cProfile
  later showed the dominant cost was admissibility re-validation,
  not forward sim.
- That `_populate_planet_position_cache` was the dominant per-turn
  cost. A direct micro-bench showed it was ~2ms locally; the
  148k-call inner loop in `predict_relative` was the actual hot
  path.
- That the long-game smoke vs `random` was sufficient evidence of
  Kaggle-fitness. It wasn't, because random's idleness made our
  agent's idleness invisible.
- That changing the warm-start function alone would suffice. It
  was masked by `opp_policy=nearest` removing most warm-start
  emissions in SA's score loop.

## What surfaced about the environment

- Host discussion 700191: 1.6 CPUs, ~8GB RAM, wall-clock timeout,
  fresh process per episode, module-level caches valid within an
  episode, numba/numpy/scipy preloaded.
- Local dev box is 3-5× faster than Kaggle's grading box (one
  competitor self-reported their Ryzen 1700 at 3-5× the
  tournament server).
- `_co_evolve` runs locally but fails silently on Kaggle (recursive
  `make()` blocked). This means local smokes that exercise the
  first turn behave differently from Kaggle.

## Pointers for the next session

- Live μ data for 53062327 and 53063161 should be visible by next
  session start.
- The next-largest lever after warm-start is probably numba JIT of
  `fs_step` / `predict_fleet_fate` (numba preloaded on Kaggle).
- The cascade-aware admissibility is currently being under-used by
  the warm-start (only single-step captures land in the top-K by
  `_capture_value`).
- Plan file at `/root/.claude/plans/wiggly-singing-elephant.md`
  contains the warm-start design specifics if a follow-up needs
  the exact insertion point.
