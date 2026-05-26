# Postmortem — 2026-05-26 PM sa_online warm-start cycle

Branch: `claude/competitive-programming-strategy-ESwSv` (separate
session from the morning's strategic-head work — that postmortem is
at `audit/2026-05-26-postmortem-competitive-programming-strategy-ESwSv.md`).

PI directive for this writeup: **observations, not conclusions.**
This file records what happened; interpretation belongs elsewhere.

## Session shape

~6-hour run focused on the sa_online cascade-aware SA agent. Three
distinct failure modes surfaced sequentially:

1. Per-turn TIMEOUT on the live ladder (subs 53059642, 53061384).
2. Agent-doesn't-emit failure after the timing fix (sub 53062327).
3. Local smoke false-positive — vs `random` opponent the agent
   looked fine; vs the peak baseline it was eliminated step 79.

Four commits landed and four Kaggle submissions were pushed.

## Numerical observations

### Sub-by-sub trajectory

| Sub | Commit | Local indicator | Live outcome |
|---|---|---|---|
| 53059642 | (earlier) | cascade-SA, no fate cache | TIMEOUT |
| 53061384 | `7d1ab2f` | tighter budget | TIMEOUT step 357/500 |
| 53062327 | `6147dcb` | fate cache + snap cache; 500-step smoke vs random "survived" | ERROR-equivalent (see note below) |
| 53063161 | `147d6aa` | warm-start + opp=noop; 200-step smoke vs peak: 69 emits, eliminated step 134, mean 284ms | PENDING at session close |

Note on 53062327: PI reported "submission failed" in chat. Exact
live μ not captured in this session's logs.

### Per-turn timing trace (sub 53061384 from `kaggle competitions logs`)

| turns | mean duration | over-1s turns |
|---|---|---|
| 1-25 | 0.634s | 2 |
| 26-50 | 0.713s | 3 |
| 51-75 | 0.937s | 8 |
| 76-100 | 0.900s | 6 |
| 101-125 | 0.963s | 8 |
| 126-150 | 1.139s | 19 |
| 151-200 | 1.10s | 14 |
| 201-300 | 1.13s plateau | ~20 |
| 301-357 | rising | — |

### cProfile result (single warm-cache agent() call)

```
0.398s _refine_step (total)
0.357s   _build_perturb_context (90%)
0.310s     _populate_admissible_set
0.308s       _compute_capture_emission_from_edge (×184)
0.351s         predict_fleet_fate (×46)
0.257s           predict_relative (×147,973)
```

The hot path is admissibility re-validation, not the forward sim I
had assumed.

### Floor cost after fate cache + snap cache (commits 4165227 + 6147dcb)

Local floor test, `SA_BUDGET_STEP_S=0.001` (effectively no SA work):
- v2 (pre-fix): ~210ms / turn mean
- v3 (post-fix): 88ms / turn (turns 5-20 with cache warm)

### Decision-quality smoke vs peak baseline (seed 7542, 200 steps)

| Variant | Emits total | Active turns | Game length | Mean dur |
|---|---|---|---|---|
| v4 (sub 53062327) | 5 | 4 | eliminated step 79 | 1435ms |
| v5 first cut (warm-start, opp=nearest) | 7 | 6 | eliminated step 80 | 1342ms |
| v5 (no co_evolve, opp=nearest) | 16 | 13 | eliminated step 118 | — |
| v5 (no co_evolve, opp=noop) | 62 | 40 | eliminated step 121 | — |
| v5 final (co_evolve on, opp=noop) | 69 | 50 | eliminated step 134 | 284ms |

Peak baseline on the same seed: 30+ planets owned by mid-game.

### Test suite

- Pre-session: pre-existing failure in
  `test_admissible_set_only_physics_valid` (4-seed test poisoned by
  shared `_FATE_CACHE` from commit 4165227 earlier in the day).
- Added autouse fixture `_reset_fate_cache_between_tests`.
- Added 6 new tests for warm-start invariants.
- End-of-session: 25/25 green.

## Procedural observations

- **Architecture conclusion before diagnostic.** From the per-turn
  growth pattern I concluded SA was structurally too slow. PI
  overrode; cProfile then localised 90% of cost to one function
  not in my architectural model.

- **Code review skill, 5 angles in parallel.** Two of five angles
  converged on the same finding (`_populate_admissible_set` work).
  One angle's specific hypothesis (`_populate_planet_position_cache`
  is the dominant cost) was wrong on micro-bench but pointed at the
  same file.

- **Smoke choice changed visible failure modes.** vs random:
  agent's idleness invisible because random is also idle. vs peak:
  agent's idleness immediately visible because peak's expansion
  ramps the game-state complexity.

- **Source-budget bug was numerically silent.** First warm-start
  used `ownership_cache[t_start]`. Smoke result was identical to
  the no-warm-start baseline (within emit-count noise). Bug was
  only found after forcing `SA_COEVOLVE_CYCLES=0` to isolate the
  warm-start path.

- **Local-vs-Kaggle environment divergence.** `_co_evolve` runs
  locally (35s turn 0) but fails on Kaggle. Smokes were testing a
  code path that doesn't ship.

- **Fate cache key correctness.** First version was
  `(src, tgt, t_dep)` — broke a multi-seed test because cached
  outcomes carry between episodes' planet geometries. Fixed by
  adding `ships_bucket` (because `fleet_speed(ships)` affects
  swept-hit step length) and an explicit test-fixture reset.

- **Test reset fixture was the right level of fix.** The cache
  key is correct for Kaggle's one-episode-per-process model;
  adding episode-seed to the key would have made the cache
  pointlessly larger. The reset hook (`reset_fate_cache()`) cleans
  between tests without changing production behavior.

- **Four submissions to converge.** Each commit was rebundled and
  submitted before the next failure mode was diagnosed. The
  diagnostic loop ran on the live ladder rather than locally.

## What changed in the repo

- `lib/sa_core.py`: `_FATE_CACHE`, `reset_fate_cache()`,
  `_ships_bucket()`, `_warm_start_from_admissible()`. Reordered
  ctx-build before initial-score in `simulated_anneal_online`.
- `agents/sa_online/main.py`: `_CACHED_SNAP`,
  `_refresh_snap_from_obs`, `_get_or_refresh_snap`. Defaults
  `SA_REFINE_OPP_POLICY=noop`, `SA_WARM_START_MAX=16`.
- `tests/test_sa_core.py`: autouse fate-cache reset fixture, 6
  new warm-start tests, fix to multi-seed test loop.
- `submissions/sa_online_v5.py`: bundle of HEAD `147d6aa`,
  submitted as sub 53063161.
- `knowledge-base/thoughts/2026-05-26-sa-online-warm-start-cycle.md`
- `knowledge-base/questions/2026-05-26-sa-online-warm-start-cycle.md`
- `knowledge-base/flags/2026-05-26-sa-online-warm-start-cycle.md`
- `audit/friction.md`: appended PM section under today's heading.

## What remains unanswered

- Live μ for 53062327 and 53063161.
- Whether the warm-start emissions that SA preserves correspond to
  the closed-form top-K by `_capture_value`, or whether SA's
  preserve-set diverges further from the ranking under forward-sim
  scrutiny.
- The relative contribution of the snap cache (saved ~10ms
  locally) vs the fate cache (~120ms locally) to the live-ladder
  outcome — couldn't separate the two on Kaggle without
  additional submissions.
- Whether numba JIT of `fs_step` or `predict_fleet_fate` is the
  next-largest lever; not measured this session.
