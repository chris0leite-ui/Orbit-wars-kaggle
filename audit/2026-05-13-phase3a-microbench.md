# Phase 3a — scalar speedup microbench

> Measured 2026-05-13 on the consolidate branch after the five-wave
> scalar speedup pass landed. CPU: whatever the dev container provides
> (consistent within this session; absolute numbers are box-dependent,
> relative numbers transfer).

## Baseline (post-Phase 2, before Phase 3a)

cProfile, 2000 random-policy 2P steps:

```
interpreter only: 1224 µs/step
fast_sim.step():  1465 µs/step  (includes env.reset() noise from terminated episodes)
```

Top hot spots inside `interpreter()`:

| Function | tottime | % | calls |
|---|---|---|---|
| swept_pair_hit | 2.40 s | 28 % | 3.0 M |
| generate_comet_paths | **1.70 s** | **20 %** | **20** |
| distance | 0.64 s | 8 % | 1.4 M |
| interpreter body | 2.24 s | 26 % | 2,000 |
| math.cos/sin/sqrt | 0.52 s | 6 % | 7 M |

**The hidden cliff:** generate_comet_paths costs ~85 ms × 5 spawn
boundaries per episode. An agent at step 49 running 50 candidates of
K=10 lookahead spends ≈5 s on comet generation alone — silently over
the 1 s/turn budget.

## After Phase 3a

Same harness, 5000 steps:

```
interpreter only: 983 µs/step   (1.24× faster, -20 %)
fast_sim.step():  1456 µs/step  (mostly unchanged — same env.reset() noise)
```

Standalone clone() bench (no env.reset() noise): **12.7 µs/call** on a
mid-game state (20 planets, 21 fleets). Clone is not the bottleneck;
Wave 2.5 (lighter clone) skipped.

## Realistic agent-turn benchmark

50 candidates × K=10 forward sim, random-policy rollouts.

| Scenario | Total time | Per-rollout |
|---|---|---|
| **Step 49 (crosses step-50 spawn boundary)** | **155 ms** | **3.1 ms** |
| Step 99 (no spawn crossing, 17 fleets) | 97 ms | 1.9 ms |

**Spawn-crossing-turn comparison** vs Phase-2-pre-cache (estimated
from the 85 ms × 50 candidates that *would* have hit
generate_comet_paths):

| Phase | Spawn-turn cost (K=10 × 50 cands) |
|---|---|
| Phase 2 (no cache) | ~5,000 ms (timeout) |
| Phase 3a (cache) | **155 ms** (~32× faster) |

This is the headline result: the spawn cliff is gone. Wide+deep
lookahead is now viable on every turn of the game.

## Where the wins came from

1. **Comet-path cache** (Wave 1) — eliminates the spawn cliff. Single
   biggest win by far. The interpreter consults
   `env.comet_path_cache[(episode_seed, spawn_step)]` before calling
   `generate_comet_paths`; first rollout's miss computes and stores;
   all subsequent rollouts in the same agent turn hit the cache.
   Shared across clones via the `_FakeEnv` reference (not
   deep-copied in `clone()`).
2. **AABB prune in swept_pair_hit** (Wave 2.1) — bounding-box reject
   on every (fleet, planet) pair before the discriminant math.
3. **Local hoists + math aliases** (Wave 2.2/2.3) — `_cos`, `_sin`,
   `_sqrt`, `_log`, `_atan2` at module level. `planets_local`,
   `fleets_local`, `comets_local` hoisted inside the per-step block.
   Reduces attribute lookups in the hot loop.
4. **planet_by_id dict** (during Wave 2.2) — replaces the per-comet
   linear scan `next(p for p in obs0.planets if p[0] == pid)` with
   O(1) dict lookup.
5. **id()-based set removal** (Wave 2.4) — `obs0.fleets = [f for f in
   fleets if id(f) not in remove_ids]` replaces the O(N·M·7)
   list-equality scan.

## Verification

- `pytest tests/test_game_parity.py -q` — 42 tests (32 base + 10 new
  cache-HIT parity) all green.
- `pytest tests/test_fast_sim_parity.py tests/test_v1_parity.py -q` —
  green.
- `pytest -q --ignore=tests/test_replay_parity.py` — pending confirm
  at wrap-up time.
- `scripts/full_episode_parity_sweep.py --seeds-2p 100 --seeds-4p 50`
  — pending confirm at wrap-up time.

## Phase 3b candidates (next time)

- **Numpy-vectorised swept_pair_hit.** Stack planets to `(P, 4)`
  ndarray per step; broadcast vs fleet segment. Estimated 5–10× on
  fleet movement. Parity risk (numpy vs math.* float semantics).
- **Vectorised generate_comet_paths dense-sample loop.** Inner 5000-
  iter cos/sin/sqrt loop is trivially vectorisable. Less urgent now
  that the cache amortises the cost.
- **Cython compile** of the three numeric helpers
  (`swept_pair_hit`, `distance`, `point_to_segment_distance`).
  Big wins (~10–50×) but Kaggle-runtime compatibility unproven.
- **Batched interpreter** (`lib/game/batch_interpreter.py`) for
  offline RL/training. Separate workflow.
