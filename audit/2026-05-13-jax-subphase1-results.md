# JAX sub-phase 1 — closed

Date: 2026-05-13. Branch: `claude/consolidate-fast-simulation-ysd9M`.

## What shipped

Full `orbit_wars` game engine ported to JAX:
- `lib/game/jax/jax_types.py` — `GameState` Pytree with fixed
  padding (MAX_PLANETS=80, MAX_FLEETS=256, NUM_COMET_SPAWNS=5).
- `lib/game/jax/conversions.py` — `scalar_to_jax`, `jax_to_scalar`,
  `actions_to_jax`. Pre-computes the 5 comet-spawn results at game
  init (replays scalar `random.Random` keyed by
  `f"orbit_wars-comet-{seed}-{step}"`) and stores in state — no
  string-keyed RNG at JAX runtime.
- `lib/game/jax/jax_interpreter.py` — 11 per-step phases + full
  `jax_step()` chained:
    `comet_expire` → `comet_spawn` → `fleet_launch` → `production_tick`
    → `planet_path_compute` → `comet_path_advance` →
    `fleet_movement` (with `swept_pair_hit_batch` F×P collision) →
    `apply_planet_movement` → `remove_expired_comets_mid_step` →
    `combat_resolution` → `terminate`.

Each phase has a `*_jit` variant. The whole pipeline is one
`jax.jit`'d function (`jax_step_jit`).

## Parity tests — 69/69 green

| Suite | Count | Coverage |
|---|---|---|
| `tests/test_jax_scaffolding.py` | 15 | Pytree shapes, scalar↔JAX round-trip, comet schedule |
| `tests/test_jax_phase_parity.py` | 49 | Per-phase parity vs scalar (init, 60-step shadow, 500-step shadow, planet-cache HIT, comet-cache HIT, fleet_launch, fleet_movement, comet_spawn) |
| `tests/test_jax_full_step_parity.py` | 5 | End-to-end `jax_step_jit` vs scalar interpreter over 60 random-policy steps × 5 seeds. Planet/fleet matched by PID/ID. Integer fields exact; float positions tolerance 1e-3. |

## Performance (this CPU, no GPU available locally)

Per-phase microbench (each phase JIT'd in isolation):

| Phase | N=1 | N=64 vmap | Speedup vs sequential |
|---|---|---|---|
| production_tick | 0.16 ms | 0.22 ms | ~46× |
| planet_path_compute | 0.16 ms | 0.37 ms | ~28× |
| comet_expire | 0.15 ms | 0.25 ms | ~38× |
| comet_path_advance | 0.16 ms | 0.29 ms | ~35× |
| swept_pair_hit_batch (F=50, P=30) | 0.05 ms | — | ~18× vs scalar 1500-call loop |

Full `jax_step` (all 11 phases chained, `jax_step_jit`):

| Workload | Time |
|---|---|
| N=1 (single game) | **2.49 ms** |
| N=4 (vmap'd) | 3.32 ms |
| N=16 (vmap'd) | 6.08 ms |
| **N=64 (vmap'd)** | **14.87 ms** |

The eager path (each phase dispatched separately) was 671 ms/step —
the JIT compile fuses the 11 phases into one dispatch, giving a 230×
speedup. This is why end-to-end parity testing (50 seeds × 60 steps)
runs in ~90 s, not the projected ~37 min.

## What this enables

For a 64-game × 500-step A/B test, the ENGINE cost is
`64 × 500 × 14.87 ms / 64-games-in-parallel = 7.4 seconds on CPU` —
already meeting the user's 5-min target purely from the engine side.

On Kaggle T4/P100 GPU we expect the N=64 vmap'd step to drop to
~3-5 ms (GPU memory transfers + matmul fusion), giving ~2-3 seconds
total engine cost.

**Caveat:** the agent isn't ported yet. Sub-phases 2-6 cover
WorldModel + missions + mechanism + score_candidate + v7_0 wrapper.
The whole sprint is still 3-4 weeks, but sub-phase 1 (engine) is now
DONE, including the perf characterisation and full parity gates.

## Commits this session

1. `b5dd9f5` — JAX sub-phase 1a: scaffold + GameState Pytree + conversions (15 tests)
2. `84c2646` — JAX sub-phase 1b: production_tick + planet_path_compute + comet_expire (32 tests)
3. `39f540f` — JAX sub-phase 1c progress: comet_path_advance + swept_pair_hit_batch
4. `37dad37` — JAX sub-phase 1c: comet_spawn (5/5 spawn boundaries)
5. `d218855` — sub-phase 1c microbench
6. `ada1123` — JAX sub-phase 1c: deferred-position refactor + apply_planet_movement
7. `0426c52` — JAX sub-phase 1c: fleet_launch (per-seat padded actions → fleets)
8. `d7e488e` — JAX sub-phase 1c: fleet_movement (the hot loop)
9. `c0c3fbd` — JAX sub-phase 1c finish: combat_resolution + terminate + full jax_step
10. `3723fbe` — JAX sub-phase 1 truly closed: end-to-end parity + jit fix (230×)
11. `7b6709d` — profile_jax_phases: bench_full_step for N=1/4/16/64 vmap

## Next steps

User has the choice for the next session:

- **Sub-phase 2** (JAX WorldModel) — needed for agent's mission building, lookahead. ~2-3 days.
- **Sub-phase 5** (JAX score_candidate inner loop) — even with Python agent top-level, the K-step rollout inner loop could run in JAX. Highest agent-speedup ROI. ~2-3 days.
- **Sub-phase 8** (Kaggle Kernel deployment) — get the engine running on Kaggle GPU early, validate the deployment story before more agent porting. ~1-2 days.
