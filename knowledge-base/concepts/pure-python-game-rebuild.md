# Pure-Python rebuild of the orbit_wars game engine

> Written 2026-05-12 EVE on `claude/consolidate-fast-simulation-ysd9M`
> after the Phase-2 work shipped. Permanent reference for the PI
> second-brain.

## The one-line version

We now own the game engine. `lib/game/interpreter.py` is a verbatim
pure-Python port of `kaggle_environments.envs.orbit_wars.orbit_wars`'s
`interpreter()` function. Same RNG path, same combat semantics, same
termination logic — byte-exact parity validated on 8 init-seeds, 6 ×
500-step random-policy episodes (full slow suite), and a separate
sweep of 150 × 500-step episodes via
`scripts/full_episode_parity_sweep.py`.

`lib/fast_sim.py` now calls our interpreter instead of Kaggle's. All
405 existing tests (excluding the pre-existing replay-parity xfail)
stay green.

## What this unlocks

- **Iteration freedom.** The game source is in our repo. We can
  instrument it (add per-step intent capture, profile hot loops, fuzz
  state transitions) without monkey-patching upstream code.
- **No `kaggle_environments` dependency for offline play.** Submission
  bundles still depend on `kaggle_environments` because Kaggle's
  runtime provides it — but for tournaments, A/B tests, and any future
  RL loop, we can drop the dependency entirely once we also reroute
  the `make("orbit_wars", ...).reset()` path through our code.
- **Substrate for vectorisation and batched simulation.** Plain-Python
  is the prerequisite — a `BatchSnapshot` that runs N games in
  parallel via numpy is the natural follow-up.
- **Substrate for RL or imitation-learning experiments.** Once batched
  sim is in, training can run at hundreds of games/second without the
  Environment framework overhead.

## What this does NOT change

Per-step cost on the local box: our interpreter is **1.01× the speed
of Kaggle's** (within noise). The win is parity, not performance.
The 100-µs/step target from the original plan was aspirational on a
faster CPU; the actual machine measures ~1100 µs/step for both
interpreters. Phase-3 optimisation work (locals hoist, vectorised
sweep collision) is on the table whenever we want it.

## Plug-in point

`lib/fast_sim.py` previously imported:

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    interpreter as _orbit_wars_interpreter,
)
```

It now imports:

```python
from lib.game.interpreter import interpreter as _orbit_wars_interpreter
```

The bundler (`scripts/bundle_agent.py`) was updated to inline
`lib/game/interpreter.py` into every submission bundle (added to
`DEFAULT_LIB_ORDER` ahead of `fast_sim`). Bundle size +12 KB.

## Parity tests

| Test | Coverage |
|------|---|
| `tests/test_game_parity.py::test_init_parity` | 8 seeds × 2/4 agents — init state byte-exact |
| `tests/test_game_parity.py::test_shadow_parity_60_steps` | 5 seeds × 2/4 agents × 60 steps (crosses step-50 comet spawn) |
| `tests/test_game_parity.py::test_shadow_parity_500_steps` | 3 seeds × 2/4 agents × 500 steps (all 5 spawn boundaries + termination) |
| `scripts/full_episode_parity_sweep.py` | 100 × 2P + 50 × 4P, 500-step episodes, ad-hoc sweep |
| `tests/test_fast_sim_parity.py` | indirect: now exercises our interpreter via fast_sim |
| `tests/test_v1_parity.py` | indirect: v1 mechanism stack runs through our interpreter |

## Out of scope (still)

- Replacing `make("orbit_wars", ...).reset()` — we still rely on the
  Kaggle env for the init handshake when constructing snapshots. The
  interpreter itself does init when called on an empty state, so the
  rebuild is complete; only the wrapper around it is still Kaggle's.
- Vectorised / batched simulator. Next phase.
- Cython / Numba. Defer.
