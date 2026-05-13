# `env.step()` and `env.clone()` cost anatomy

> Source: derived 2026-05-12 from
> `/usr/local/lib/python3.11/dist-packages/kaggle_environments/core.py`
> and `.../envs/orbit_wars/orbit_wars.py`. Microbench numbers in
> `audit/2026-05-12-fast-sim-bench.md`.

## Why this is worth knowing

The lookahead probes
(`audit/2026-05-11-lookahead-phase{1a,1b,2}-*`) found that
**`env.clone()+env.step()` is the only currently-shipping path that
reaches Sim<K=50> AUC ≈ oracle (0.952)**, but at 5.6–22 ms/step it
caps lookahead breadth at ~2–5 candidates per real turn. The cost is
*almost entirely framework overhead*; the game physics inside
`interpreter()` is microseconds. Future agents will live or die by
how much of that overhead they can skip.

## Per-step cost breakdown (env-cold, before warmup)

| Phase | Where | Roughly |
|---|---|---|
| Action schema validation | `core.py:262`, `process_schema()` | ~1 ms (every action dict validated against JSON schema) |
| Interpreter call (the actual physics) | `interpreter()` in `orbit_wars.py:334-717` | ~0.5–1 ms |
| `structify()` wrapping | `core.py:600`, recursive `Struct(**)` | ~0.3 ms |
| State-history append | `core.py:277-278`, `self.steps.append(...)` | <0.1 ms initially; grows |
| Status & reward update | `core.py:272-275` | <0.1 ms |
| Other (logs, structuring per-agent action_state) | `core.py:246-249` | <0.5 ms |

Cold env per-step: ~5–6 ms (matches Phase 2 audit:91-93).

## `env.clone()` cost — it's NOT really a clone

`Environment.clone()` (`core.py:527-536`) does NOT deep-copy:
- It passes `steps=self.steps` (a *reference*).
- It re-invokes `__init__`, which calls `__set_state(steps[-1])`.
- `__set_state` runs `process_schema()` per agent state — a JSON-schema
  validation pass that costs ~3 ms.

Once the env has been advanced N steps, `self.steps` has N+1 elements.
Clone-time scales linearly: empirically ~22 ms after 20 warmup steps
on the orbit_wars env.

## What `fast_sim` skips (and why it's bit-safe to skip)

`lib/fast_sim.py::step()` calls the env's `interpreter()` directly,
operating on a minimal `Snapshot` dataclass. It bypasses:

1. **Action schema validation** — the interpreter's `process_moves`
   already validates each move's shape, planet ownership, and ship
   count (`orbit_wars.py:478-489`). Malformed actions are silently
   dropped, same as before.
2. **`structify()` wrapping** — `fast_sim` keeps state as `Struct`
   instances natively (Struct is just a dict+attr access shim), so
   there's no per-step recursive copy.
3. **State-history append** — the lookahead inner loop discards
   intermediate states anyway; carrying the history adds memory
   pressure with no benefit. If a future consumer wants a history,
   they can clone the snapshot at each tick.
4. **`Environment.clone()`** — `fast_sim.clone()` does a targeted
   element-level copy of just the mutating fields
   (`planets`/`fleets`/`comets`/`comet_planet_ids`/scalars). 9 µs
   median, vs. 22 ms for `env.clone()` in mid-game.

Net: **183× per-step speedup** on a warmed-up env
(`audit/2026-05-12-fast-sim-bench.md`).

## What `fast_sim` *cannot* skip

- The interpreter's own linear `next(p for p in obs0.planets if p[0]
  == pid)` searches at `orbit_wars.py:486, 553, 637`. These are
  O(n_planets) per call, but rewriting them requires forking the
  interpreter — explicitly out of scope for the foundation. A future
  `lib/fast_sim_native.py` will index these (target: 0.05 ms/step).
- The comet-spawn RNG path — the interpreter constructs
  `random.Random(f"orbit_wars-comet-{seed}-{step+1}")` per spawn step
  (50/150/250/350/450). Faithfully seeded, so deterministic given
  `episode_seed`. The Snapshot carries `episode_seed` for exactly
  this reason. Without it (live ladder), comet spawns past the first
  boundary diverge.

## Implications for future search consumers

At 0.12 ms/step (fast_sim warm), the per-turn budget allows
~7000 step evaluations (with overhead headroom).

- A depth-2 PIMC search with 8 opponent samples × 8 our-action
  candidates × K=30 rollout depth ≈ 1920 step calls × 0.12 ms = 230 ms.
  Fits comfortably under 1 s `actTimeout`.
- A beam search with width 16 at K=50 = 800 step calls × 0.12 ms = 96 ms.
- The remaining time can fund either deeper rollouts (bigger K), more
  candidates, or a learned-value head call per leaf.

The Phase 2 probe at K=50 self-play hit AUC 0.952 = oracle. With this
budget we can confidently reach for *strategic*-quality lookahead, not
just *predictive*-quality. The candidate enumerator is the next ceiling
(audit/2026-05-11-v3-lookahead-mvp-parity.md §"What's needed for lift").
