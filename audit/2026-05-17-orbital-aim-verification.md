# 2026-05-17 — predict_relative semantics verification (Layer 1 of v11)

Branch: `claude/recover-main-foundations-MV0e2`
Status: verification step BEFORE the Layer 1 revert.

## Why this exists

Commit `7cd7a3f` ("v10 fix: orbital offset wait_N-1 in _aim_and_eta
predict_relative call") changed `agents/v8_scavenge/main.py:200` from

    lead = wait_N

to

    lead = max(0, wait_N - 1)

claiming "the env's actual position at env-step N equals
predict_relative(N-1)" and citing Felipe seed `1492346051` turn 15 as
empirical proof.

That empirical proof is suspect — Rule 38 says fix-verification has to
reproduce the failure state. The cited check appears to have used
predict_relative from `env.steps[0]`, where `env.steps[0]` and
`env.steps[1]` are positionally identical (the env initializes BEFORE
any rotation), so a 1-step off-by-one is undetectable.

## The test

Run from `env.steps[15]` (mid-game), advance 6 idle steps to
`env.steps[21]`, and ask: which `predict_relative` lead matches?

```python
from kaggle_environments import make
from lib.orbit import predict_relative, is_orbiting
import math

env = make('orbit_wars', configuration={'seed': 1492346051})
env.reset(num_agents=2)
omega = env.steps[0][0]['observation']['angular_velocity']
for _ in range(15):
    env.step([[], []])
obs_at_15 = env.steps[15][0]['observation']
for _ in range(6):
    env.step([[], []])
obs_at_21 = env.steps[21][0]['observation']

for p15, p21 in zip(obs_at_15['planets'], obs_at_21['planets']):
    if not is_orbiting(p15):
        continue
    pr_6 = predict_relative(p15, omega, 6)
    pr_5 = predict_relative(p15, omega, 5)
    actual = (p21[2], p21[3])
    err6 = math.hypot(pr_6[0]-actual[0], pr_6[1]-actual[1])
    err5 = math.hypot(pr_5[0]-actual[0], pr_5[1]-actual[1])
    ...
```

## Results — env.steps[15] → env.steps[21], same seed 1492346051

| planet | actual @ step 21      | err(lead=6) | err(lead=5) |
|---:|---|---:|---:|
| 12 | (42.102, 84.376) | **0.0000** | 1.6934 |
| 13 | (15.624, 42.102) | **0.0000** | 1.6934 |
| 14 | (84.376, 57.898) | **0.0000** | 1.6934 |
| 15 | (57.898, 15.624) | **0.0000** | 1.6934 |
| 16 | (44.949, 72.067) | **0.0000** | 1.0869 |
| 17 | (27.933, 44.949) | **0.0000** | 1.0869 |
| 18 | (72.067, 55.051) | **0.0000** | 1.0869 |
| 19 | (55.051, 27.933) | **0.0000** | 1.0869 |
| 20 | (54.803, 85.771) | **0.0000** | 1.7328 |
| 21 | (14.229, 54.803) | **0.0000** | 1.7328 |
| 22 | (85.771, 45.197) | **0.0000** | 1.7328 |
| 23 | (45.197, 14.229) | **0.0000** | 1.7328 |
| 28 | (31.027, 82.288) | **0.0000** | 1.7980 |
| 29 | (17.712, 31.027) | **0.0000** | 1.7980 |
| 30 | (82.288, 68.973) | **0.0000** | 1.7980 |
| 31 | (68.973, 17.712) | **0.0000** | 1.7980 |

**`lead=6` matches exactly. `lead=5` is off by ~1.7 units (one full
step of rotation at this seed's omega=0.048).**

## Results — env.steps[0] → env.steps[6] (the FLAWED test the commit used)

| planet | actual @ step 6 | err(lead=6) | err(lead=5) |
|---:|---|---:|---:|
| 12 | (66.737, 81.047) | 1.6934 | **0.0000** |
| 13 | (18.953, 66.737) | 1.6934 | **0.0000** |
| 14 | (81.047, 33.263) | 1.6934 | **0.0000** |
| ... (all orbiting planets show same pattern) | | | |

From step 0, `lead=5` looks correct — but this is a measurement
artifact. `env.steps[0]` and `env.steps[1]` are positionally identical,
so 6 advance steps only produce 5 rotations of motion. The off-by-one
disappears from any mid-game starting state.

This matches the documentation in `lib/orbit.py` for `predict_absolute`:
"Uses the empirically-correct N-1 rotation count for N>=1." The env
has the rotation-count = step-count-minus-1 quirk; predict_relative
does NOT — it rotates exactly `lead_turns` of angular motion.

## Implication

`predict_relative(planet@K, omega, wait_N)` already gives the correct
position at env step `K + wait_N` from any mid-game obs. The
"fix" subtracted 1 from `lead`, **introducing** a 1-step aim error on
wait-then-fire candidates.

## Action

Revert `agents/v8_scavenge/main.py:200` from
`lead = max(0, wait_N - 1)` back to `lead = wait_N`. Then run a
controlled cell to confirm v10-vs-v7_0 improves on Felipe seed.

## Why did the commit's claim "v10-vs-v8 went 0/2 → 2/2" land?

Two candidate explanations, neither implying the fix was correct:
1. Eta-discretization (`math.ceil`) coincidence on the single Felipe
   seed: the wrong aim still landed close enough due to planet radius +
   ceil rounding, and downstream chooser changes (not the aim) actually
   produced the win.
2. The fix masks a DIFFERENT off-by-one elsewhere in the wait-phase
   rollout (e.g. inside `fs_step`'s clone of the env's orbit advance).
   If so, the right fix is in fast_sim or aim_orbiting, not in the
   agent's predict_relative call.

Both will be distinguished by the seed-cell comparison.
