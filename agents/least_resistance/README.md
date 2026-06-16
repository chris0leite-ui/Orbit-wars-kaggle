# least_resistance

A simulation-driven expansion agent. Each turn it builds a coordinated
launch plan by **forward-simulating every candidate move** and keeping only
the launches that improve the simulated outcome — rather than scoring moves
with hand-tuned weights.

## How it decides (plain English)

1. **List the sensible moves.** For every planet we don't own, work out the
   coordinated launch that captures it — from one planet, or several ganging
   up when one isn't enough — using the exact lead-intercept physics
   (`lib/aim`), with a cheap `path_clears_sun` pre-filter. Also list "stream
   idle ships forward" moves. Order the list by the *path of least resistance
   to production*: most production per turn of travel first, ties broken toward
   whatever shortens our distance to the nearest opponent. (Full path safety —
   off-board, wrong-planet, undersized waves — is left to the simulation in
   step 2, which is the exact ground truth.)

2. **Decide by simulation.** For each candidate, play the move and roll the
   game forward ~14 turns with both sides following a fast greedy policy
   (`lib/fast_sim` + `lib/opp_model.lite_greedy_policy`). Score the result as
   *our ships minus theirs* at the horizon (`delta_us_minus_them`). Commit a
   launch only if it improves that score; keep adding launches until nothing
   helps.

This is the same machinery our strongest agents use (`lib/v7_search`, the
baseline trajectory chooser). The leaf value — ships at the horizon — is the
self-calibrating "more production" objective, so there are no strategy
weights to tune.

## Why this beats the hand-tuned version

The first draft scored moves with weighted terms (production / ETA / frontier
/ reserve). It beat `random` 100% but lost to the `nearest` sniper ~6%: it
bled ships on launches that looked good on paper but didn't convert. Replacing
the weighted score with forward simulation lifted it to ~94% vs `nearest`.
Reserves, gang-up vs solo, attack-vs-expand, and "don't bleed ships" all fall
out of the simulation for free.

## What emerges from the simulation (not coded as rules)

- **Reserves:** draining a planet the rollout shows the opponent then captures
  lowers our score, so we keep exactly the reserve worth keeping.
- **Gang-up:** a partial wave that doesn't capture shows no gain; a coordinated
  wave that does gets committed as a unit.
- **Attack vs. expand:** whichever simulates to the larger ship advantage.
- **Accumulate:** if no launch beats doing nothing, we hold and bank
  production.

## Parameters

Only **compute bounds** (not strategy tuning), env-overridable for
benchmarking:

| var | default | meaning |
|---|---|---|
| `LR_SIM_HORIZON` | 14 | turns of forward simulation per candidate |
| `LR_WALLCLOCK_MS` | 700 | per-turn rollout budget (bails the greedy loop) |
| `LR_MAX_CANDIDATES` | 24 | most candidate moves simulated per turn |
| `LR_VALUE_EPS` | 1.0 | min simulated ship gain (ships) to commit a launch |

## Run

```
python fast.py smoke agents/least_resistance        # vs random + nearest
python fast.py bench agents/least_resistance         # per-turn ms
python fast.py eval  agents/least_resistance --vs-panel
python fast.py play  agents/least_resistance --seed 7
```

## Bundle (single-file submission)

The shared `DEFAULT_LIB_ORDER` is missing `kinematic_table` (a module
`lib/trajectory.py` lazily imports), so pass an explicit `--lib` list with it
added after `orbit`:

```
python scripts/bundle_agent.py agents/least_resistance --skip-parity-gate \
  --lib geometry fleet orbit kinematic_table aim combat world_model intent \
        trajectory mechanism mission scoring missions/snipe missions/reinforce \
        planner game/interpreter fast_sim opp_model
# -> submissions/least_resistance.py
```

Parity (source ≡ bundle) is verified with `ORBIT_WARS_PARITY_WALLCLOCK_MS`
set high so the wallclock bail can't introduce timing nondeterminism (the
agent reads that override at call time). Built bundle: 34/34 turns matched.

## Standing (2026-06-16)

| opponent | result |
|---|---|
| `random` | 32/32 (100%) |
| `nearest` | 30/32 (94%) |
| `v7_0` (our tuned champion) | 1/8 (12%); longer horizon K=26 → 0/8 |

Timing (single process): p50=74ms, p95=400ms, max=682ms, zero turns ≥1000ms.

It decisively beats the competition baselines but trails our heavily-tuned
champion. The longer-horizon-is-worse result shows the ceiling is the rollout
policy (`lite_greedy`), not foresight: to compete with `v7_0`/producer the
rollout needs a stronger — but still cheap — evaluator (e.g. a learned value
head or the producer's `orbit_lite` scorer). That's a larger change, flagged
for a decision rather than done blind.
