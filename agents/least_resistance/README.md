# least_resistance

A simulation-driven expansion agent. Each turn it builds a coordinated
launch plan by **forward-simulating every candidate move** and keeping only
the launches that improve the simulated outcome — rather than scoring moves
with hand-tuned weights.

## How it decides (plain English)

1. **List the sensible moves.** For every planet we don't own, work out the
   coordinated launch that captures it — from one planet, or several ganging
   up when one isn't enough — using the exact physics (`lib/aim`,
   `lib/trajectory`). Also list "stream idle ships forward" moves. Order the
   list by the *path of least resistance to production*: most production per
   turn of travel first, ties broken toward whatever shortens our distance to
   the nearest opponent.

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

```
python scripts/bundle_agent.py agents/least_resistance
# -> submissions/least_resistance.py  (default lib order inlines all deps)
```
