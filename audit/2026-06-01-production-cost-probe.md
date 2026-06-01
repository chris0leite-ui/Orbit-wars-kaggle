# production-cost probe — 2026-06-01

> Source: `scripts/production_cost_probe.py` (seed=42, episodeSteps=500, baseline-vs-baseline).
> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch `lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` to catch already-imported references.

## Per-turn cost (all seats merged)

```
  [ALL]  n_turns = 536
    agent_ms     median   81.25  mean   91.38  p95  237.05
    sim_ms       median   51.30  mean   48.11  p95  118.53
    non-sim_ms   median   25.39  mean   43.27  (policy + proposer + bookkeeping)
    clone_ms     median    0.28  mean    0.68
    sim_steps    median   355.0  mean   810.5
    ms/sim_step  median   0.059
    ms/clone     median   0.017
```

```
  [early]  n_turns = 332
    agent_ms     median   81.25  mean   91.32  p95  247.25
    sim_ms       median   53.19  mean   51.21  p95  138.82
    non-sim_ms   median   25.39  mean   40.11  (policy + proposer + bookkeeping)
    clone_ms     median    0.28  mean    0.52
    sim_steps    median   355.0  mean   732.0
    ms/sim_step  median   0.081
    ms/clone     median   0.018
```

```
  [mid]  n_turns = 204
    agent_ms     median   51.82  mean   91.48  p95  216.42
    sim_ms       median    4.81  mean   43.08  p95  108.37
    non-sim_ms   median   42.89  mean   48.40  (policy + proposer + bookkeeping)
    clone_ms     median    0.61  mean    0.92
    sim_steps    median    42.0  mean   938.2
    ms/sim_step  median   0.045
    ms/clone     median   0.017
```

```
  [late] empty
```

## Headroom analysis

```
  Per-turn medians:  agent 81.3 ms  =  sim 51.3 ms  +  non-sim 25.4 ms
  Median sim-step count per turn:  355
  Mean ms per sim step (in production traffic):  0.059 ms
  Remaining of 950 ms budget after current chooser:  869 ms

  Headroom counterfactuals (median turn):
    if sim were 2x cheaper:   turn = 51 ms  →  +26 ms free
    if sim were 10x cheaper:  turn = 31 ms  →  +46 ms free
    if non-sim (policy) were 2x cheaper:  turn = 64 ms  →  +13 ms free
    if non-sim (policy) were 10x cheaper: turn = 54 ms  →  +23 ms free

  At current cost 0.059 ms/sim_step, 950 ms budget = 16002 sim steps theoretical max.
  Today we use 355 (=21 ms) — leaving headroom only after non-sim costs settle.
```
