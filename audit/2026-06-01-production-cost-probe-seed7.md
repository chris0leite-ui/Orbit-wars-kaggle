# production-cost probe — 2026-06-01

> Source: `scripts/production_cost_probe.py` (seed=7, episodeSteps=500, baseline-vs-baseline).
> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch `lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` to catch already-imported references.

## Per-turn cost (all seats merged)

```
  [ALL]  n_turns = 520
    agent_ms     median  230.25  mean  183.34  p95  427.53
    sim_ms       median  117.07  mean   98.89  p95  267.21
    non-sim_ms   median   87.98  mean   84.45  (policy + proposer + bookkeeping)
    clone_ms     median    0.69  mean    0.98
    sim_steps    median   859.5  mean  1244.8
    ms/sim_step  median   0.083
    ms/clone     median   0.018
```

```
  [early]  n_turns = 332
    agent_ms     median  230.25  mean  204.06  p95  463.84
    sim_ms       median  121.71  mean  115.94  p95  278.26
    non-sim_ms   median   87.39  mean   88.12  (policy + proposer + bookkeeping)
    clone_ms     median    0.69  mean    0.87
    sim_steps    median   877.0  mean  1157.0
    ms/sim_step  median   0.116
    ms/clone     median   0.018
```

```
  [mid]  n_turns = 188
    agent_ms     median  169.75  mean  146.75  p95  333.32
    sim_ms       median   33.97  mean   68.79  p95  161.57
    non-sim_ms   median  113.69  mean   77.96  (policy + proposer + bookkeeping)
    clone_ms     median    0.87  mean    1.16
    sim_steps    median   364.5  mean  1399.8
    ms/sim_step  median   0.050
    ms/clone     median   0.017
```

```
  [late] empty
```

## Headroom analysis

```
  Per-turn medians:  agent 230.2 ms  =  sim 117.1 ms  +  non-sim 88.0 ms
  Median sim-step count per turn:  860
  Mean ms per sim step (in production traffic):  0.079 ms
  Remaining of 950 ms budget after current chooser:  720 ms

  Headroom counterfactuals (median turn):
    if sim were 2x cheaper:   turn = 147 ms  →  +59 ms free
    if sim were 10x cheaper:  turn = 100 ms  →  +105 ms free
    if non-sim (policy) were 2x cheaper:  turn = 161 ms  →  +44 ms free
    if non-sim (policy) were 10x cheaper: turn = 126 ms  →  +79 ms free

  At current cost 0.079 ms/sim_step, 950 ms budget = 11957 sim steps theoretical max.
  Today we use 860 (=68 ms) — leaving headroom only after non-sim costs settle.
```
