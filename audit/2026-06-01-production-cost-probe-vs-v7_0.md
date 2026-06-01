# production-cost probe — 2026-06-01

> Source: `scripts/production_cost_probe.py` (seed=42, episodeSteps=500, baseline-vs-baseline).
> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch `lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` to catch already-imported references.

## Per-turn cost (all seats merged)

```
  [ALL]  n_turns = 1084
    agent_ms     median  285.56  mean  342.96  p95  700.80
    sim_ms       median  141.40  mean  170.21  p95  361.72
    non-sim_ms   median  138.37  mean  172.75  (policy + proposer + bookkeeping)
    clone_ms     median    1.42  mean    1.41
    sim_steps    median  1801.0  mean  1700.6
    ms/sim_step  median   0.095
    ms/clone     median   0.020
```

```
  [early]  n_turns = 830
    agent_ms     median  276.09  mean  328.18  p95  691.99
    sim_ms       median  139.24  mean  165.97  p95  369.99
    non-sim_ms   median  134.39  mean  162.21  (policy + proposer + bookkeeping)
    clone_ms     median    1.20  mean    1.19
    sim_steps    median  1531.0  mean  1428.1
    ms/sim_step  median   0.102
    ms/clone     median   0.020
```

```
  [mid]  n_turns = 254
    agent_ms     median  293.43  mean  391.27  p95  729.76
    sim_ms       median  152.55  mean  184.07  p95  336.46
    non-sim_ms   median  156.15  mean  207.20  (policy + proposer + bookkeeping)
    clone_ms     median    1.98  mean    2.14
    sim_steps    median  2789.5  mean  2591.1
    ms/sim_step  median   0.069
    ms/clone     median   0.020
```

```
  [late] empty
```

## Headroom analysis

```
  Per-turn medians:  agent 285.6 ms  =  sim 141.4 ms  +  non-sim 138.4 ms
  Median sim-step count per turn:  1801
  Mean ms per sim step (in production traffic):  0.100 ms
  Remaining of 950 ms budget after current chooser:  664 ms

  Headroom counterfactuals (median turn):
    if sim were 2x cheaper:   turn = 209 ms  →  +71 ms free
    if sim were 10x cheaper:  turn = 153 ms  →  +127 ms free
    if non-sim (policy) were 2x cheaper:  turn = 211 ms  →  +69 ms free
    if non-sim (policy) were 10x cheaper: turn = 155 ms  →  +125 ms free

  At current cost 0.100 ms/sim_step, 950 ms budget = 9491 sim steps theoretical max.
  Today we use 1801 (=180 ms) — leaving headroom only after non-sim costs settle.
```
