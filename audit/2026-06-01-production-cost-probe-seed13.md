# production-cost probe — 2026-06-01

> Source: `scripts/production_cost_probe.py` (seed=13, episodeSteps=500, baseline-vs-baseline).
> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch `lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` to catch already-imported references.

## Per-turn cost (all seats merged)

```
  [ALL]  n_turns = 206
    agent_ms     median  283.32  mean  362.31  p95  708.54
    sim_ms       median  199.55  mean  219.47  p95  409.97
    non-sim_ms   median  101.43  mean  142.84  (policy + proposer + bookkeeping)
    clone_ms     median    0.72  mean    1.18
    sim_steps    median   799.0  mean  1279.3
    ms/sim_step  median   0.158
    ms/clone     median   0.022
```

```
  [early]  n_turns = 206
    agent_ms     median  283.32  mean  362.31  p95  708.54
    sim_ms       median  199.55  mean  219.47  p95  409.97
    non-sim_ms   median  101.43  mean  142.84  (policy + proposer + bookkeeping)
    clone_ms     median    0.72  mean    1.18
    sim_steps    median   799.0  mean  1279.3
    ms/sim_step  median   0.158
    ms/clone     median   0.022
```

```
  [mid] empty
```

```
  [late] empty
```

## Headroom analysis

```
  Per-turn medians:  agent 283.3 ms  =  sim 199.5 ms  +  non-sim 101.4 ms
  Median sim-step count per turn:  799
  Mean ms per sim step (in production traffic):  0.172 ms
  Remaining of 950 ms budget after current chooser:  667 ms

  Headroom counterfactuals (median turn):
    if sim were 2x cheaper:   turn = 201 ms  →  +100 ms free
    if sim were 10x cheaper:  turn = 121 ms  →  +180 ms free
    if non-sim (policy) were 2x cheaper:  turn = 250 ms  →  +51 ms free
    if non-sim (policy) were 10x cheaper: turn = 210 ms  →  +91 ms free

  At current cost 0.172 ms/sim_step, 950 ms budget = 5537 sim steps theoretical max.
  Today we use 799 (=137 ms) — leaving headroom only after non-sim costs settle.
```
