# production-cost probe — 2026-06-01

> Source: `scripts/production_cost_probe.py` (seed=99, episodeSteps=500, baseline-vs-baseline).
> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch `lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` to catch already-imported references.

## Per-turn cost (all seats merged)

```
  [ALL]  n_turns = 194
    agent_ms     median  121.60  mean  134.98  p95  289.74
    sim_ms       median   91.06  mean   92.26  p95  192.01
    non-sim_ms   median   26.35  mean   42.73  (policy + proposer + bookkeeping)
    clone_ms     median    0.30  mean    0.55
    sim_steps    median   399.5  mean   724.9
    ms/sim_step  median   0.163
    ms/clone     median   0.019
```

```
  [early]  n_turns = 194
    agent_ms     median  121.60  mean  134.98  p95  289.74
    sim_ms       median   91.06  mean   92.26  p95  192.01
    non-sim_ms   median   26.35  mean   42.73  (policy + proposer + bookkeeping)
    clone_ms     median    0.30  mean    0.55
    sim_steps    median   399.5  mean   724.9
    ms/sim_step  median   0.163
    ms/clone     median   0.019
```

```
  [mid] empty
```

```
  [late] empty
```

## Headroom analysis

```
  Per-turn medians:  agent 121.6 ms  =  sim 91.1 ms  +  non-sim 26.4 ms
  Median sim-step count per turn:  400
  Mean ms per sim step (in production traffic):  0.127 ms
  Remaining of 950 ms budget after current chooser:  828 ms

  Headroom counterfactuals (median turn):
    if sim were 2x cheaper:   turn = 72 ms  →  +46 ms free
    if sim were 10x cheaper:  turn = 35 ms  →  +82 ms free
    if non-sim (policy) were 2x cheaper:  turn = 104 ms  →  +13 ms free
    if non-sim (policy) were 10x cheaper: turn = 94 ms  →  +24 ms free

  At current cost 0.127 ms/sim_step, 950 ms budget = 7464 sim steps theoretical max.
  Today we use 400 (=51 ms) — leaving headroom only after non-sim costs settle.
```
