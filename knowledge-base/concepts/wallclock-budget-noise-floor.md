# Wallclock-budget noise floor

**Concept:** Time-budgeted agents introduce non-determinism that
dominates n=32 local A/B variance.

## Mechanism

The v15/v20/v23 family chooser has an adaptive validate cap derived
from wallclock budget:

```python
wallclock_ms = _effective_wallclock_ms()
budget_for_validate = wallclock_ms - _RESERVED_OVERHEAD_MS
n_affordable = max(8, int(budget_for_validate / per_cand_ms))
```

`per_step_ms` is measured by `time.perf_counter()` on a single
fast_sim probe. Under different CPU contention (background process
load, worker pool scheduling), `per_step_ms` varies, so
`n_affordable` varies, so the chooser validates different candidate
sets. Different candidates → different actions → diverging game
trajectories.

## Empirical evidence (2026-05-17)

Determinism test: same seed 0, same agents, two sequential runs.
```
RUN 1: 281 steps, p1_win
RUN 2: 262 steps, p1_win
```
19-step divergence on identical inputs.

n=32 A/B swings on identical seed sets:
```
v26 vs v15 n=32 RUN 1 (fast.py, lone): 21/32 = 65.6%
v26 vs v15 n=32 RUN 2 (fast.py, n=64 first tier): 14/32 = 43.8%
```
Both use seeds 0..15 with balanced seats. ~22pp swing from
wallclock variance alone.

Pooled across both runs: 35/64 = 54.7%.

## Implications for past results

Any local A/B at n≤32 is dominated by this noise. Specifically:
- v20 vs v15 65.6% local → live -21μ regression is consistent with
  wallclock noise (and local-overpredict)
- "Falsification" results for v17/v18/v19/v24/v25 may all be noise-
  dominated; can't trust any of them in isolation
- The Rule 37 "axis closure" claim earlier today was premature given
  the noise floor

## Mitigations

1. **n=64+ for any decision-grade A/B.** n=32 has ±9% Wilson width
   PLUS this wallclock variance — too noisy to trust.
2. **Pool across re-runs.** If the same A/B is run twice, the pooled
   2n result is the right number.
3. **Fixed-validate-count variant.** Replace wallclock-derived cap
   with a constant (say 64 candidates). Changes the agent's behavior
   but removes noise. Useful for clean A/B; can decouple from the
   shipping agent.
4. **CPU-pinned timing probe.** Use `cpu_time()` instead of wall
   time for `per_step_ms` measurement. Makes the cap CPU-load-
   independent. Cleanest fix; doesn't change agent semantics in
   the typical case.

## Submission-decision impact

A submission decision based on a single n=32 A/B is gambling on a
±10pp noise floor. Either submit only changes with strong absolute
gains (like the v23 bug fix, which has independent Rule 38
verification), or require n=64+ pooled before pushing.

## Origin

2026-05-17 v26 surprise-positive investigation: 65.6% h2h vs v15
on the first n=32, dropped to 43.8% on the second n=32 with the
same nominal seed set 0..15.
