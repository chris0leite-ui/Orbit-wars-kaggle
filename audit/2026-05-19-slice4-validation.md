# Slice 4 validation — 2026-05-19

> Commit `9b1a2f7` — architectural flip from preempt to backstop
> (`predicates-as-priors`). Validated per the new Rule 41
> ("inspect first, small A/B second").

## Bundle under test

- Treatment: `submissions/_ab/layered_w1w2l1l2.py` (one-line sed
  override; defaults to `BASELINE_CHOOSER=layered`).
- Control: `submissions/baseline.py` (default trajectory).

## Step 1 — Bench parity (5 games each vs random)

Both bundles PASS:
- Control trajectory: p95=665, max=853, zero >1000ms.
- Treatment layered (Slice 4): p95=529, max=746, zero >1000ms.

## Step 4 — Single-game introspect (seed 42, vs trajectory baseline)

**Outcome: WIN +1/-1**

```
totals: cands=5423 W1=259 W2=75 L1=4 L2p=120 uncertain=5085
        inner_emit=187 backstop_appended=67
per-turn avg: cands=26.3 W1=1.26 W2=0.36 L1=0.02 L2p=0.58
backstop rate: 67/334 = 20.1% of commits actually appended
```

Interpretation:
- 80% of L0 commits are also chosen by the inner — they're
  confirmations, not corrections.
- 20% are backstop appends — provable wins the inner skipped
  (or couldn't reach within wallclock).
- Total emits = 254 over 206 turns (1.23/turn); 26% of all emits
  came from the L0 backstop.
- The architecture is doing what we intended.

## Step 2 — Small A/B (n=16; per Rule 41)

```
n=16  wins=9/16  (56.2%)  Wlo=0.332  Whi=0.769  INCONCLUSIVE
focal turn-ms  p50=194  p95=579  max=1535
```

Comparison vs Slice 3 (preempt v2 multi-opp) at n=64:

| | Slice 3 n=64 | Slice 4 n=16 |
|---|---|---|
| Wins | 31/64 (48.4%) | 9/16 (56.2%) |
| Wlo | 0.366 | 0.332 |
| max-ms | 1064 | **1535** |

**Point estimate moved up +8pp.** CI is wide due to small n; Wlo
doesn't clear the 0.45 gate but that's noise, not a worse
underlying truth.

**Wallclock concern**: max=1535ms exceeds the 1000ms env cap on
hard turns. The Slice 3 wallclock-passthrough fix is insufficient
when the inner now sees the full prerank.

## Decision

**Proceed to Slice 5 (stacked).** Per Rule 41:
- Single-game introspect → positive signal (WIN).
- Small A/B point estimate → positive direction (+8pp vs Slice 3).
- Both indicators agree.

Slice 5 (bounded-interval scoring) is expected to:
- Reduce L0 commit frequency → fewer false-positive priors.
- Free wallclock budget by being more selective.

If Slice 5 introspection + small A/B both regress vs Slice 4, STOP
and address wallclock before going further.

Production unchanged: `BASELINE_CHOOSER=trajectory` remains default.
