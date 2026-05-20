# Slice 8 validation — 2026-05-20 — speed PASS, win-rate FAIL

> Commit `a3a8dd8` — `chooser_differential.py` (closed-form leaf eval,
> no fast_sim). Per Rule 41: bench, inspect, small A/B.

## Step 1 — Bench parity (5 games vs random)

```
focal differential: n=579 turns
  p50=16 (vs trajectory ~252; 15x faster on median turn)
  p95=78 (vs ~665; 8.5x faster)
  max=202 (vs ~853; 4.2x faster)
  over_1000ms=0
```

**PASS — massive speed improvement.** Won all 5 vs-random games.

## Step 4 — Single-game introspect (seed 0 vs trajectory)

**Outcome: TIE (1, 1)** — both alive at episode end (499 turns).

```
totals: cands=10225 positive-delta=2523 emits=373
per-turn avg: cands=20.5 positive-delta=5.1 emits=0.75
emit/positive ratio: 0.15 (we emit only 15% of viable candidates)
```

### Key finding: under-emit due to wait_N lock

Many turns: top candidate is wait_N>0 (a wait-then-fire plan).
The greedy emit picks it, reserves src+tgt, emits nothing.
Same source then sits idle. Next turn, same wait plan appears
with wait_N decremented, locks source again. Source never fires.

Concrete step 488:
```
top: (Δ=+23.9, 13→21, 20ships, wait_N=3, eta=3)
```
Differential's leaf eval correctly values this plan ("at H,
production has accrued"), but the chooser's emit logic only
fires wait_N=0. So the "decision" never produces a launch.

Many late-game Δ=+0.0 entries: closed-form correctly says
"capture-of-comet-with-no-production-left = no value." Good math.

## Step 2 — Small A/B (n=16, vs trajectory baseline)

```
n=16  wins=6/16  (37.5%)  Wlo=0.185  Whi=0.614  INCONCLUSIVE
focal turn-ms  p50=50  p95=415  max=810
total elapsed 372.9s  (half of trajectory's typical 700s)
```

Per plan §13 decision matrix:
- Wlo=0.185 < 0.30 → **STOP — architecture wrong; document and pivot**.

## Diagnosis

The differential leaf-eval substrate is correct AND fast. The
failure mode is in the chooser's emit logic:

1. Differential's Δ scoring rewards plans that capture-and-hold
   for the longest window (production × time_remaining).
2. wait_N>0 candidates accrue MORE production at the leaf (extra
   wait gives compounded prod).
3. Greedy per-source emit picks the highest-Δ candidate.
4. wait_N>0 candidates win the per-source race → source locked.
5. Lock prevents fire-now from same source → 0 emit this turn.
6. Source sits idle. Wait plan never fires because next turn the
   same pattern repeats with wait_N - 1.

Trajectory chooser doesn't have this failure because its rollout
SIMULATES the wait — at wait_N=K, source's ships accumulate for K
turns then fire. The score reflects "what happens if we wait K
turns and then fire." Differential's static projection assumes
the wait plan executes deterministically, but the chooser's emit
logic doesn't actually execute it.

## Decision

**STOP this slice.** Production unchanged: `BASELINE_CHOOSER=trajectory`
remains default. Differential is opt-in research code via
`BASELINE_CHOOSER=differential`.

The wallclock win (5× faster) is real and reproducible. The
under-emit is a fixable behavioral bug, not a fundamental flaw
in the substrate. A natural next slice (Slice 8c, not in current
plan):

- Drop wait_N>0 candidates from the differential's input set.
- Re-run inspect + small A/B.
- If wins recover, ship differential as the speed-tier alternative.

But per the §13 STOP rule, this is a PI decision, not auto-execute.

## What's preserved

- `agents/baseline/chooser_differential.py` — the closed-form leaf
  eval, opt-in.
- `tests/test_chooser_differential.py` — 11 unit tests pinning the
  math (favor projection, idle baseline, score sign, emit shape).
- `scripts/differential_introspect.py` — single-game inspection
  for further diagnosis.

The differential leaf-eval primitive is reusable for future axes:
- Audit-replay validation (project leaf state vs actual outcome).
- Training-data labelling (closed-form value targets).
- Hybrid scoring (use Δ-favor as a feature in the trajectory chooser).
