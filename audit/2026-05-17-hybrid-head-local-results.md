# 2026-05-17 — hybrid head local results (composite-2P + A2-4P)

> Branch: `claude/kaggle-baseline-strategy-lO4mm` (this branch).
> Commit: `a97806a` (favor_hybrid wired).
> Recorded for cross-branch comparison with
> `claude/audit-workflow-performance-btjeK` and any other worker
> investigating the composite head.

## What was tested

`agents/baseline/main.py` with `BASELINE_VALUE_HEAD=hybrid` env var.
`favor_hybrid` dispatches by `num_seats`:

- 2P (`num_seats <= 2`) → `favor_composite` (calls
  `lib.value_heads.composite_capture_value`).
- 4P (`num_seats > 2`) → `favor` (with A2 4P-weakness multiplier +
  elimination bonus).

Composite and A2 are domain-disjoint: composite has no 4P opp
aggregation (`composite-value-head-2p-only.md` flag); A2's per-WEAKEST
multiplier + elim bonus only fire when `num_seats > 2`.

Background — composite head A/B from the other branch (per nutshell
relayed by PI):

- Composite vs v9_scavenge (team peak μ=1119.9): **93.8 %, Wlo=0.799**
  — decisive.
- Composite vs v15 (rolling champion μ=1112.8): **67.2 %, Wlo=0.550**
  — point estimate clearly winning, right on the gate at n=64.

A2 4P signal from our prior FFA runs:

- n=16 seeds (64 games): baseline 71.9 %, v15 62.5 % → +9.4 pp.
- n=32 seeds (128 games): baseline 66.4 %, v15 64.1 % → +2.3 pp.
- Pooled (192 games): baseline 68.2 %, v15 63.5 % → +4.7 pp.
- Within the ~10 pp wallclock-noise floor; directional but
  not statistically significant on its own.

## Local validation results (this branch, this session)

### Unit tests — `tests/test_baseline_value.py`

```
17 passed, 3 warnings in 3.88s
```

Includes 3 new tests for the hybrid dispatch:
- `test_select_favor_fn_hybrid_path` — `BASELINE_VALUE_HEAD=hybrid`
  swaps to `favor_hybrid`.
- `test_favor_hybrid_dispatches_2p_to_composite` — 2P call to hybrid
  matches `favor_composite` exactly.
- `test_favor_hybrid_dispatches_4p_to_favor` — 4P call to hybrid
  matches canonical `favor` (with A2) exactly.

### Bench — `BASELINE_VALUE_HEAD=hybrid python fast.py bench baseline`

```
== bench baseline vs v7_0  budget 1000ms ==
   seed=0   n_steps=221  focal p95= 594ms  max= 662ms  outcome=p0_win
   seed=1   n_steps=201  focal p95= 329ms  max= 421ms  outcome=p1_win
   seed=2   n_steps=298  focal p95= 376ms  max= 439ms  outcome=p0_win

   focal baseline: n=717  p50=213  p95=550  p99=601  max=662ms
   over_1000ms=0
   total wallclock 245.9s
   verdict: PASS  (gate: p95<800ms AND zero >=1000ms)
```

Hybrid is ~80 % slower at p95 than the A2-only default
(290ms → 550ms, consistent with PR #29's composite smoke). Within
the 800ms gate; well clear of the 1000ms actTimeout.

### Smoke — `BASELINE_VALUE_HEAD=hybrid python fast.py smoke baseline`

```
== smoke baseline (/home/user/Orbit-wars-kaggle/agents/baseline/main.py) ==
   16 seeds × 2 seats × 2 opp = 64 games, 8 workers

   opponent           wins      %    Wlo    Whi   p95ms  seconds
   random           32/32   100.0   0.89   1.00     866    250.9
   nearest          32/32   100.0   0.89   1.00     897    367.2

   verdict: PASS  (clears both smoke floors with Wlo≥0.55)
```

100 % vs both random and nearest. Per-turn p95 is higher than bench
(866ms / 897ms vs 550ms) — this is CPU-contention noise from 8 parallel
worker processes competing on the same machine. **No turn exceeded
1000ms.** Kaggle's evaluator runs agents alone (no contention), so
real-world p95 should track the bench number (~550ms).

## What has NOT been run on this branch yet

- **h2h vs v15 (n=64) under hybrid.** This would directly verify the
  other branch's 67.2 % composite-vs-v15 result reproduces on this
  branch's merged code path. ~25-45 min wallclock; deferred pending
  PI choice.
- **4P FFA panel with hybrid.** Would verify A2's 4P side of the
  hybrid still helps (or at least doesn't regress) when composite is
  active in 2P. ~30-50 min wallclock; deferred.
- **Bundle + submission.** Single-shot per Rule 1; requires PI
  approval. Not done.

## Files / commits this session

- `agents/baseline/value.py` — added `favor_hybrid` + `select_favor_fn`
  "hybrid" branch (commit `a97806a`).
- `tests/test_baseline_value.py` — added 3 dispatch + behavior tests
  (commit `a97806a`).
- `audit/2026-05-17-hybrid-head-local-results.md` — this file.

## How to reproduce on a peer branch / worker

```bash
git fetch origin claude/kaggle-baseline-strategy-lO4mm
git checkout claude/kaggle-baseline-strategy-lO4mm
git rev-parse HEAD       # expect a97806a or later

# unit
python -m pytest tests/test_baseline_value.py -q

# bench
BASELINE_VALUE_HEAD=hybrid python fast.py bench baseline

# smoke
BASELINE_VALUE_HEAD=hybrid python fast.py smoke baseline

# h2h vs v15 (the gate that would clear submission, deferred here)
git show f315dc7:agents/v15/main.py > /tmp/v15_src.py
mkdir -p agents/_v15_h2h && cp /tmp/v15_src.py agents/_v15_h2h/main.py
python scripts/bundle_agent.py agents/_v15_h2h/main.py
mv submissions/main.py /tmp/v15_bundle.py
rm -rf agents/_v15_h2h
BASELINE_VALUE_HEAD=hybrid python fast.py eval baseline \
  --vs /tmp/v15_bundle.py --max-seeds 32 --gate 0.50
```

Expected (cross-branch) result for the h2h: **66-70 % Wlo > 0.55** if
the other branch's 67.2 % reproduces. Anything materially below 60 %
would indicate the hybrid wire-up or the merge introduced a regression
to verify.
