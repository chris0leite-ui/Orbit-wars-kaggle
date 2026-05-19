# Layered chooser validation — 2026-05-19

> Validation of Layer-0 closed-form predicates (W1/W2/L1/L2) +
> swap-able `chooser_layered.py`. Commit `671edb1` on branch
> `claude/strategy-framework-design-OyoYR-rebased`.
>
> Plan reference: `/root/.claude/plans/take-the-lens-of-magical-shore.md`
> §10. No source code changes during this validation — only bundle
> generation and `fast.py` runs.

## Bundles under test

- **Control**: `submissions/baseline.py` (built `2026-05-19` from
  `agents/baseline/` HEAD `671edb1`; sha256 `bcf0b238a03cbe04`).
  Default `BASELINE_CHOOSER=trajectory`.
- **Treatment**: `submissions/_ab/layered_w1w2l1l2.py` (one-line `sed`
  override of the control: `setdefault("BASELINE_CHOOSER", "trajectory")`
  → `"layered"`). Default `BASELINE_INNER_CHOOSER=trajectory`.

Diff is exactly one line — verified via `diff`:
```
11553c11553
< os.environ.setdefault("BASELINE_CHOOSER", "trajectory")
---
> os.environ.setdefault("BASELINE_CHOOSER", "layered")
```

## Step 1 — Bench parity (5 games each vs random)

**Command:**
```bash
python fast.py bench submissions/baseline.py --vs random --games 5
python fast.py bench submissions/_ab/layered_w1w2l1l2.py --vs random --games 5
```

**Control (trajectory)**:
- n=837 turns, p50=252, p95=442, p99=565, max=924, over_1000ms=0
- verdict: **PASS** (gate: p95<800ms AND zero ≥1000ms)

**Treatment (layered)**:
- n=525 turns, p50=26, p95=438, p99=643, max=679, over_1000ms=0
- verdict: **PASS**

**Verdict**: Wallclock parity confirmed. Layered's p50 is ~10× lower
than trajectory's (26 ms vs 252 ms) — Layer 0 short-circuits easy
turns. p95 essentially identical (438 vs 442 ms). Max is markedly
lower under layered (679 vs 924 ms) — fewer rollouts on the worst
turns. **No wallclock regression; gentle improvement.**

## Step 2 — Head-to-head: layered vs trajectory, n=32

**Command:**
```bash
python fast.py eval submissions/_ab/layered_w1w2l1l2.py \
    --vs submissions/baseline.py --max-seeds 32 --workers 4
```

**Output**: (in progress — pending)

**Verdict**: TBD

## Step 3 — Panel calibration + h2h, n=32

**Command:**
```bash
python fast.py eval submissions/_ab/layered_w1w2l1l2.py \
    --vs-panel default --require-h2h submissions/baseline.py \
    --max-seeds 32 --workers 4
```

**Output**: (pending Step 2 verdict)

**Verdict**: TBD

## Step 4 — Predicate-fire diagnostic

(pending Step 2/3 verdicts)

**W1 fires/turn**: TBD
**W2 fires/turn**: TBD
**L1 discards/turn**: TBD
**L2 prunes/turn**: TBD

## Decision

(pending all step verdicts)

## Notes

- The current rolling-pair floor is μ=1118.8 (HANDOVER 2026-05-19;
  subject to ladder drift). Submission decision waits on all 3 gates
  passing.
- This audit doc is the source of truth for the validation; no
  shortcuts.
