# 2026-05-19 — Analytics Verification (trajectory_roi v3.1)

> Phase A blocker for the goal-directed `trajectory_portfolio`
> planner. See `/root/.claude/plans/optimized-questing-shell.md`.

## Summary

| Test | Result | Notes |
|---|---|---|
| Test 1 — Projection vs Reality | PASS | 20/20 cases bit-exact |
| Test 2 — Projection Determinism | PASS | 10/10 cases identical |
| Test 5 — Capture Math Units (pytest) | see pytest output |  | 3 deterministic units; run with `pytest tests/test_analytics.py` |

## Test 1 — Projection vs Reality

**Result:** PASS — 20/20 cases bit-exact

Per-case detail:

- ✓ episode-76990778-replay.json@t25 seat0 passive: pred=-47.0000 actual=-47.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76990778-replay.json@t25 seat0 agent: pred=-19.0000 actual=-19.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76990778-replay.json@t25 seat1 passive: pred=62.0000 actual=62.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76990778-replay.json@t25 seat1 agent: pred=132.0000 actual=132.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76991765-replay.json@t25 seat0 passive: pred=-51.0000 actual=-51.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76991765-replay.json@t25 seat0 agent: pred=-51.0000 actual=-51.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76991765-replay.json@t25 seat1 passive: pred=216.0000 actual=216.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76991765-replay.json@t25 seat1 agent: pred=350.0000 actual=350.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992190-replay.json@t25 seat0 passive: pred=26.0000 actual=26.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992190-replay.json@t25 seat0 agent: pred=47.0000 actual=47.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992190-replay.json@t25 seat1 passive: pred=170.0000 actual=170.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992190-replay.json@t25 seat1 agent: pred=179.0000 actual=179.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992572-replay.json@t25 seat0 passive: pred=410.0000 actual=410.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992572-replay.json@t25 seat0 agent: pred=547.0000 actual=547.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992572-replay.json@t25 seat1 passive: pred=-281.0000 actual=-281.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992572-replay.json@t25 seat1 agent: pred=179.0000 actual=179.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992967-replay.json@t25 seat0 passive: pred=70.0000 actual=70.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992967-replay.json@t25 seat0 agent: pred=145.0000 actual=145.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992967-replay.json@t25 seat1 passive: pred=40.0000 actual=40.0000 delta=+0.0000e+00 (rel=0.00%)
- ✓ episode-76992967-replay.json@t25 seat1 agent: pred=136.0000 actual=136.0000 delta=+0.0000e+00 (rel=0.00%)

## Test 2 — Projection Determinism

**Result:** PASS — 10/10 cases identical

Per-case detail:

- ✓ episode-76990778-replay.json@t25 seat0: v1=-47.000000 v2=-47.000000 identical=True
- ✓ episode-76990778-replay.json@t25 seat1: v1=62.000000 v2=62.000000 identical=True
- ✓ episode-76991765-replay.json@t25 seat0: v1=-51.000000 v2=-51.000000 identical=True
- ✓ episode-76991765-replay.json@t25 seat1: v1=216.000000 v2=216.000000 identical=True
- ✓ episode-76992190-replay.json@t25 seat0: v1=26.000000 v2=26.000000 identical=True
- ✓ episode-76992190-replay.json@t25 seat1: v1=170.000000 v2=170.000000 identical=True
- ✓ episode-76992572-replay.json@t25 seat0: v1=410.000000 v2=410.000000 identical=True
- ✓ episode-76992572-replay.json@t25 seat1: v1=-281.000000 v2=-281.000000 identical=True
- ✓ episode-76992967-replay.json@t25 seat0: v1=70.000000 v2=70.000000 identical=True
- ✓ episode-76992967-replay.json@t25 seat1: v1=40.000000 v2=40.000000 identical=True

## Phase B Gating

- [x] Tests 1-4 all PASS → unblocked pending Test 5 pytest.

## Bugs Found

None — all checks PASS.
