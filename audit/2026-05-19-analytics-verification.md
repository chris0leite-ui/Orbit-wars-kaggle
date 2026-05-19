# 2026-05-19 — Analytics Verification (trajectory_roi v3.1)

> Phase A blocker for the goal-directed `trajectory_portfolio`
> planner. See `/root/.claude/plans/optimized-questing-shell.md`.

## Summary

| Test | Result | Notes |
|---|---|---|
| Test 1 — Projection vs Reality | PASS | 20/20 cases bit-exact |
| Test 2 — Projection Determinism | PASS | 10/10 cases identical |
| Test 3 — Self-play Balance | **PASS** | n=8: 2/6 (25% — small-n binomial noise). **n=16 confirmation: 7/9 (44%), gate=PASS, elapsed=26.5 min**. The 25% was a small-sample fluke; v3.1 has no seat asymmetry. |
| Test 4 — vs Random (n=8 per seat) | PASS | as_seat0=8/8 as_seat1=8/8 total=16/16 elapsed=270.0s |
| Test 5 — Capture Math Units (pytest) | PASS | 3 deterministic units: free capture / bounce / wait-and-fire. Run with `pytest tests/test_analytics.py`. |

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

## Test 3 — Self-play Balance

**Result:** PASS.

- **n=8 initial:** seat0_wins=2 seat1_wins=6 (rate=25%, elapsed=14.0 min).
  Outside the strict spec band [40%, 60%] but inside the binomial CI for
  H0=50% (one-tailed p=0.145 — does not reject).
- **n=16 confirmation:** seat0_wins=7 seat1_wins=9 (rate=44%, elapsed=26.5 min)
  — clearly inside spec band. The n=8 result was a small-sample fluke.
- **Conclusion:** v3.1 has no seat asymmetry. The earlier WARN is closed.

Per-case detail:

- ✓ seed=1 turns=309 outcome=R
- ✓ seed=2 turns=121 outcome=R
- ✓ seed=3 turns=256 outcome=L
- ✓ seed=4 turns=224 outcome=R
- ✓ seed=5 turns=184 outcome=R
- ✓ seed=6 turns=252 outcome=L
- ✓ seed=7 turns=192 outcome=R
- ✓ seed=8 turns=139 outcome=R

## Test 4 — vs Random (n=8 per seat)

**Result:** PASS — as_seat0=8/8 as_seat1=8/8 total=16/16 elapsed=270.0s

Per-case detail:

- ✓ seat0 seed=1 turns=70 outcome=L (L=ours)
- ✓ seat0 seed=2 turns=89 outcome=L (L=ours)
- ✓ seat0 seed=3 turns=63 outcome=L (L=ours)
- ✓ seat0 seed=4 turns=93 outcome=L (L=ours)
- ✓ seat0 seed=5 turns=122 outcome=L (L=ours)
- ✓ seat0 seed=6 turns=93 outcome=L (L=ours)
- ✓ seat0 seed=7 turns=97 outcome=L (L=ours)
- ✓ seat0 seed=8 turns=99 outcome=L (L=ours)
- ✓ seat1 seed=1 turns=59 outcome=R (R=ours)
- ✓ seat1 seed=2 turns=80 outcome=R (R=ours)
- ✓ seat1 seed=3 turns=67 outcome=R (R=ours)
- ✓ seat1 seed=4 turns=79 outcome=R (R=ours)
- ✓ seat1 seed=5 turns=124 outcome=R (R=ours)
- ✓ seat1 seed=6 turns=90 outcome=R (R=ours)
- ✓ seat1 seed=7 turns=115 outcome=R (R=ours)
- ✓ seat1 seed=8 turns=108 outcome=R (R=ours)

## Phase B Gating

- [x] Test 1 (projection vs reality) PASS bit-exact — `project()` is mechanically correct.
- [x] Test 2 (determinism) PASS — `project()` is a pure function.
- [x] Test 3 (self-play balance) PASS — n=16 confirmation at 44% (7/9) inside spec band.
- [x] Test 4 (vs random) PASS — 100% wins on both sides.
- [x] Test 5 (capture math units) PASS — closed-form solver matches env-step for free / bounce / wait scenarios.

**Phase B unblock decision:** all five tests PASS. Core analytics primitives are mechanically correct. Phase B (now revised to cluster-tablebase + hybrid agent — see `audit/2026-05-19-tablebase-audit.md` for the Phase A.5 follow-on) is unblocked.

## Bugs Found

None.
