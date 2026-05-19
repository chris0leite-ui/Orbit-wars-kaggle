# 2026-05-19 — Analytics Verification (trajectory_roi v3.1)

> Phase A blocker for the goal-directed `trajectory_portfolio`
> planner. See `/root/.claude/plans/optimized-questing-shell.md`.

## Summary

| Test | Result | Notes |
|---|---|---|
| Test 1 — Projection vs Reality | PASS | 20/20 cases bit-exact |
| Test 2 — Projection Determinism | PASS | 10/10 cases identical |
| Test 3 — Self-play Balance (n=8) | **WARN** | seat0_wins=2 seat1_wins=6 → 25%/75% — outside spec band [40%, 60%] but inside the WARN band; n=8 binomial CI doesn't reject H0=50% (p=0.145). Needs n=16 confirmation before treating as a real asymmetry. |
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

## Test 3 — Self-play Balance (n=8)

**Result:** WARN — seat0_wins=2 seat1_wins=6 (rate=25%) draws=0 elapsed=840.3s

Spec gate is `seat0_wins / n ∈ [0.4, 0.6]`. At n=8 the strict band is [3.2, 4.8] (i.e. only seat0_wins=4 strictly passes). We got 2 wins → outside spec band.

However: at n=8 the binomial CI for true balance H0=50% is [0.07, 0.61] — the observed 25% does NOT reject H0 (one-tailed p=0.145). This could be small-n noise or a real ~70/30 asymmetry; **n=16 is required to disambiguate** (binomial CDF ≤4/16 under H0=0.5 is 0.038, which would be significant). A confirmation run is queued.

Note also that the analogous live-episode trace (`audit/live-episodes/52784853/episode-76990778-replay.json`, baseline-vs-ladder, NOT trajectory_roi) showed seat 1 also winning despite a roughly mirror-image setup. Some seat asymmetry is likely env-side (home-group offset for seat 1 differs from seat 0 by 3 indices in `interpreter.generate_planets`), not agent-side. The WARN therefore points to "expected geometric drift, not v3.1 bug" until n=16 says otherwise.

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
- [ ] Test 3 (self-play balance) **WARN at n=8** — re-running at n=16 for disambiguation.
- [x] Test 4 (vs random) PASS — 100% wins on both sides.
- [x] Test 5 (capture math units) PASS — closed-form solver matches env-step for free / bounce / wait scenarios.

**Phase B unblock decision:** core analytics primitives are verified mechanically correct. The Test 3 WARN, if it persists at n=16, indicates a behavioural / geometric asymmetry rather than a primitive bug. Phase B (`trajectory_portfolio` planner) can be drafted in parallel with the n=16 confirmation; the build is on hold for n=16's result only if it lands in the FAIL band (rate < 20% or > 80%).

## Bugs Found

None at the primitive level. Test 3 WARN is a behavioural observation, not a primitive bug, and may resolve to PASS at n=16.
