# 2026-05-22 — session plan

Branch: `claude/review-skills-improvements-moKOR`
Goal: find a significant lift in an A/B test vs current best
(consolidated, μ ≈ 1124).

## Live ladder (15:13 UTC)

| Sub ID | Agent | μ (climbing) | Position |
|---|---|---:|---|
| 52894340 | _phase4_step1_FND (sibling) | 940.8 | Rolling pair (recent) |
| 52893236 | baseline_full (this branch) | 1079.2 | Rolling pair (older) |
| 52882014 | consolidated (this branch) | 1124.0 | EVICTED — best on this branch |

baseline_full regressed ~45 μ from consolidated (still settling).
Both new submissions are climbing; final settlement may shrink the
gap to ~25 μ.

## Test-infra bug found

In-process A/B test env-var pollution (documented in
`knowledge-base/thoughts/2026-05-22-test-infra-env-var-leak.md`).
Variant-vs-variant tests sharing env-gated constants are invalid.
Workaround: hardcoded variant bundles.

## Test campaign

### Test 1 (KILLED): sniper-only vs consolidated
Killed at 13+ min wallclock — slow sniper variant + n=16 took too
long for diagnostic value (live ladder already confirmed
baseline_full regression).

### Test 2 (RUNNING): variant_topk8_locked vs variant_topk5_locked
Hardcoded constants, n=16 (8 seeds × 2 seats), 4 workers.
- topk5: JOINT_TOP_K_PER_TARGET=5, JOINT_MAX_PAIRS=60 (consolidated)
- topk8: JOINT_TOP_K_PER_TARGET=8, JOINT_MAX_PAIRS=100

Hypothesis: PI 2026-05-21 image showed 319-ship planet sitting idle
adjacent to combat. Lifting JOINT_TOP_K should let it join the
attack via JOINT pair enumeration.

Risk: 4P joint over-commitment (comment at
chooser_trajectory.py:880-884). reinforce post-pass mitigates.

### Test 3 (READY): variant_milp_on vs variant_milp_off
- MILP opening planner (cherry-picked from analytical track,
  commit 101729c) — multi-turn opening optimization for step < 30.
- Hardcoded OPENING_MILP_ENABLED=True/False.

### Test 4 (READY): variant_wallclock800 vs variant_wallclock600
- Per-turn budget bump: 600ms → 800ms.
- Kaggle's actTimeout is 1000ms; 200ms headroom for env overhead.
- Hardcoded WALLCLOCK_BUDGET_MS=800/600.

## Decision tree

- Test 2 LIFTS (WLo ≥ 0.55) → escalate to n=32 confirmation;
  document as deliverable.
- Test 2 PARITY → run Test 3 (MILP).
- Test 2 REGRESSES → JOINT axis closed; run Test 3 (MILP), then Test 4
  (wallclock).
