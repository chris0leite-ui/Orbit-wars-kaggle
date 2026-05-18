# Phase C bench: bundle (lite vs off) vs v7_0

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Trigger**: Phase B me-followup shipped; Phase C plan called for A/B vs v7_0 + agents/baseline with `BUNDLE_ME_FOLLOWUP=lite`. PI asked to bench one simulated game and profile before kicking off the full 64-game A/B.

## TL;DR

Bundle loses 0-20 to v7_0 on seed=42 **regardless of mode**. me-followup
doesn't change the strategic outcome — it just adds unsafe latency
(4 turns > 1000ms vs 0 in off mode). **Do not submit lite mode.**
The bundle-vs-v7_0 gap is a strategic problem, not a self-consistency
problem.

## Three runs (all seed=42, bundle as P0, v7_0 as P1)

| Run | Result | Bundle p50 / p95 / max | Bundle turns > 1000ms | Bundle elapsed |
|---|---|---|---|---|
| Lite vs random (2 games, prior bench) | Bundle wins both | 800 / 868 / 922 ms | 0/998 (but 493 > 800ms) | ~390 s/game |
| Lite vs v7_0 | Bundle **eliminated turn 118** (0-20) | 794 / 977 / 1078 ms | **4/118** | 116.5 s |
| Off vs v7_0 | Bundle **eliminated turn 121** (0-20) | 259 / 435 / 491 ms | 0/121 | 66.9 s |

## What the profile showed

`audit/2026-05-18-bundle-lite-profile.prof` (lite mode):
- Total wallclock 116 s
- `predict_my_followup_via_event_driven_lite_greedy`: 811 calls, **62.3 s = 54% of total runtime**
- Per-call: ~77 ms (dominated by ~10 `snapshot_at` invocations per event-driven walk)
- BundleSearch only managed **~7 score calls per turn** (vs ~15 expected) — deadline-bailing early

`audit/2026-05-18-bundle-off-profile.prof` (off mode):
- Total wallclock 66.6 s
- BundleSearch did **~24 score calls per turn** (13786 / 121) — 3.4x more exploration
- No me-followup function in the hotspot list (guarded by `my_followup_mode == "off"`)

The cost shift is unambiguous: lite mode trades ~60% of search budget
for per-score followup application. The search is *less* informed
strategically as a result.

## Strategic observation (n=1, low confidence)

Both off and lite runs followed the same opening through turn 20:
bundle captured 3-4 close neutrals, then v7_0 captured all the
distant neutrals + neighboring frontier. Bundle never recovered.

At turn 84 (off run) and turn 88 (lite run) bundle was already down
to 2-3 planets vs 17-18 for v7_0. v7_0 hit ~700ms several times mid-
game — it's also a heavy search — but its capture velocity was higher.
This game's loss is **not** about me-followup; it's about bundle's
opening play (or its evaluator's weighting) being weaker than v7_0's
opening play.

This is n=1 with a single seed. A wider A/B (n=8 or n=32) would
quantify the structural delta.

## Decision impact

1. **Do not submit with `BUNDLE_ME_FOLLOWUP=lite`.** 4/118 turns
   over actTimeout=1000 ms = game-drop risk in live env. Even if
   the strategic case were positive, the timing alone disqualifies.
2. **Phase C A/B `BUNDLE_ME_FOLLOWUP=lite` is likely a null.**
   Spending compute on the full n=32 vs v7_0 run with lite mode
   would just measure the same loss pattern twice. Better: run a
   small (n=8) A/B with off mode vs v7_0 to calibrate the
   structural gap, then decide whether to submit off mode or
   pivot.
3. **The "isolate-fix-verify" principle worked.** Oracle A5
   correctly verified the me-followup mechanism. Production
   correctly reported "but the mechanism alone doesn't win games."
   Both findings are valid; neither obviates the other. The
   oracle wasn't lying — it just tested mechanism, not outcome.

## Recommended next steps (PI to ratify)

- **Cheap diagnostic**: n=8 quick A/B bundle (off) vs v7_0. ~20 min
  wallclock. Gives the actual winrate, not just one-seed evidence.
- **Cheap diagnostic**: n=8 bundle (off) vs agents/baseline. Same
  cost, calibrates against the OTHER target.
- **If off mode A/B is positive (Wlo > 0.40)**: submit off mode.
  Skip lite entirely until cost can be amortized (cache, selective
  application, reduce max_events).
- **If off mode A/B is null/negative**: bundle has a deeper gap vs
  v7_0. Rule 37 (axis exhaustion) territory — consider value-head
  (1.A) or IL warm-start (1.C) instead of further chooser tuning.

## Cost optimization ideas (deferred, do not action without PI)

If lite mode's strategic case eventually justifies the cost:
- Reduce `max_events=10` → 5 or 3 (halves per-call cost)
- Cache `snapshot_at(t)` across the event walk (largest single hot spot)
- Apply me-followup ONCE per BundleSearch call (at root) instead of
  per-score (loses per-candidate granularity but ~15x cheaper)
- Only run me-followup when the candidate bundle's first launch has
  arrival ETA within `horizon * 0.6` (so the followup window is
  non-degenerate; pre-empts the "no events within horizon" case
  that A5 originally hit)

## Friction logged

`oracle-passes-production-loses-pattern` — Oracle suite green is
necessary but not sufficient. A mechanism can pass its synthetic
isolation test, the production cost-benefit can still be net
negative. Always pair oracle work with at least one live-style game
profile before committing to a Phase C A/B.

## Artifacts

- `audit/2026-05-18-bundle-lite-profile.prof`
- `audit/2026-05-18-bundle-off-profile.prof`
- `scripts/profile_bundle_vs_v7_0.py`
