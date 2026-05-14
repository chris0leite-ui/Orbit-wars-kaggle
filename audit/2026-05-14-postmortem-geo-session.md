# 2026-05-14 — Postmortem: geo agent session (autonomous iteration)

## What happened

Session goal evolved through user direction:

1. **Initial**: "fast iteration" infrastructure → built `fast.py` (single-file harness).
2. **Strategic redesign**: "geometric strategy" → built `lib/geo/{sense,posture,allocator}.py` + `agents/geo/main.py`.
3. **Game-theoretic combination**: combine sense tilts with v7_0's K=10 lookahead.
4. **Top-10 archetypes**: blend concentrated + saturation candidate variants.
5. **Wallclock fix**: signal-based per-score timeout (A).
6. **Composite value head**: tried (C); regressed -19pp; reverted.
7. **gang_up + empty_out + tap_capture** batch (v3.2): only gang_up survived.
8. **Submit**: pushed geo v3.1 to ladder.
9. **Ladder result**: μ=984.0 σ-discounted floor (~80-130 episodes); team score
   unchanged at 1064.4 (v7_pv carries us).

29 commits to `claude/simplify-fast-setup-azW8T`. All pushed.

## What worked

- **`fast.py`** validated against ground-truth audit data (bit-identical
  reproduction of v7_1 vs v7_0 = 53.1%). The iteration loop is now ~30s
  for `smoke`, ~10-15 min for `eval` vs v7_0 at n=64.
- **Geometric substrate** (`lib/geo/sense.py`): clustering, Voronoi, front,
  threat, comet claims. 17 unit tests pass. Pure-function, no module-level
  state.
- **Signal-based wallclock timeout (A)**: dropped max from ~2900ms to
  ~1200ms. Robust submission-safety fix without strategic cost.
- **Local A/B vs v7_0**: ~+7pp consistent across n=192 combined.
- **Local A/B 4P vs 3× v7_0**: ~+31pp first-place over baseline at n=128.

## What failed and why

| Attempt | Result | Root cause |
|---|---|---|
| v1 posture multipliers | -37pp | Cross-class score ≥2× crushed settle_plan's ranking |
| v1 greedy-multi allocator | -31pp | Global score-sort over-concentrates at strong sources |
| v1 _aggressive_for(DEFEND)=False | -22pp | Non-aggressive snipe sizing is strictly dominated |
| v2.4 lite_greedy follow-up | -17pp | Cheaper opp model → lookahead picks non-transferring candidates |
| v2.5 WALLCLOCK 500→350 | -20pp | Tighter gate drops valuable tilts; first-score unbounded anyway |
| v2.7 K=10→K=8 | -20pp | Too shallow for geo's candidate count |
| v3.0 composite value head | -19pp | Survivor_bonus dominates small-scale composite; ranking noisy |
| v3.2 empty_out + tap_capture | -4pp cumulative | Individually noisy; combined drag |
| **geo v3.1 live ladder** | **μ=984.0 floor** | Local panel = v7_0 only; ladder has v3.5.1, v7_pv, top-10 distribution. **Same trap as v3.5.1 on 2026-05-12.** |

## Frictions logged

See `audit/friction.md` 2026-05-13 and 2026-05-14 entries:

1. `tag: geo-v1-three-failed-wallclock-fixes` — when 3 orthogonal knobs
   all regress, the config IS the local optimum; stop tuning.

2. `tag: geo-v2-iteration-trajectory-downward-not-individually-regressing` —
   each addition can be within noise yet cumulatively drag. Need to test
   each individually against the FINAL baseline, not just step-by-step.

3. `tag: local-vs-v7_0-only-misses-ladder-distribution` (NEW, this session) —
   v3.5.1 (2026-05-12, -150μ) and geo v3.1 (2026-05-14, μ=984 floor) both
   regressed live despite strong local A/B vs v7_0 only. **The local
   panel MUST include ≥3 opponent classes** (v3.5.1, v7_pv, v7_0,
   ideally a top-10 bundle). Friction recommendation: add a
   `--vs-panel` flag to `fast.py eval` that runs against a fixed
   3-opponent panel by default.

## Promotion candidates (to `.claude/skills/kaggle-comp/improvements.md`)

1. **Stop tuning when N orthogonal knobs regress.** (from v2.4/v2.5/v2.7)
2. **Local A/B must span ≥3 opponent classes before ladder submission.**
   (from v3.5.1 + geo regressions)
3. **Test each batch component individually against final baseline,
   not against incremental baseline.** Trajectories can drift downward
   even when individual steps are within noise. (from v3.2a/b/c)

## Calibration snapshot

| Submission | Local predicted | Live observed | Δ |
|---|---|---|---|
| **v3.5.1 (5/12)** | +56.6% Wlo vs v3_snipe | μ=945.6 (regression) | -150μ |
| **geo v3.1 (5/14)** | +7pp 2P / +31pp 4P first-place vs v7_0 | μ=984.0 floor (~80 ep) | TBD (could rise) |

**2 consecutive local-overpredict events.** This calibration trend
suggests our local panel systematically over-predicts ladder
performance. Tag this as `tag: local-overpredict-2x` in HANDOVER's
"Where we are."

## What to do next session

(See HANDOVER.md "Next-session first-action" — ranked 1-4.)

Top priority: re-check geo's Score after 24h, run loss-mode diagnostic,
broaden local A/B panel before ANY further submission.

## Time budget

This session: ~12 hours of autonomous iteration (with user direction at
each major branch point). Output: 29 commits, 1 submission, 1 live
ladder data point. Pace: ~25 min/commit, weighted toward eval
wallclock (~10-15 min per n=64 eval × ~10 evals = 2-3 hours of compute
alone).
