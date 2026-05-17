# Direction A endpoint — v4 trajectory chooser A/B results

> Filed 2026-05-17 as a checkpoint before investigating
> `predict_fleet_fate` false rejects. Captures the empirical state at
> the end of the Direction A iteration (per
> `knowledge-base/concepts/probability-of-winning-framework.md`).
> Branch: `claude/audit-workflow-performance-btjeK` @ `b8da492`.

## Final A/B numbers (trajectory chooser vs v15 `/tmp/v15_resurrect/main.py`)

| variant | leaf | filter | emit logic | n | wins | rate | Wlo |
|---|---|---|---|---:|---:|---:|---:|
| v3 | binary owner-check | on | multi-launch | 32 | 0 | 0.0% | 0.000 |
| v4-favor-on-multi | favor | on | multi-launch | 32 | 12 | 37.5% | 0.229 |
| v4-hybrid-on-multi | hybrid (composite/A2) | on | multi-launch | 32 | 11 | 34.4% | 0.204 |
| v4-hybrid-on-single | hybrid | on | composite (1-per-src) | 32 | 11 | 34.4% | 0.204 |
| **v4-hybrid-OFF-single** | hybrid | **off** | composite (1-per-src) | 64 | 31 | **48.4%** | 0.366 |
| composite_a2 (shipped) | hybrid | n/a | composite (1-per-src) | 64 | 43 | **67.2%** | 0.550 |

Wallclock: v4 max turn-ms 1145-1342, similar to composite_a2's 1196-1580.

## Two quantified bugs

1. **Binary leaf (v3 → v4)**: ~37pp cost. The owner-check leaf collapses
   strategic information that the continuous `favor` leaf preserves.
   Confirms Hypothesis 1 from the prior 3-suggestions analysis.
2. **predict_fleet_fate filter (v4 ON → v4 OFF)**: ~14pp cost. The
   `lib.trajectory.predict_fleet_fate` primitive is false-rejecting
   candidates that the engine + fast_sim rollout would accept. **This
   primitive is also used in production code** (`lib.mechanism.sun_avoid`
   in `v3_snipe` missions; `agents/baseline/proposer.PROPOSER_TRAJECTORY_FILTER`).

## Remaining gap (~19pp) — most likely cause

`wait_N>0` handling. v4 drops these candidates entirely; composite chooser
RESERVES src+tgt (effectively delaying a launch for accumulation). The
proposer emits wait_N=0, 5, 12 variants per (src, tgt); if composite picks
wait_N=12 sometimes and that's strategically valuable, dropping them costs.
Not investigated yet — out of scope this checkpoint.

## What this checkpoint preserves

- v4 implementation (`agents/baseline/chooser_trajectory.py`) with all
  diagnostic env-vars: `BASELINE_CHOOSER=trajectory_v3`,
  `TRAJECTORY_SKIP_ADMISSIBILITY=on`. Reproducible.
- The 4 concept docs framing this work:
  - `trajectory-first-architecture.md`
  - `trajectory-chooser-v2-sketch.md`
  - `probability-of-winning-framework.md`
  - (this file as the empirical endpoint)
- Open question for next session: **why does `predict_fleet_fate` produce
  false rejects?** That investigation is the next plan (see
  `plans/you-are-a-senior-wild-hamming.md` for the playbook).

## Lessons (for the postmortem skill)

1. **Successive falsification of refined hypotheses works.** Per Rule 4
   ("never give up; saturation is bounded"), 4 trajectory chooser
   variants taught us more than 1 would have. We climbed 0 → 48.4%
   vs v15 by fixing one bug at a time.
2. **Per-fleet binary scoring is too coarse.** The "trajectory thinks
   in fleet outcomes" intuition needs continuous-favor scoring to be
   competitive.
3. **A static primitive used as a hard filter must match the dynamic
   ground truth to within noise.** A 14pp false-reject rate is enough
   to make a structurally-correct chooser look broken.
4. **Live-ladder μ readings settle slowly.** 52744856 dropped to 1041
   in early-window noise and is now at 1106.9 (near v15). The
   `early-trueskill-mu-unreliable` friction tag earned its keep.
