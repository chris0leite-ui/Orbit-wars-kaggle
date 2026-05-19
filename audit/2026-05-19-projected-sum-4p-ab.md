# 4P FFA A/B — favor vs projected_rank_diff_sum — TIE

Branch: `claude/strategy-framework-design-OyoYR` @ `26db155`
Date: 2026-05-19
Tooling: `python -m scripts.ffa_panel --focals submissions/_ab/favor.py
submissions/_ab/projected_sum.py --background submissions/v7_0_drop_one.py
submissions/v4_planner.py submissions/v3.5.1.py --seeds 32 --workers 8`
Games: 32 seeds × 4 seat rotations × 2 focals = 256

## Result

| Focal | 1st-place | Wilson 95% | p95 turn ms |
|---|---|---|---|
| `submissions/_ab/projected_sum.py` | **87/128 (68.0%)** | [59.5%, 75.4%] | 640.4 |
| `submissions/_ab/favor.py` | 85/128 (66.4%) | [57.9%, 74.0%] | 652.7 |

Both clear the absolute gate. Point estimate gap is +1.6 pp in
projected_sum's favour. CI overlap is near-total. **Statistical tie.**

## Compare to prior 4P A/B (`audit/2026-05-19-projected-value-head-4p-ab.md`)

Same 32 seeds × 4 seats × fixed background, same `submissions/v7_0_drop_one.py
+ v4_planner.py + v3.5.1.py`. Only the candidate differed
(`projected` then, `projected_sum` now). Re-running `favor` gave a
**different** result this run:

| Focal | Prior run | This run | Δ |
|---|---|---|---|
| `favor` | 93/128 (72.7%) | 85/128 (66.4%) | **−6.3 pp** |
| `projected (max)` | 82/128 (64.1%) | — | — |
| `projected_sum` | — | 87/128 (68.0%) | — |

`favor` swung 6.3 pp on identical configuration — the per-game RNG is
not fully determined by `configuration.seed`. About half of the prior
run's "favor +8.6 pp" gap was noise. The real gap of the `max`
aggregator was likely **closer to 4-5 pp**, not 8.6 pp.

## Verdict — leaf-side axis exhausted under the baseline-chooser

**The aggregator-fix recovered Variant 1's deficit, but did not lift
beyond favor.** Combining this run with the prior:

1. Variant 1 (max): 4P regression vs favor (4-8 pp net). Falsified.
2. Variant 2 (sum): 4P tie with favor. Per-seat projection adds **no
   measurable lift** when the aggregator matches favor.

The two heads compute different scalars but give the chooser the
same argmax in most frames. Single-game inspection
(`/tmp/inspect_one_4p.py`) confirmed bit-identical trajectories on
the seeds checked. The leaf-side reframing axis is structurally
saturated under the current baseline-chooser architecture (Rule 37
analog at variant count 2 of 3).

## What this tells us about where the bottleneck is

Behavioural-audit signal: focal (any value head) launches 0.06–0.17
per turn against the panel; v7_0 launches 0.31 per turn (matches the
0.57/turn vs 1.0/turn aggression gap from `audit/2026-05-18-archetype
-action-audit-gap-vs-even.md`). The aggression deficit is upstream
of the leaf value — the chooser's argmax over the K=10 rollout filters
candidates before the leaf scoring matters.

Probable bottlenecks (chooser-side, untested this session):
- Reactive opponent model in 4P rollouts may over-estimate counter-
  punishment, killing Δ>0 candidates that real opponents wouldn't
  actually punish.
- Proposer's per-source dedup may starve us of useful joint launches
  that the chooser then cannot recover.
- K=10 horizon may be too short to see the capture's payoff, but in
  4P the opp's response chain is longer — so the leaf often shows a
  loss even when the real outcome is a gain.

## Decision

**Do not submit projected_sum.** Parity with favor + noise → no
expected μ-lift, but evicts `52766596` (μ=1119.6) from rolling-last-2
for nothing. Rule 12 says spend the budget, but Rule 26 says predict-
then-measure: predicted live μ = current ±σ, which is a non-shot.

**Next session direction:** pivot from leaf-side to chooser-side.
Concrete candidates (write the plan next session, don't churn now):
- Reactive-opp-strength A/B in 4P: replace `lite_greedy_policy` with
  `top_tier_mirror_policy` *symmetrically* in baseline-and-action legs
  (CRN-safe). Tests whether weak opp model is suppressing aggressive
  launches.
- K sweep: 10 → 15 → 25 in 4P only; measures whether horizon-too-short
  is the bottleneck.
- Proposer dedup loosening: emit top-K per source (not 1), measure
  whether starvation is the bottleneck.

## Artifacts

- `audit/tournaments/ffa-panel-20260519T104015Z.json` — full per-game JSON.
- `audit/2026-05-19-projected-value-head-4p-ab.md` — prior run (max aggregator).
- `audit/2026-05-19-projected-value-head-2p-ab.md` — 2P A/B (tie).
- `audit/friction.md` — entry `full-panel-AB-before-single-game-evidence`
  (don't run extensive panels before single-game inspection shows
  the change is expressive).
