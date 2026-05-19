# 2026-05-19 — Value-Head Axis Exhausted (3-variant cap reached)

Branch: `claude/strategy-framework-design-OyoYR-rebased` (rebased onto
`origin/claude/audit-workflow-performance-btjeK`).

## TL;DR

Production-compounding value-head reframing (`projected_rank_diff`,
`projected_rank_diff_sum`) tested in three variants. All fail to lift
over the live `favor` head. **Do not submit projected variants to
ladder.** Leaf-side value axis is closed under the baseline chooser;
next move is proposer-side or chooser-side.

## Variants and results

| # | Variant | Baseline | Test | Outcome |
|---|---|---|---|---|
| 1 | `projected_rank_diff` (max-agg over opps) | stale favor (d25e9d3, PV on) | 4P FFA panel n=128 | LOSS −8.6 pp (64.1% vs favor 72.7%) |
| 2 | `projected_rank_diff_sum` (sum-agg in 4P) | stale favor (d25e9d3, PV on) | 4P FFA panel n=128 | TIE (68.0% vs favor 66.4%, CIs overlap) |
| 3 | `projected_rank_diff_sum` (rebased) | **btjeK favor** (A2 + bug fixes) | 4P FFA panel n=128 | **LOSS −22.6 pp** (42.2% vs favor 64.8%) |

Variant 3 is the decisive falsification — the rebased baseline includes
btjeK's A2 4P weakness-exploitation hybrid, bug fixes #3/#4/#11/#12,
PV-off, and the post-submission hold-feasibility filter. Against a
properly-tuned current-state-of-the-art baseline, projected_sum loses
22.6 pp with non-overlapping CIs.

JSON artifacts:
- Variant 1: `audit/tournaments/ffa-panel-<earlier>.json`
- Variant 3: `audit/tournaments/ffa-panel-20260519T134612Z.json`

## Why it failed — single-game deep dive

Deep-dive `/tmp/inspect_deep.py --seed 7 --seat 1 --focal
submissions/_ab/projected_sum_btjek.py`:

| Outcome | projected_sum | favor |
|---|---:|---:|
| CAPTURE | 20 | 17 |
| bounce | 12 | 8 |
| REINFORCE (sent to OUR planet) | 35 | 9 |
| OOB / sun (empty-space) | 24 | 6 |
| **Total launches** | **93** | **41** |
| **Wasted %** (reinforce + OOB) | **63%** | **37%** |

Projected_sum launches **2.3× more** but only **+3 captures** (18% more).
The extra 52 launches are mostly waste — 26 more reinforces, 18 more
OOB/sun, 4 more bounces.

**Mechanism:** the per-seat `P_p × (T − step)` term inflates the value
of marginal candidate-actions. Favor's tighter F1+F2 + A2 hybrid
correctly rejects them. Projected_sum's optimism + linear horizon +
in-flight credit overestimates their value, the chooser's Δ-threshold
greenlights them, and the rollout dutifully simulates fleets that go
OOB or reinforce our own planets. Bad signal at the leaf becomes bad
actions at the root.

This is a NEW failure mode, different from the F4 / dogpile / drift
graveyard:
- Those failed by **CRN symmetry cancellation** — the leaf term was
  zero-mean in expectation between baseline and action legs.
- Projected_sum has full CRN safety (pure state function of the leaf).
  It fails by **action-quality permissiveness** — different mechanism.

p95 wallclock for projected_sum_btjek was 1534 ms — over the 1000 ms
budget. The extra launches inflate K=10 rollout cost, the chooser's
safe_deadline pre-bail drops some turns' decisions. Compounds the
regression.

## Sign-agreement diagnostic (also from the deep dive)

At every sampled turn (step 0..498), both heads agreed on the sign of
V (yes/no winning). They disagree only on magnitude. So the action-
selection argmax converges to similar actions on most frames — but the
∆-threshold cut is different, and that's enough to greenlight bad
marginals.

## Implications

1. **Leaf-side value axis is exhausted.** Three variants tried (Rule 37
   cap). The per-seat ProjectedTotal scalar isn't more accurate than
   favor's split F1+F2 + A2 hybrid in any tested regime.
2. **The bottleneck is proposer-side.** Many candidate launches in the
   proposer's enumeration go OOB / reinforce / bounce. Favor's value
   head was doing real work *filtering them out*. Removing favor and
   substituting a permissive head exposes the proposer's noise.
3. **Forward path:** proposer-side filtering. The
   `a7f9383: proposer hold-feasibility filter` commit on btjeK is
   exactly this direction (it drops captures we can't hold). The
   `chooser_roi`/`trajectory_roi` exploration also saturated at 5
   iterations per the `bdce835` wrap — consistent finding.

## What we don't do

- Do NOT submit projected_sum_btjek. Expected live μ regression ~30-50.
- Do NOT iterate further on value-head variants this session. Cap hit.

## What's next

- (Session-end) Audit + commit this writeup.
- (Next session) Look at the proposer's candidate pool. Specifically:
  - What % of proposed candidates target OOB/sun trajectories?
  - What % target our own planets at fire-time?
  - Can the proposer's `cheap_marginal_value` rank or
    `aim_and_eta` resolution filter these out cheaply?
- (Reference) The `a7f9383` hold-feasibility filter is already on
  btjeK but not in any live submission yet. Worth verifying its
  4P-panel effect cleanly before any further axis work.

## Rule references

- Rule 37 (consecutive-falsification cap on value-head axis): 3
  variants tried, all fail. Axis closed.
- Rule 26 (predict-then-measure): the single-game deep-dive
  predicted the regression before the 4P A/B finished. Aligned.
- Rule 40 (modelling > restriction-tuning): consistent with the
  finding — the issue is upstream (proposer candidates) not
  downstream (leaf valuation tuning).
- Rule 12 (rolling-last-2 discipline): NOT submitting because local
  data predicts ladder regression and would burn a slot.
