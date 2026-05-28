# 4P delayed-launch cushion — falsified

Date: 2026-05-28. Branch: `claude/game-theory-winning-strategy-SEU7P`.
Companion to `audit/2026-05-28-peak-53088099-share.md` (the
motivating measurement) and `audit/2026-05-27-rf-v1-root-cause.md`
(the upstream chooser falsification).

## Setup

`agents/baseline_4p_cushion/main.py` — thin wrapper that returns no
actions in 3P/4P games for the first 60 ticks, then delegates to
`agents/baseline/main.agent`. Motivating doctrine prediction
(`evaluation-metrics.md` §5, n=92 empirical): 4P winners' first-
capture median is step 137; 4P losers' is 72; current Kaggle peak
launches at step ~20 in 4P. Cushion gate puts first-launch into the
"delay" regime to test whether matching the winner-fingerprint
yields ladder lift.

## A/B

`scripts/ffa_tournament.run_ffa_tournament`, 4 seats, rotated, 8
seeds × 4 rotations = 32 games per focal. Two background panels
ran against the same two focals (cushion-on, baseline-no-cushion):

| Background | Focal | First-place / n | Rate | Wilson-lo |
|---|---|---:|---:|---:|
| 3× random | cushion | 32/32 | 1.000 | 0.893 |
| 3× random | baseline | 32/32 | 1.000 | 0.893 |
| 3× nearest | cushion | **4/32** | **0.125** | 0.050 |
| 3× nearest | baseline | **26/32** | **0.812** | 0.647 |

vs random: both crush (random can't punish the cushion). vs nearest:
**cushion loses 22 pp to baseline.** Cushion is **catastrophically
worse**, not just slightly weaker.

## Verdict

**Falsified.** The doctrine's 4P-winners-launch-later fingerprint
does not translate into actionable improvement on our μ-band when
operationalised as "no launches for the first N ticks." Two
explanations consistent with the data:

1. **Correlation, not causation.** 4P winners in the n=92 study
   launched later because their geometries (initial-cluster
   placement, comet timing, opponent skill) gave them spare time
   — not because they chose to delay. Forcing delay against
   capable opponents just gives away the early-neutral phase.

2. **The cushion is too coarse.** Real 4P winners may launch
   DEFENSIVELY and SMALL in the early phase, not be silent. A
   pure-silence gate throws out the defensive launches that
   protect home while accumulating cushion. v2 of this fix would
   need a per-source defensive carve-out, not a blanket gate.

(2) is testable as a v2 variant, but Rule 37 caps the axis at 3
variants and the directionality is so negative (12% vs 81%, a
6-7× gap) that a further variant on this axis is hard to justify.

## Implications for the doctrine

This is the **third consecutive falsification** on the chooser /
doctrine line:
- v1 chooser (0/20 vs baseline).
- v2 chooser with hold-floor + gang-up (0/32 vs baseline).
- 4P cushion gate (4/32 vs nearest, against baseline's 26/32).

The doctrine's MATH remains sound (n=92 share-of-integral
separation 0.488 between winners and losers is a real effect).
What's now empirically falsified is the doctrine's PRESCRIPTIVE
power on our μ-band — none of its operational predictions
(closed-form ρ chooser, hold-floor + gang-up, delayed launch in
4P) survive contact with baseline-strength opponents.

For the user's original question ("can we use what we've learned
to improve our best Kaggle submission"): the chooser-line
learnings have not produced an actionable lift. Three independent
operationalisations have failed. The doctrine's contribution to
this comp is now: durable measurement infrastructure (Rule 48,
`measure_hold_times`, share-of-integral) and a documented
falsified prediction set — useful as durable knowledge but not a
ladder move.

## Status

Doctrine-derived improvements to baseline: **exhausted on this
axis.** No further work proposed on the chooser / doctrine line
without a fundamentally different operationalisation.

The wrapper agent at `agents/baseline_4p_cushion/` is kept in tree
as the falsified-variant reference (useful for any future
"reproduce-the-null" check), but not a submission candidate. Rule
1: no submission. Rule 42: no push-claim row appended.
