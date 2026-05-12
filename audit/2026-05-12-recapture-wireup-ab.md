# Recapture wire-up A/B — NEGATIVE result, REVERTED

**Date:** 2026-05-12. **Branch:** `claude/game-strategy-analysis-XXxEK`.
**Tournament JSON:** `audit/tournaments/20260512T075340Z.json`.

## The plan

Wire `propose_recapture_missions` into `agents/v3_snipe/main.py`
alongside `snipe` + `reinforce`. Prior was HIGH (+100-150 μ) because
`audit/2026-05-11-v3-snipe-games-analysis.md` §2 identified the
comeback gap as the single biggest known weakness: wins recover to a
median 28 planets after losing the home cluster; losses peak at 6.

The plan (approved) gated submission on a 100-seed × 2-side self-play
A/B vs the unmodified v3_snipe, requiring Wilson lower bound ≥ 55%.

## What happened

200 games. Candidate (with recapture) vs baseline (without):

| split | wins / games | winrate | Wilson lo |
|-------|-------------:|--------:|----------:|
| candidate as P0 | 34 / 100 | 34.0% | 0.255 |
| candidate as P1 | 38 / 100 | 38.0% | 0.288 |
| **combined**    | **72 / 200** | **36.0%** | **0.297** |

**Wilson 95% CI on combined: [0.297, 0.429]. Gate FAILED by ~25
percentage points.** This is not a marginal miss: the recapture
wire-up regresses the agent by **14 percentage points** absolute, on
both sides. Reverted in the same commit.

p95 turn time was 22-24 ms on both sides — still well within the 1 s
budget. Speed is not the issue.

## Why the prior was wrong (three hypotheses)

The recapture module (`lib/missions/recapture.py`) is well-tested
(28 / 28 mission tests pass with it registered) and the integration is
the smallest possible change (one import, one `+ propose_recapture(...)`
clause). The regression must be at the **planner level**, not the
mission-class level. Most likely causes:

1. **Score-scale mismatch.** Recapture scores via
   `bonus * value / (0.5 * base_ships + d + 1.0)`, with `bonus` up to
   `1.5×`. Snipe and reinforce use their own scoring formulas
   (`lib/missions/snipe.py`, ~274 lines). If recapture's bonus is on a
   different scale, `settle_plan`'s per-source greedy may
   systematically prefer recapture targets over more valuable snipe
   targets on the same source.

2. **Proposal volume.** Recapture emits one Mission per `(lost_target,
   eligible_source)` pair. With ~5-10 lost planets in a typical
   late-game state and ~16 owned sources, that's 80-160 proposals
   per turn alongside the existing snipe + reinforce volume. Even if
   scores are well-calibrated, the long tail of low-value recapture
   proposals dilutes the planner's per-source decisions.

3. **Premature commitment on hopeless recaptures.** Recapture filters
   `t.ships > RECENTLY_LOST_GARRISON_MAX (=50)` but not by
   recapture-fleet feasibility against the predicted future garrison.
   We may be committing fleets that arrive too late to take the
   planet — burning ships that snipe/reinforce would have spent better.

## Decision

- **No submission.** Per the plan's gate rule, do not push to Kaggle.
- **Code revert.** `agents/v3_snipe/main.py` restored to the
  pre-2026-05-12 form. The recapture module stays on disk (tested,
  good shape) for a future calibrated re-attempt.
- **Strategy menu update.** Option A is downgraded from S-effort /
  HIGH-prior to **M-effort / unknown-prior**, pending the three
  hypotheses above. The keystone for closing the comeback gap is
  not just "register the proposer" — it is "calibrate its score
  scale against snipe/reinforce AND cap its per-turn proposal
  count."

## Suggested next probes (no code committed)

1. **Volume cap.** Limit recapture to the top-K (K=3? K=5?) scored
   missions per turn before passing to `settle_plan`. Same A/B.
2. **Score-scale calibration.** Take 10 lost-planet scenarios from
   the candidate's losing games; compute recapture vs snipe scores on
   the SAME source planet, and check whether recapture's bonus is
   inflating it beyond the snipe winner's score. If so, divide
   `bonus` by a calibration constant fitted to make the
   median-ratio at parity.
3. **Feasibility filter.** Pre-screen recapture targets by checking
   whether `base_ships + production * eta <= source.ships`. Drop
   missions where the time-of-arrival garrison would still beat us.

Each probe is an independent ~1-day cycle gated by the same 100-seed
A/B before any submission.

## What this changes for the strategy plan

- **The recapture gap is real** — see games-analysis §2 — but it is
  not a free same-day win. It is a calibration-and-volume problem.
- **The next strategic option to attempt** should be one whose mission
  class is geometry-isolated and small in volume:
  - **Option C (orbital phase-lead library)** is now the more
    attractive same-day win, because it is pure-library and ships
    as a no-op merge (no agent change yet). It also unlocks
    Options D, G, E later.
  - **Option B (sun-tangent routing)** is a single mechanism change
    in `lib/trajectory.py`; the proposal volume is unchanged (it just
    rescues missions that currently fail `sun_avoid`). Same-day
    submittable after its own A/B.
- The full geometry report
  (`audit/2026-05-12-battlefield-geometry-report.md`) still stands:
  the geometric facts (19.5% sun-blocked pairs, 35% / 75% orbital
  closest-approach savings, 4-fold symmetry on 93% of seeds, 20 free
  comets per game) are all unchanged.

## Reproduction

```
python3 scripts/run_recapture_ab.py --seeds 100 --workers 4
```

Writes `audit/tournaments/<utc>.json`. The harness is kept on disk
(`scripts/run_recapture_ab.py`) for the next attempt; the candidate
file `/tmp/v3_snipe_baseline_main.py` is regenerated by hand-copying
the pre-recapture form of `agents/v3_snipe/main.py`.
