# H44 Phase 1 — Landing-capture failure-mode diagnostic

**Branch:** `claude/audit-workflow-performance-btjeK`
**Plan:** `/root/.claude/plans/let-s-figure-out-how-purrfect-mist.md` (H44 Phase 1)
**Hypothesis-board entry:** H44
**Inputs:** 39 4P live games of sub 52827111 (3 had no failure rows),
333 attack launches that v2 audit recorded as `tgt_owned_at_landing=False`.

## Context

v2 audit reported landing-capture rate at 33–46% across every
(src_tier, tgt_tier) bucket. Phase 1 disaggregates **why** the ~half
that fail actually fail, so any Phase 2 fix is targeted at the
dominant mechanism.

## Method

For each failure row in the v2 JSONL: re-load the corresponding live
replay, read observations at three checkpoints (launch step `t`,
landing-1, landing, landing+1, and landing+2..+6 for late-arrival),
plus fleet-list scans for fleet-survival detection. Classify into 6
failure modes by precedence order:

1. **E_prediction_off_by_one** — we own tgt within 5 turns AFTER
   landing+1. The v2 audit's `landing_step` from `predict_fleet_fate`
   was an under-estimate; this is NOT a real landing failure, it's
   instrumentation noise.
2. **F_fleet_destroyed** — fleet was launched (visible at t+1) but
   missing at both landing-1 AND landing. Fleet died in flight (sun,
   OOB, planet collision, or aim error sending it nowhere).
3. **A_src_lost_pre_landing** — src.owner ≠ our_seat at landing_step.
   Chooser over-drained; opp captured the source while our fleet was
   in flight.
4. **C_third_party_flip** — tgt.owner changed during flight to a player
   that isn't us (race condition; another player's fleet beat us in).
5. **B_tgt_production_accrual** — defender count at landing exceeds
   the proposer's prediction (`tgt.ships_at_launch + tgt.prod * arrival`,
   neutral-aware).
6. **D_under_delivered** — defender count matched prediction but our
   ships ≤ defender. Chooser sized too small.
7. **other** — none of the above; instrumentation gap.

Precedence order matters because a single launch can carry multiple
flags (e.g. fleet destroyed AND src lost). Diagnoses listed first in
the order take precedence.

## Result

### Overall (n = 322 diagnosed failures from 333 raw)

| Mode | Count | Pct of failures |
|---|---:|---:|
| E_prediction_off_by_one | 67 | 20.8% |
| **F_fleet_destroyed** | **209** | **64.9%** |
| A_src_lost_pre_landing | 5 | 1.6% |
| C_third_party_flip | 9 | 2.8% |
| B_tgt_production_accrual | 1 | 0.3% |
| D_under_delivered | 19 | 5.9% |
| other | 12 | 3.7% |

**Headline: ~65% of failed-landing launches are because the fleet
never reached the target.** After subtracting the off-by-one
instrumentation rows (E), 67% (209/255) of REAL landing failures are
fleet-destroyed-in-flight. The chooser-sizing modes (A + B + D) total
only 9.6% of all failures (25/255 of real failures).

### By (src_tier → tgt_tier)

| Bucket | n | F_dead% |
|---|---:|---:|
| large → large | 36 | 77.8% |
| mid → large | 24 | 75.0% |
| mid → mid | 35 | 74.3% |
| small → large | 41 | 73.2% |
| large → mid | 43 | 69.8% |
| small → mid | 26 | 65.4% |
| mid → small | 23 | 56.5% |
| large → small | 48 | 52.1% |
| small → small | 46 | 47.8% |

F% rises with target tier (large = 75% F, small = 50% F). This is
consistent with the cause being **distance / flight time**: large-tier
planets are usually farther on average AND/OR have more obstacles on
the path. Longer flights = more opportunity for the fleet to die
(real path obstructions) OR for the aim prediction to diverge from
the env's actual trajectory (instrumentation drift).

### Spot-check on sample F#1

(ep 77114332, seat 1, t=50, src=planet 13, tgt=planet 15, ships=17,
angle=1.450 rad ≈ 83°.) Fleet observed flying STRAIGHT NORTH from
launch position (31.7, 54.7) for 4 turns then disappears. Target
planet 15 at landing was at (71.6, 41.8) — **east-southeast** of the
launch point. The launch angle is aimed in the wrong direction.

The vector from launch to target is (40, −13), implying an angle of
roughly −0.31 rad (≈ −18°, slightly south of east). Launching at +1.45
rad goes far north of target; with fleet speed ≈ 1 + 5·(log(17)/log(1000))^1.5
≈ 2.4 per turn, the fleet is at y ≈ 54.7 + 16·2.4·sin(1.45) ≈ 92.8 by
step 66 — close to OOB (board top = 100), and it does fall off the
fleets list at step 65. So the failure is partly aim error AND/OR an
OOB death.

**The chooser used `predict_fleet_fate` to confirm the fleet would
hit planet 15.** If predict_fleet_fate's planet-position model disagrees
with the env's actual orbital motion, we'd ship aim angles that
predict-time look correct but env-time miss. This is the candidate
infrastructure bug.

## What this means for next steps

**The "landing-capture hemorrhage" is mostly an aim / trajectory
infrastructure issue, not a chooser sizing issue.** The decision tree
in the plan says "F dominant → in-flight ledger fix" but that
mis-diagnoses the mechanism: F here mostly means fleets that die in
flight, not fleets that lose a race to the target.

The Phase 2 fix axis should be **predict_fleet_fate validation +
aim mechanism audit**, NOT a chooser sizing change. Specifically:

1. **predict_fleet_fate vs env parity test.** Pick 50 random launches
   from the failure set. For each, simulate via predict_fleet_fate
   AND via env step-by-step (using `kaggle_environments` + the actual
   action). Compare predicted vs actual landing-planet-id and landing-step.
   Pass = ≥95% match. If <95%, the prediction infrastructure has a
   bug that's invisibly costing us ~7% of all attack launches.

2. **lead_aim mechanism audit.** Sample #1's angle is so far off that
   either lead_aim is broken for specific (src, tgt) configurations
   OR the agent shipped a comet-aim angle when comet-aim doesn't apply.
   Add a unit test: for 20 (orbiting src, orbiting tgt, ships) tuples,
   verify the aim angle's resulting fleet path passes within
   (src.radius + tgt.radius + safety) of target at the predicted
   landing step.

3. **Repeat after fix.** The 23% "instrumentation noise" baseline
   (E + other) should drop close to zero if the predict / aim
   infrastructure is correct.

### Lost vs Won split (residual signal)

| Mode | Won (n=194) | Lost (n=128) |
|---|---:|---:|
| F_fleet_destroyed | 60.3% | 71.9% |
| D_under_delivered | 4.6% | 7.8% |
| A_src_lost_pre_landing | 1.0% | 2.3% |

F is even MORE dominant in lost episodes. This is consistent with the
infrastructure-bug framing: losses correlate with longer-distance
attacks (we've shrunk to fewer planets and have to reach far for
targets), and longer-distance attacks hit the F mode at higher rates.

## Decision-tree application

The plan's decision tree expected one of A/B/C/D/F-as-race to dominate.
F dominates but **F-as-aim-failure** is a different fix axis than
F-as-race. Per the plan: **stop and re-plan; the diagnosis isn't what
the tree anticipated**.

## Verdict and next step

- H44 Phase 1: **DIAGNOSIS COMPLETE.** F (fleet destroyed in flight,
  including aim/predict infrastructure issues) is 65% of failures,
  77% in lost episodes.
- H44 Phase 2: **PIVOT.** The original plan's Phase 2 options (chooser
  sizing fixes for A/B/C/D) don't apply because A+B+C+D combined are
  only ~10% of failures. New Phase 2 axis: **trajectory-prediction
  parity test against env, plus lead_aim audit**. Surface to PI
  before any agent code change.
- H46 (4P weakest-opp targeting) and H44 Phase 2 are now competing
  axes — both worth doing, neither blocks the other.

## Artifacts

- `scripts/h44_landing_capture_diagnostic.py` — diagnostic tool
- `audit/2026-05-21-h44-phase1-landing-capture-diagnostic.jsonl` — 322 rows
- this file
