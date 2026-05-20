# H44 Phase 1 — CORRECTION: F-flag was a false positive

**Branch:** `claude/audit-workflow-performance-btjeK`
**Date:** 2026-05-21 PM (correction of same-day finding)
**PI catch:** "I have not seen fleets getting out of bounds or missing
targets. Give me an example to verify myself."

## The error

The first H44 Phase 1 commit (`106afbe`) reported F (fleet destroyed
in flight) as 65% of failures and claimed an aim/predict_fleet_fate
infrastructure bug. PI was correct to push back. The F flag fired
whenever the fleet wasn't visible in the fleets list at landing−1 OR
landing — but **fleets vanish from the list when combat resolves at
the target**, regardless of outcome. F was over-firing on launches
that arrived correctly but lost combat.

Spot-checked 5 F-flagged launches: all had miss distance < target
radius (i.e. the fleet was at or inside the target circle at landing).
None went OOB. The aim was fine.

## Corrected diagnosis (52827111, μ=1136.6)

322 diagnosed failures, with F removed and G (near-tie combat) added:

| Mode | Count | % of failures | % of real failures (excl. E) |
|---|---:|---:|---:|
| E_prediction_off_by_one | 67 | 20.8% | — (instrumentation noise) |
| D_under_delivered | 60 | 18.6% | 23.5% |
| C_third_party_flip | 55 | 17.1% | 21.6% |
| A_src_lost_pre_landing | 41 | 12.7% | 16.1% |
| other | 77 | 23.9% | 30.2% |
| B_tgt_production_accrual | 14 | 4.3% | 5.5% |
| G_near_tie_combat | 8 | 2.5% | 3.1% |

### Lost-episode focus (the half we care about most)

| Mode | Lost (n=128) | Won (n=194) |
|---|---:|---:|
| D_under_delivered | **24.2%** | 14.9% |
| A_src_lost_pre_landing | **21.9%** | 6.7% |
| C_third_party_flip | 17.2% | 17.0% |
| other | 19.5% | 26.8% |
| E_off_by_one | 13.3% | 25.8% |
| B / G | 3.9% | 8.8% |

**In lost episodes, A + D = 46.1%.** These are both chooser ship-sizing
failures:
- **D (under-delivered, 24%):** the chooser sent too few ships; arrived
  and lost combat against an unchanged defender.
- **A (over-drained source, 22%):** the chooser sent so many ships that
  opp captured the (now-defended-by-few-ships) source before our fleet
  landed.

These are not the same fix — D wants MORE ships, A wants FEWER — but
they share the underlying root: **the chooser's sizing model doesn't
correctly balance "deliver enough to capture" against "leave enough
to defend the source."**

The 30% "other" bucket remains an instrumentation gap. Plausible
contents: env combat-math edge cases (tie behavior; production
accrual on planets owned during partial frames; comet collisions in
the path; multi-fleet co-arrival).

## Concrete examples (for PI verification)

(All from ep `77114332` and `77114661`, 4P FFA, viewable via
`https://www.kaggle.com/competitions/orbit-wars/leaderboard?dialog=episodes-episode-<EPID>`.)

| Ep | Turn | Action | Defender at launch | Defender at landing | Result | Mode |
|---|---:|---|---:|---:|---|---|
| 77114332 | 50 | 17 ships, planet 13 → 15 | 16 (neutral) | 16 (still neutral) | tie | G_near_tie |
| 77114661 | 42 | 31 ships, planet 11 → 15 | 22 (opp 2) | 24 (opp 0) | third-party | C_race |
| 77114661 | 90 | 85 ships, planet 3 → 9 | 84 (neutral) | 6 (opp 1) | third-party | C_race |
| 77114661 | 94 | 42 ships, planet 7 → 8 | 84 (neutral) | 84 (still neutral) | under-sized | D |
| 77114661 | 116 | 52 ships, planet 12 → 0 | 59 (opp 2) | 56 (still opp 2) | under-sized | D |

In every case the fleet's predicted endpoint is within target radius —
the fleets arrive correctly. They just lose combat or arrive too late.

## Impact on Phase 2 design

The original plan's decision tree fits AFTER all (Mixed → re-plan).
But now we have a cleaner two-prong story for lost episodes (A+D = 46%).
Phase 2 options to consider:

1. **Chooser sizing recalibration** (Rule 40 modeling fix):
   - Inspect the chooser's `score_candidate_v4` ship-sizing logic for
     why under-delivery is so common (D = 24% in lost episodes).
   - Inspect the drain-defense math (`_source_survives_launch` at
     `proposer.py:438`) for why source losses still happen (A = 22%).
2. **In-flight ledger for C** (race conditions, 17% of lost-ep failures):
   - The ledger that 5/21 morning falsified was for OUR commits; a
     ledger for OPP'S in-flight fleets converging on the same target
     is a different mechanism, not the same axis.
3. **Combat-math audit** for the residual "other" 19% — likely
   reveals one or two env-rule edge cases we're modeling wrong.

Surface for PI re-plan. The previous Phase 2 recommendation
(predict_fleet_fate parity test) was based on the F-flag false positive
and is NO LONGER the recommended axis. Aim is fine.

## Artifacts

- `scripts/h44_landing_capture_diagnostic.py` — corrected diagnostic
- `audit/2026-05-21-h44-phase1-CORRECTED.jsonl` — 322 re-diagnosed rows
- this file (supersedes `audit/2026-05-21-h44-phase1-landing-capture-diagnostic.md`
  which contains the original F-flag-overcount story)

## Process learnings

- **A fleet's absence from the fleets list is not evidence of failure.**
  Fleets always vanish at combat resolution. To detect mid-flight death,
  need a stronger signal — e.g. compare last_seen_position against
  target position at time of last sighting; if far from any planet
  AND fleet missing next step, that's a true mid-flight death.
- **Spot-check before publishing.** I should have hand-traced one F
  example before writing the verdict. The PI's "I haven't seen fleets
  going OOB" challenge would have surfaced the bug in 5 minutes of
  thought. Promotion candidate: any diagnostic claiming a dominant
  mode must show 3 hand-traced examples that match the diagnosis.
