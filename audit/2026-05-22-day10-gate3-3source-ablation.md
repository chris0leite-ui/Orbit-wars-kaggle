# 2026-05-22 — Day 10 Gate 3: 3-source ablation verdict

## Summary

Gate 3 asks the fundamental falsifier: does 3-source coordination
EVER produce strictly better bundles than 2-source on the same target?
If essentially never, the 3-source machinery is computational dead
weight.

**Result (8 seeds × ~50 turns × 2 perspectives = 786 samples):**

| Metric | Value |
|---|---:|
| Turns with any 3-source strict win | 14 |
| **Fraction of turns with 3-source win** | **1.8%** |
| Positive lifts observed (across all targets) | 15 |
| Mean lift when 3-source wins | **+9.47** |
| Max lift observed | +24.00 |
| 3-only targets (unreachable by 2-source) | **0** |
| Frac 3-only of all 3-source candidates | 0.0% |

**Verdict: FALSIFIED on the 5% threshold.** Ship at `MAX_BUNDLE_SIZE=2`.

## Nuance: rare but meaningful wins

3-source wins are rare (1.8%) but when they happen, the lift is
substantial (mean +9.47, max +24.00). However:

- The 5% frequency threshold catches this as "below the line for
  default-on behaviour".
- "3-only targets = 0" — 3-source NEVER unlocks targets that 2-source
  can't reach. It's only marginal improvement on already-reachable
  targets.
- Computational cost: enumerating 3-source bundles roughly doubles the
  bundle count in mid-game (more arrival-window subsets). The Tier-2
  budget is already tight (633ms total pipeline vs 600ms agent
  budget). Trimming to 2-source saves compute headroom.

EV trade-off:
- 1.8% × +9.47 ≈ +0.17 mean tier2-score per turn from MAX_BUNDLE_SIZE=3
- Compute pressure relief from MAX_BUNDLE_SIZE=2 enables future
  enhancements (e.g., wider cheap_filter K, longer Tier-2 horizon)
  with much larger EV potential.

## Strategic insight

In symmetric self-play, 2-source coordination captures essentially
all the joint-stacking value:

- By the time 3 sources can simultaneously deliver to a target within
  the arrival window, the target was likely already capturable by 2.
- Most strong attack opportunities are 2-source pair-stacking
  (which minimal's existing joint-pair pass already covers).
- 3-source coordination would matter most against opponents who can
  withstand 2-source attacks — likely against late-game large-fleet
  enemies, not in the mid-game scenarios this probe sampled.

## Decision: MAX_BUNDLE_SIZE default reduced 3 → 2

Code change: `agents/coord/main.py::MAX_BUNDLE_SIZE = 2`.

Constraint test (`test_enumerate_respects_bundle_size_cap`) still
passes — the assertion is `len(legs) <= MAX_BUNDLE_SIZE` which now
caps at 2 instead of 3. All 73 unit tests green.

Revisit in v2 if Gate 4 multi-opponent panel shows specific
opponents where 3-source coordination would have helped (e.g.,
late-game ladder agents with large garrison thresholds).

## Acceptance: gate proceeds

Gate 3 falsified the strong-form premise (3-source as default) but
the design machinery remains valuable: the Lagrangian + cheap-filter
+ Tier-2 pipeline handles 1-source and 2-source bundles cleanly,
which IS the actual production scope. Day 11 (Gate 4 multi-opponent
panel) proceeds at MAX_BUNDLE_SIZE=2.

## Artifacts

- `audit/20260522T131005Z-gate3-3source-ablation.json` — full 786-sample
  probe result.
- `audit/20260522T125954Z-gate3-3source-ablation.json` — earlier
  2-seed × 15-turn smoke (0 3-source candidates in early game).
- `scripts/check_coord_3source_ablation.py` — re-usable probe.
