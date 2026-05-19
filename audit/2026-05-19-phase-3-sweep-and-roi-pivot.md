# Phase 3 compound-weight sweep + ROI / scenario-gate pivot

**Date**: 2026-05-19
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Outcome**: Phase 3 lever fully characterised (saturates at 0.3).
PI redirected from "iterate bundle scorer" to "drop bundle decision
stack, rebuild clean ROI on the kept architecture, gate it on
observation-grounded synthetic scenarios."

## Phase 3 compound-weight sweep — final results

`scripts/phase_c_ab.py` with `BUNDLE_COMPOUND_WEIGHT ∈ {0.05, 0.1, 0.2, 0.3, 0.5}`. 
Same panel as Phase C: 8 seeds × 2 sides × 2 matchups = 32 games per weight.
Bundle = cands=5, lead-aim OFF (post `4bd2eec` revert).

| compound_weight | vs v7_0 | vs baseline | baseline Wlo | note |
|---|---|---|---|---|
| 0.0 (control) | 13/16 = 81% | 2/16 = 12.5% | 0.035 | Phase C result |
| 0.05 | 13/16 = 81% | 2/16 = 12.5% | 0.035 | too small, no effect |
| 0.1 | 13/16 = 81% | 3/16 = 18.8% | 0.066 | +1 baseline win, no v7_0 cost |
| 0.2 | 11/16 = 69% | 4/16 = 25% | 0.102 | first time Wlo > 0.10 vs baseline |
| **0.3** | **11/16 = 69%** | **5/16 = 31%** | **0.142** | peak vs baseline |
| 0.5 | 11/16 = 69% | 5/16 = 31% | 0.142 | identical to 0.3 — saturated |

**Pareto:**
- `0.1` is the free Pareto point (no v7_0 regression, +1 vs baseline).
- `0.3` is the peak vs baseline (+3 vs baseline, -2 vs v7_0).
- `≥0.3` is flat — the lever has no remaining headroom.

Raw tournament JSONs:
- `audit/tournaments/20260519T031944Z.json` (compound=0.1)
- `audit/tournaments/20260519T041254Z.json` (compound=0.05)
- `audit/tournaments/20260519T044849Z.json` (compound=0.2)
- `audit/tournaments/20260519T052310Z.json` (compound=0.3)
- `audit/tournaments/20260519T054231Z.json` (compound=0.5)

## Strategic verdict

`Wlo=0.142` is the bundle-vs-baseline ceiling on the current scorer
axis. Below the 0.55 gate by a wide margin. The bundle decision stack
(chooser, scorer coefficients, candidate enumeration) has been
characterised to exhaustion across Phase A→E.

Pattern from 5+ sessions on this axis (per
`knowledge-base/thoughts/2026-05-16-chooser-family-saturation.md` and
2026-05-18 strategic-redirect):
- Mechanism nulls dominate (joint-coordination Phase 1, bounce-penalty
  Phase 2 — both NULL on win-rate despite mechanism working).
- Single positive signal of this session is compound-weight (Wlo 0.035
  → 0.142). Real, but small.
- Rule 37 axis-cap was already triggered before this session; we ran
  3 more variants past the cap.

## PI redirect ratified this session

Drop bundle's decision stack entirely. Keep `lib/*` primitives
(trajectory_layer, world_model, trajectory, fast_sim, opp_model).
Rebuild a clean ROI agent at `agents/trajectory_roi/` using a
**6-primitive modular architecture** so scenario-compliance fixes
extend a primitive instead of bolting on hotfixes (Rule 40):

1. `enumerate()` — single + multi-source candidates as first-class
2. `predict_arrival()` — owner / garrison / delivered at arrival
3. `reachable()` — ray-cast through trajectory layer
4. `score()` — value-at-target minus cost-at-source over committed-ships
5. `refine_via_rollout()` — opt-in fast-sim K-turn rollout for margin-risky
6. `select()` — greedy with WorldModel dedup

Gate: ROI must pass 100% of an observation-grounded scenario suite
BEFORE any tournament A/B.

## Five named failure modes (PI's live observations)

To be encoded as V0 scenarios next session:
- **(a) Recapture-loss** — fleets to planets recaptured before/after arrival.
- **(b) Drift-loss** — eccentric-orbit drift or sun-blocks-raycast.
- **(c) Garrison-counter** — neutral adjacent to strong enemy whose
  perfectly-timed counter dominates.
- **(d) Split-majority capture** — multi-source coordination failure.
  Canonical: 100+100 vs 50, solo exposes source.
- **(e) Distant-planet idleness** — far-from-front-line planets sit
  idle on huge ship counts; should bundle-forward or large-fast strike.

## What to NOT do (PI ratified)

- Speculative scenario authoring. Scenarios come from observed live
  failures (replay-mined or PI-named), not hypothetical mistakes.
- Hotfix patches in ROI. Every scenario-compliance is a primitive
  extension. No `if scenario_pattern_X: do_Y` patterns.
- Submit bundle. Bundle is v7_0-class; pushing it evicts the live
  champion. Bundle remains in repo (`agents/bundle/`) but is not
  iterated.

## Approved plan reference

`/root/.claude/plans/no-go-forward-test-fluttering-token.md` — final
approved plan, ready for next-session execution.

## Files this session touched

- `agents/bundle/main.py` — compound_weight env var (added, but unused
  in next plan; will be dropped along with the rest of bundle's
  decision stack)
- `lib/trajectory_layer.py:BundleEvaluator.score` — compound term
  (same fate)
- 5 tournament JSONs under `audit/tournaments/`
- This audit doc
