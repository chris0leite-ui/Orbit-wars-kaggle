# Spatial leaf — negative result postmortem (2026-05-18)

## TL;DR

Implemented `favor_hybrid_spatial` (per-ship weight inverse of
distance-to-nearest-non-our-planet) targeting the 43.8% isolated
ship-turns measured in `audit/replays/idle-trajectory-2026-05-17.md`.
A/B FAILED:
- **2P** (clean bundle, n=64, vs forced-hybrid trajectory): 26/64 =
  **40.6%** Wlo=0.295 Whi=0.529 — significant regression
- **4P** (8 seeds × 4 seat-rotations, vs 3x hybrid trajectory):
  **3/32 first-place = 9.4%** Wilson95=[3.2%, 24.2%] — way below
  uniform 25%; **max turn-ms=1503**, 6/32 games over 1000ms cap
- 2P wallclock: max=2541ms, p95=991ms — over env actTimeout

**Submission: NO.** Current production (trajectory+hybrid, μ=1271.8
settled) preserved.

## What worked (audit infrastructure)

`scripts/idle_trajectory_audit.py` measured the leak quantitatively:

| sub | μ | isolated | rear | long-launch | staging-opp |
|---|---:|---:|---:|---:|---:|
| 52754310 trajectory+hybrid | 1271.8 | 43.8% | 10.7% | 11.6% | 46.2% |
| 52744856 composite_a2 | 1152.7 | 34.9% | 16.4% | 8.4% | 40.8% |
| 52710995 v15 | 1115 | 40.1% | 16.4% | 8.3% | 41.3% |

Conclusion of audit: spatial leaf is high-leverage if it works
(40%+ of ship-turns isolated); staging proposer is marginal (only
~12% of launches are long and ~46% of those had a staging option).

The audit script + tagged outputs are reusable and stay in repo.

## Why the spatial leaf failed

Hypothesis-A: **double-counting**. `favor_hybrid` (composite in 2P)
already credits capture EV via `production × time_remaining` for
arriving fleets. The spatial term ADDS pull toward non-our planets,
but this is correlated with where captures already pay off. Net
effect: chooser over-weights forward launches that don't actually
capture, AND under-weights defensive/reinforcement launches that
keep our position safe.

Hypothesis-B: **proposer-chooser mismatch**. The proposer's
cheap-rank uses `composite_capture_value`-style scoring. The chooser
validates with the spatial-augmented head. When the proposer ranks
candidates expecting one valuation but the chooser rescores with
different weights, the top-K candidates that reach validation are
biased toward the proposer's view, not the spatial view. The
chooser then sees marginal candidates with high spatial value but
low capture value — and emits them anyway because spatial pushes
Δ above 0.

Hypothesis-C: **wait-N timing breakage**. The chooser's wait-N
candidates score against the idle baseline. With spatial active,
idle baseline tracks the spatial value of staying put. Wait-N
candidates that "fire later from same source" have nearly identical
spatial position to fire-now (the source planet's position is the
same). So spatial doesn't discriminate wait-N candidates well, but
DOES bias all firing candidates upward — collapses the wait-N
calibration.

Most likely: all three contribute. The spatial term is not the
right modeling correction for the isolated-ship-turns leak.

## Wallclock issue (max 2541ms)

In solo bench: max=657ms at SPATIAL_WEIGHT=0.5. Under A/B contention
(two evals running in parallel, OS CPU saturation): max=2541ms.

The spatial term itself is cheap (O(my_planets × non_our_planets) per
favor call, ~30-50 ops). But it ADDS work per-call, and under CPU
starvation the chooser's `safe_deadline` pre-bail may not fire fast
enough — the engine timeout (1000ms `actTimeout`) wins and the agent
loses moves.

The 4P bench showed max=1503ms in clean (low-contention) conditions.
4P has more planets so spatial cost is higher per call, AND more
candidates to evaluate. Net: spatial is wallclock-fragile in 4P.

## 2P-only fix attempted

Patched `favor_hybrid_spatial` to short-circuit when `num_seats > 2`
(commit 558bd61). Spatial in 2P, hybrid in 4P. This handles the 4P
regression but does NOT fix the 2P A/B failure (40.6% still bad).

The 2P-only variant is opt-in via `BASELINE_VALUE_HEAD=hybrid_spatial`
and stays in the codebase as reference but is not the default.

## What's preserved

- Current production: 52754310 (trajectory + hybrid) at μ=1271.8.
- Rolling pair partner: 52744856 (composite_a2_hybrid) at μ=1152.7.
- Daily submission budget: 5/day; 5/17 used 3, 5/18 used 0.

No submissions this session — the floor (1152.7) stays.

## What this session shipped

- `scripts/idle_trajectory_audit.py` — re-runnable measurement of
  idle ship-turns + launch ETA distribution per submission.
- `agents/baseline/value.favor_hybrid_spatial` — opt-in spatial
  leaf (2P only via num_seats short-circuit). Default OFF.
- New env vars `BASELINE_SPATIAL_WEIGHT`, `BASELINE_SPATIAL_DECAY`
  for follow-up tuning.
- 7 new unit tests in `tests/test_baseline_value.py` (24/24 green).
- Friction tag `env-var-shared-process-breaks-ab-isolation` —
  documents the within-process A/B isolation issue.

## Next session candidates (ranked by EV / cost)

1. **Direction B (joint candidate evaluation)** — pre-scoped in
   `knowledge-base/thoughts/2026-05-17-direction-b-joint-action-
   scoping.md`. The idle-fleets problem might be better addressed
   by joint multi-step planning than by single-step leaf tweaks.
   Joint scoring captures "launch A→B to stage, then B→T to
   capture" naturally.

2. **Different positional formulation** — instead of pulling toward
   non-our PLANETS, pull toward CONTESTED neutrals only (excludes
   non-capturable opp planets). Or pull toward fleet centroids.
   Or pull toward "high-EV" planets weighted by production. Rule
   37 budget still allows 2 more variants on the positional axis.

3. **Rule 22 — mine top-5 public notebooks**. We're at μ=1271.8,
   likely top 1% but not top-of-LB. The romantamrazov reference
   (μ=1224) was 47 below us; top of LB may be 1300+. Mining
   their structural choices might reveal the next gap.

4. **Staging proposer** — only 12% of launches are long-ETA, of
   which 46% have staging options. Low leverage; deprioritize.

## Constraints on next session

- Trajectory chooser at μ=1271.8 is well above local A/B prediction
  (was 65.6% point estimate vs v15 implying ~1140 mu; got 1271).
  Local A/B undercalls live performance for this architecture.
  Future A/B-positive results should be SCALED UP cautiously.
- Don't trust within-process A/B without module isolation (see
  friction `env-var-shared-process-breaks-ab-isolation`).
- 4P performance is HARD to test locally and matters substantially
  (36% of ladder games). New value heads must include 4P regression
  check before submission.

## Rule applications

- **Rule 40** (modeling vs restriction-tuning): the spatial leaf
  IS a modeling fix, not a restriction. Rule 40 was respected.
  The modeling fix happened to be wrong. Rule 40 doesn't guarantee
  the modeling fix is correct — just that you should TRY modeling
  before restrictions.
- **Rule 37** (3-variant axis cap): spatial-positional is now 1/3
  variants on the "positional value in leaf" axis. 2 more available.
- **Rule 22** (plateau public-notebook scan): not yet fired this
  session; queued for next.
- **Rule 1** (submission discipline): no submission without PI
  approval AND positive A/B. A/B failed → no submission. Floor
  preserved.
