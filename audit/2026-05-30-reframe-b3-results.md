# B.3 CRN-paired advantage — corpus pipeline, smoke verdict, A/B results

**Date:** 2026-05-30  **Branch:** `claude/competition-objective-alignment-hqNVM`
**Predecessor:** B.2 first cut falsified 0/32 (selection bias).
**Successor:** TBD — pending production-scale decision after launch_rules_universal A/B.

## Why this exists

B.2 (per-candidate value head, observational K=10 ship-delta labels)
went 0/32 vs bare pv_eta because the trace only emitted features for
*accepted* candidates. At inference the chooser asks the head about
candidates it never saw at training time, producing unconstrained
LightGBM extrapolation. **Axis closed under Rule 37.**

B.3 keeps the additive-head architecture but replaces the labels with
**CRN-paired advantage**:

```
A(s, a) = focal_margin(after action a, K) − focal_margin(after idle, K)
```

Both rollouts use pv_eta as the focal-seat policy from step 1 onward
AND as the opp policy throughout. Coverage = all top-N prerank
candidates per state (not just accepted). No selection bias.

## Pipeline (5 files, 9cf740f + 8b65f1b + be344eb)

1. `agents/baseline/_trace_hook.py` — `trace_prerank()` writes one
   JSONL row per scored prerank candidate (pristine `leaf_delta` +
   `cheap_delta`). Gated on `BASELINE_PRERANK_TRACE` env var.
2. `agents/baseline/chooser_trajectory.py` — 20-LOC insertion in the
   `score_candidate_v4` scoring loop, after `trace_solo()` and
   BEFORE any ML/VH score adjustments.
3. `scripts/probe_pveta_selfplay.py` — replay.jsonl extended with the
   full obs fields `env_from_obs()` needs (`comets`, `comet_planet_ids`,
   `initial_planets`, `angular_velocity`, `next_fleet_id`,
   `remainingOverageTime`) + per-tick `actions` taken by each seat.
4. `scripts/compute_crn_advantage.py` — NEW stage-2 driver. Per game:
   group prerank rows by (step, seat), keep top-N by cheap_delta,
   reconstruct env state via `env_from_obs`, run paired K-tick
   rollouts (idle + each top-N candidate), record `label =
   margin_action − margin_idle`. Features computed inline from
   recorded obs via `World.from_obs` + `WorldModel.from_world` +
   `lib.value_head_features.encode_features`.
5. `scripts/gen_b3_corpus.py` — orchestrator. Stage 1 → Stage 2.

## PI design choices (resolved 2026-05-30)

| Question | Choice |
|---|---|
| Top-N candidates per state | 5 |
| Rollout horizon K | 5 |
| Compute platform | local single-core overnight |
| Wallclock per pv_eta call | 100 ms |
| Focal-wins-only filter | OFF (first cut) |
| Feature: cheap_delta vs leaf_delta | **leaf_delta** (richer) |
| Substrate: fast_sim or env.clone+step | env.clone+step (Step 0 found fast_sim offered no benefit) |

## Smoke (4 games, workers=4)

| Stage | Wall |
|---|---|
| Stage 1 — selfplay (4 games) | **282 s** |
| Stage 2 — CRN labelling | **11,123 s (3.1 h)** |

Stage 2 was **5-10× slower** than the original B.3 plan estimated.
Per-game stage-2 cost extrapolates to ~38 h for 50 games on 4
workers — instead of the 6-9 h overnight I'd projected. The bottleneck
is pv_eta wallclock contention across 4 cores; sequential games would
likely be ~150 h. Future runs may need wallclock reduction
(BASELINE_WALLCLOCK_MS=50 for stage 2 rollouts) or a different
parallelization strategy.

### Corpus quality (`data/value_head/b3-smoke/corpus.jsonl`, 14,086 rows)

| Metric | Value |
|---|---|
| Rows | 14,086 |
| σ(label) | 19.14 (in [5, 500] sanity band ✓) |
| mean(label) | −1.68 (slight negative skew; more bad top-5 than good) |
| min / max | −1425 / +90 (one big end-game cascade outlier) |
| Zero labels | 85.6% (CRN cancellation at K=5) |
| Spearman(leaf_delta, label) FULL | +0.089 |
| **Spearman(leaf_delta, label) NON-ZERO** | **+0.244 (n=2030)** |

B.1's observational ρ on K=10 ship-delta was ~0.06. The CRN-paired
non-zero subset has **4× the signal**. The reframe worked at the
architecture level.

## Head training (`data/value_head/b3-smoke/value_head_model.txt`)

| Metric | Value |
|---|---|
| Train n / Val n | 10,401 / 3,685 (game-level split, val = 1 of 4 games) |
| Trees | 9 (early stopping) |
| Val σ(label) | 18.6 |
| Val RMSE | 18.56 |
| **Spearman ρ (scipy, val)** | **+0.174** (above 0.10 gate) |
| Spearman ρ (non-zero subset) | +0.238 |
| Walker parity vs Booster | 0.000e+00 |
| Prediction range | [−24.12, +0.39] |

**The head is one-sided.** It only predicts ≤0 — i.e. it demotes bad
candidates rather than promoting good ones. Of 79 predictions <−0.5,
68% had truly negative labels (mean actual = −9.91 ships). Works as
an additive correction term in pv_eta's chooser: bad candidates get
pushed to score < 0 and dropped by the existing `if score > 0`
filter.

### Trainer change (be344eb)

`_load_corpus` auto-appends `leaf_delta` to feature index 14 when
the row carries 14-d features + a top-level `leaf_delta` field. Keeps
B.2's 15-d schema while letting B.3 store the leaf separately for
analyzability.

σ(label) band widened from [200, 1500] to [5, 1500] (B.3 advantage
values are an order of magnitude smaller than B.2 absolute deltas).

### Known issue — `_spearman_correlation` over-reports with ties

The trainer's internal Spearman (lines 100-105 of
`scripts/train_value_head.py`) used `np.argsort(np.argsort(...))`
which assigns sequential ranks to tied values. On a label
distribution with 85% zeros, this inflated reported Spearman from the
true 0.174 to 0.595. **Fix candidate:** replace with
`scipy.stats.spearmanr(a, b)[0]` or use proper tie-handling
(`tied_rank` average). Not blocking — the gate still passed at
the true 0.174.

## Bundle + Rule 46 smoke

| Gate | Status |
|---|---|
| `scripts/bundle_pv_eta_vh.py --model b3-smoke/value_head_model.txt` | PASS (772 KB) |
| `pytest tests/test_bundle.py` | **10/10 PASS** in 49 s |
| `fast.py play submissions/baseline_pv_eta_vh_b3smoke.py` | PASS — 288 turns, p0 win |

### Timing risk

| Quantile | B.3 bundle | Bare pv_eta |
|---|---|---|
| p50 | 715 ms | ~140 ms |
| p95 | 816 ms | ~250 ms |
| max | **954 ms** (one-game) / **1178 ms** (panel) | ~700 ms |

The per-candidate pure-Python tree walker on the head dominates the
~500 ms overhead. p95 is safely under the 1000 ms cap but max ≥1000 ms
appears on dense states. Live submission would risk occasional turn
timeouts. Fix candidate: cache featurization across candidates
sharing src or tgt planets.

## A/B vs bare pv_eta (n=32, Rule 43b + Rule 45)

| Result | Value |
|---|---|
| Wins | **32 / 32 (100%)** |
| Wilson 95% lower bound | **0.893** |
| Wilson 95% upper bound | 1.000 |
| Verdict | **PASS** (Wlo ≥ 0.55) |
| Total elapsed | 1066.8 s |

## Multi-opponent panel (Rule 43a, default panel + champion h2h)

Total wall: 86.6 min (4 workers, ~22 min per pairing).

| Opponent | Wins | Wilson-lo | Verdict |
|---|---|---|---|
| **baseline_pv_eta_probe** (champion h2h) | 32/32 (100%) | **0.893** | PASS |
| v7_0 | 31/32 (96.9%) | 0.843 | PASS |
| v4_planner | 28/32 (87.5%) | 0.719 | PASS |
| v3.5.1 | 28/32 (87.5%) | 0.719 | PASS |

**Panel verdict: PASS** (worst Wilson-lo = 0.719, gate is 0.55).

Closes both Rule 43 (panel + champion) and Rule 45 (n ≥ 32 lift) for
the bare-pv_eta-class opponent set.

## Submission readiness as of 2026-05-30 ~19:30 UTC

| Gate | Status |
|---|---|
| Rule 43a (panel Wlo ≥ 0.55) | PASS |
| Rule 43b (champion h2h n ≥ 32, Wlo ≥ 0.50) | PASS |
| Rule 45 (n ≥ 32 for lift claim) | PASS |
| Rule 46 (bundle + parity smoke) | PASS |
| Rule 42 (cross-branch push coordination) | **NEEDS RE-CHECK** — Kaggle state has changed since the plan was authored (see below) |
| Timing safety margin to 1000 ms cap | **TIGHT — max=1178 ms once during panel** |

## Important Kaggle state shift since plan authorship

Refresh 2026-05-30 19:41 UTC: rolling pair has changed.

| Sub ID | Date (UTC) | Agent | μ |
|---|---|---|---|
| **53182323** (newest) | 2026-05-30 11:26 | **baseline_launch_rules_universal** | **1173.6** (new peak — above pv_eta's 1155 historical) |
| 53177486 (older half — would be evicted) | 2026-05-30 08:23 | baseline_redeploy_gangup | 1017.2 |

Both rolling pair entries are NEWER than the original plan's snapshot.
The current ladder ceiling is `launch_rules_universal` (1173.6), not
bare pv_eta. The B.3 head's panel data is against pv_eta-class
opponents, not against this stronger 1173.6 architecture. A/B vs
launch_rules_universal in progress (16 games, started 19:42).

## Open items

1. **launch_rules_universal A/B** (running 19:42). If the B.3 head holds
   up against this stronger opponent → submission gate cleared with
   higher confidence. If it regresses → the head is calibrated against
   pv_eta only and doesn't generalize.
2. **Timing fix.** Cached featurization per (src_id, tgt_id) keyed on
   prerank turn — would reduce max from 1178 ms to safely below the
   1000 ms cap. Not in current bundle.
3. **Spearman trainer fix.** `_spearman_correlation` tie-handling
   inflates reported metric. Cosmetic but misleading.
4. **Production stage 1+2.** 50 games would take ~38 h wall on 4
   workers. Current 14k-row smoke produced a head that already
   passes panel. Cost-benefit of scaling up is open.

## Files added / modified

- `agents/baseline/_trace_hook.py` (extended)
- `agents/baseline/chooser_trajectory.py` (extended)
- `scripts/probe_pveta_selfplay.py` (extended)
- `scripts/compute_crn_advantage.py` (new)
- `scripts/gen_b3_corpus.py` (new)
- `scripts/train_value_head.py` (extended — 14-d corpus schema)
- `data/value_head/b3-smoke/corpus.jsonl` (14,086 rows)
- `data/value_head/b3-smoke/value_head_model.txt`
- `submissions/baseline_pv_eta_vh_b3smoke.py` (772 KB bundle, not committed — generated artifact)

## Audit trail

| commit | date | summary |
|---|---|---|
| b245afe | 2026-05-29 | B.2 falsification + B.3 plan |
| 7c29ce9 | 2026-05-30 | Step 0 verification |
| 9cf740f | 2026-05-30 | Corpus pipeline (smoke running) |
| 8b65f1b | 2026-05-30 | Inline feature encoding at stage 2 |
| be344eb | 2026-05-30 | Trainer accepts 14-d schema |
