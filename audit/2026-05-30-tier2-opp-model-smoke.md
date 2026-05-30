# Tier 2 opp-emit predictor — implementation + smoke (2026-05-30 PM2)

## Context

B.3 (CRN-paired advantage value head) shipped clean but A/B lift was
marginal (18/32 = 56.2 %, Wilson-lo 0.393 vs `launch_rules_universal`;
fails Rule 43b). PI decision 2026-05-30 PM: HOLD the B.3 bundle,
advance to Tier 2 opp model as the next lift source.

Hypothesis: pv_eta's chooser scores candidates inside `fast_sim`
rollouts whose opp policy is a cheap rule-base (`lite_greedy_policy`
default, `top_tier_mirror_policy` if `BASELINE_OPP_TIER=1`). Replacing
the rule-base with a learned opp policy should make the chooser's leaf
state a closer match to what we see on the live ladder.

Plan: `/root/.claude/plans/you-are-a-machine-snoopy-russell.md`.

## What landed this session

### Design reframe vs. handover

Handover sketched modifying `lib/world_model.py:predict_garrison_at`.
Investigation found that function is only called from
`score_candidate_static` (legacy v2) and the counterfactual at
`chooser_trajectory.py:331` — NOT from the active v4 leaf path. The
active v4 scorer reaches the opp via `opp_actions_for_snap` →
`_select_opp_policy` → `lib/opp_model.py`. `opp_model.py:128` already
had a `trained_logreg_policy` stub literally reserved for this exact
work (docstring named `data/shot_validator/schema.json`).

PI-ratified design: plug Tier 2 into `lib/opp_model.py:trained_logreg_policy`
as a **filter on Tier 1 candidates** (PM5 booster as gate, threshold 0.30,
self-reinforce passes through unfiltered per konbu17 design). Three
alternative shapes considered + rejected (argmax replacement: B.2
selection-bias precedent; score-rescaled rank: Reframe-A failure
precedent).

### Code changes

| File | Change |
|---|---|
| `lib/opp_model.py:124-243` | Implement `trained_logreg_policy` — lazy-load shot-validator booster (gzip+base64 blob or disk fallback), enumerate Tier-1 candidates, encode 45-d features per emit, score with `predict_proba`, drop sub-threshold. Self-reinforce pass-through (konbu17). Falls back to Tier 1 on any failure. |
| `agents/baseline/chooser.py:22,44-60` | Import `trained_logreg_policy`; add `BASELINE_OPP_TIER=2` branch in `_select_opp_policy`. |
| `agents/baseline/chooser.py:124-133` | **Bug-fix (separable from Tier 2):** `affordable_validate_cap` was probing per-step cost with EMPTY actions, missing the opp-policy cost. With Tier 1 (~5 ms/call) or Tier 2 (~6 ms/call) the cap was undersized by ~10×, blowing the 1000 ms env cap. Probe now uses `opp_actions_for_snap(probe, ...)` to capture real cost. With Tier 0 (lite_greedy ~0.01 ms/call) the probe is unchanged → no behavior change for current rolling-pair. |
| `agents/baseline_pv_eta_vh_opp/main.py` | NEW. Wrapper preamble: pv_eta foundation + `BASELINE_OPP_TIER=2` + `BASELINE_VH_LAMBDA=0` (clean Tier-2 attribution, no head mixing) + threshold 0.30 + kinematic-table OFF. |
| `scripts/bundle_pv_eta_vh_opp.py` | NEW. Clone of `bundle_pv_eta_vh.py` patching `_OPP_BOOSTER_B64` instead of `_VH_MODEL_B64`. |

### Pre-A/B latency bench

Per-call cost measured on a step-40 obs (after env reset + 40 idle
steps), 30 reps × 2 seats each:

| Tier | Policy | Cost | Notes |
|---|---|---:|---|
| 0 | `lite_greedy_policy` | 0.01 ms | obs-only ROI greedy; default in chooser |
| 1 | `top_tier_mirror_policy` | 5.02 ms | World+WorldModel+missions+settle+realize |
| 2 | `trained_logreg_policy` | 5.88 ms | Tier 1 + 45-d featurize + booster predict_proba |

Tier-2 inference adds only ~17 % over Tier 1 (the heavy cost is the
World/WorldModel rebuild + mission proposers, shared with Tier 1).
Tier 2 is ~600× slower than Tier 0 — load-bearing for chooser cap.

### Single-game smoke results

| Run | Wallclock budget | Probe fix | Outcome | p50 turn-ms | p95 | max |
|---|---:|:---:|---|---:|---:|---:|
| Pre-fix, seed=0, BASELINE_WALLCLOCK_MS=50 | 50 ms | no | p1_win (Tier 2 lost) | 172 | 535 | 922 |
| Post-fix, seed=0, default 1000 ms | 1000 ms | yes | _running_ | — | — | — |

Pre-fix turn cost averaged 3.4× the wallclock budget — confirms the
chooser cap was undersized with Tier 2 active. p95=535 ms, max=922 ms
at a 50 ms budget meant the chooser was validating far more candidates
than budget allowed. Post-fix should normalise (cap shrinks with the
correct per-step probe).

### Pre-existing test_bundle.py environment failure (flagged)

`pytest tests/test_bundle.py` errors with `ModuleNotFoundError: No
module named 'kaggle_environments'` at line 1075 of the bundled file.
Reproduced on HEAD with all my changes stashed — pre-existing, not a
Tier-2 regression. The bundler's own internal `_smoke_import` step DOES
load `kaggle_environments` correctly (smoke import OK on the new
bundle). Rule 46 in spirit verified via the bundler's own smoke; the
pytest harness needs a separate friction fix. **Not blocking Tier-2 A/B.**

## What's next (next agent / session)

1. **Confirm post-fix timing.** Wait for the in-flight single-game
   smoke at default 1000 ms wallclock. p95 < 1000 ms required for
   submit viability.
2. **A/B vs `launch_rules_universal` at n=32** (Rule 45, gate 0.50):
   ```
   python fast.py eval submissions/baseline_pv_eta_vh_opp.py \
       --vs submissions/baseline_launch_rules_universal_local.py \
       --max-seeds 32 --workers 8 --gate 0.50
   ```
3. **A/B multi-opponent panel (Rule 43, gate 0.55)** — only if step 2
   clears:
   ```
   python fast.py eval submissions/baseline_pv_eta_vh_opp.py \
       --vs-panel default \
       --require-h2h submissions/baseline_launch_rules_universal_local.py \
       --max-seeds 32 --workers 8 --gate 0.55
   ```
4. **If both clear** → Rule 42 pre-submit checklist + PI sign-off.
5. **If h2h fails** → diagnose latency first (verify cap didn't shrink
   below the chooser's useful floor of 8 candidates per turn), then
   threshold sweep (0.20 / 0.30 / 0.40), then retrain corpus on B.3-style
   opp-specific replays if all thresholds fail.

## Carry-forward artifacts

| File | Status |
|---|---|
| `submissions/baseline_pv_eta_vh_opp.py` | 981 KB Tier-2 bundle, smoke import OK; A/B pending |
| `lib/opp_model.py` | Tier 2 implemented; falls back to Tier 1 if booster fails to load |
| `agents/baseline/chooser.py` | `affordable_validate_cap` probe fix |
| `data/shot_validator/validator_booster.txt` | PM5 LightGBM booster (val_acc 0.83), reused as-is |
