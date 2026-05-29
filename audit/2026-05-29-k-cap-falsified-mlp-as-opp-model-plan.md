# 2026-05-29 — opponent-model rate-cap falsified; pivot to MLP-as-opp-model

Branch: `claude/kaggle-submission-review-gZsCu` (this branch).
Session: single PM block, no submissions made.

This file is self-contained on purpose: it is BOTH the day's findings and
the next-session implementation plan. Context compaction must not split
them, so neither half points to the other by "see above."

---

## Part 1 — what shipped, what didn't

### The hypothesis under test

The rollout's opponent-policy predictor (`lib/opp_model.py::lite_greedy_policy`)
was suspected of over-firing. It launches from every owned planet whose
top return-on-investment target clears a threshold, every game tick.
That produces roughly 3-5 simulated launches per opponent tick. The
"fingerprint" data on real top-10 ladder players (from the strategy
analysis) puts realised opponent launches at ~1.3 per tick. So the
predictor over-shoots reality by roughly 3×.

PM3 (yesterday) ratified the pivot from the macro-layer null result to
"fix the opponent model first." Today's session implemented option (a) —
a top-K-by-return-on-investment cap per tick.

### What was built

- **Code change (committed as `4892a0e`):** new env var
  `BASELINE_OPP_MAX_LAUNCHES`. When set, `lite_greedy_policy` sorts
  candidate launches by descending ROI and keeps only the top K. K=0
  means "no cap" (current behaviour, byte-identical to pre-patch).
- **Tests:** 5 new unit tests in `tests/test_opp_model_max_launches.py`,
  11/11 green. 10/10 bundle tests still green.
- **Bundles:** `submissions/baseline_opp_launches_k{0,1,2,3}.py` built
  via `scripts/bundle_agent.py`. K=0 is the control (identical to
  current `baseline_leaf_pv_2p` shape); K=1/2/3 are the rate-limited
  variants.

### A/B results

Stage 1, vs `submissions/v4_planner.py` (weaker anchor):

| Variant | Result | Wilson 95% LB |
|---|---|---|
| K=0 (control) | 16/16 = 100% | 0.806 |
| K=1            | 15/16 = 93.8% | 0.717 |
| K=2            | 16/16 = 100% | 0.806 |
| K=3            | 15/16 = 93.8% | 0.717 |

All four hit the ceiling. The test is non-discriminating because the
current `baseline_leaf_pv_2p` stack already crushes `v4_planner`
regardless of opponent-model wiring. **The v4_planner anchor is dead
for this comparison axis** — note for next time.

Stage 2, vs `submissions/baseline_leaf_pv_2p.py` (live champion μ=1102.5):

| Variant | Result | Wilson 95% LB | Verdict |
|---|---|---|---|
| K=2 (joint Stage-1 leader) | 6/16 = 37.5% | 0.185 | FAIL (below gate 0.50) |

The rate-cap **regresses against the current live agent**. K=2 was not
submitted. Daily submission budget for 2026-05-29 UTC: **0 used.**

### Why the cap likely failed

The structural issue: this A/B is asymmetric in the opponent model.
Our K=2 chooser's rollout predicts the opponent will fire at most 2
launches per tick. The live opponent (no env var set) actually emits
3-5 per tick. Our chooser therefore under-predicts the threat density,
mis-sizes fleets, mis-times captures, and loses.

This is not purely an A/B artefact — the realised ladder population at
μ ≥ 1100 may genuinely fire closer to 3-5 launches per tick than the
1.3 historical fingerprint suggested. The fingerprint was top-10
historical play; today's ladder is a different distribution.

Net: the calibration target itself (≤ K launches per tick) is the
wrong knob, at least at the values the fingerprint implied.

### Per-axis falsification status (Rule 37)

- "Top-K-by-ROI cap" is ONE variant of the rate-limiting axis. Three
  K values were tested but only K=2 was tested against the meaningful
  opponent. By Rule 37, this is a single axis-failure, not 3+.
  However, the structural reason for failure (asymmetric chooser/opp
  models) is **shared** across all K values that don't equal the
  realised opponent's true rate — which we don't know.
- **Pivot rather than iterate K.** Bumping K to 4 or 5 walks the same
  axis with the same structural problem. Pivot to a smarter predictor
  (Part 2 below).

---

## Part 2 — pivot: the trained MLP from sub 53131296 can double as the opponent model

### Discovery

Kaggle submission history (today's session, via `kaggle competitions
submissions orbit-wars`):

> Sub 53131296, `baseline_validated.py`, submitted 2026-05-28 23:22 UTC
> from branch `claude/competition-objective-alignment-hqNVM`, public
> score μ=**1086.1**. Description: "3-MLP ensemble (seeds 42/100/7);
> 24-d konbu17 features + F2 combat_margin_at_arrival; threshold 0.30".

The sibling branch trained a 3-MLP ensemble that classifies any
candidate launch as `P(launch will succeed)`. The submission uses it
as a **shot validator on our side** — the chooser proposes emits as
usual and the MLP filters out any candidate with `P < 0.30`. Self-
reinforcement passes through.

The same network can be queried with the opponent's seat swapped in.
Used that way, it becomes a learned opponent model: the rollout opp
fires only candidate launches where the MLP says `P ≥ τ`. This is
strictly better than `lite_greedy_policy` because:

- **Trained on real outcomes.** Labels are "did the launching player
  still own the target 10 turns after arrival?" from 50 top-10 replays
  + 10 midpack. That is literally the calibration target we were
  guessing at with K.
- **Self-limiting.** No K to tune, no per-tick budget. The threshold τ
  controls aggressiveness, and τ is a single scalar on a probability.
- **Already bundled.** Weights ship inline as gzip+base64 in the
  existing submission. No new artefact to package.
- **Symmetric encoder.** `lib/shot_features.py::encode_shot_features`
  already takes a `focal_seat` parameter. Calling it with the opp's
  seat gives opp-side features for free.

### Substrate location (verified via `git ls-tree` on origin)

On branch `origin/claude/competition-objective-alignment-hqNVM`, the
MLP version was last clean at commit `4a8e4c0` ("bundle_validator.py
— wrapper-bundler for baseline_validated"). The state at that commit
matches sub 53131296.

| File | Role |
|---|---|
| `lib/shot_features.py` | 25-d feature encoder. Exports `FEATURE_DIM`, `encode_shot_features`, `target_owned_by`. |
| `lib/_validator_tree_walker.py` | Pure-Python forward pass. Exports `parse_booster_text`, `predict_proba`. **CAVEAT** — at commit 4a8e4c0 this may already be the LightGBM tree walker; the MLP forward may live elsewhere. Verify before copying. |
| `agents/baseline_validated/main.py` | Wrapper that filters our chooser's emits using the MLP. Contains the inline base64 weights blob. |
| `scripts/train_validator.py` | Training script (only needed if we re-train; we don't, for the first cut). |
| `scripts/embed_validator_weights.py` | Embeds trained weights as base64 in the agent file (only needed if we re-train). |
| `data/shot_validator/schema.json` | Feature spec, v2 = 25 dims. |
| `data/shot_validator/README.md` | Plain-English doc of the feature spec and labelling rule. |

The Phase-2-v2 (LightGBM) work landed AFTER 4a8e4c0 (`d2608e3`,
`04f44d1`). For our hot-swap, we want commit 4a8e4c0 era — the
**MLP** version, not the booster.

### Verified caveats

- Sub 53131296's live μ=1086.1 sits BELOW its base predecessors
  (`baseline_leaf_pv_2p` 1102.5, `baseline_pv_eta` 1157.2). The
  shot-validator-as-our-side-filter is not yet a clear win. The local
  A/B was 60.9% n=64 vs baseline, Wilson 95% [0.487, 0.719] —
  inconclusive by the Rule 43 0.55 gate.
- The validator's τ=0.30 is calibrated for **our-side rejection**
  (high recall, don't kill good shots). For opp-model use we want
  **high precision** (opp fires only when the MLP is confident).
  Different threshold, almost certainly higher. Start with τ = 0.5.
- The MLP encoder uses F2 (`combat_margin_at_arrival`), which requires
  `lib/world_model.py::WorldModel.predict_garrison_at`. That import
  chain must survive bundling. The sibling branch already shipped
  this, so the bundler handles it, but verify on our branch after
  cherry-picking.

---

## Part 3 — implementation plan (executable next session)

### Goal

Add a new opponent-policy tier in `lib/opp_model.py` that uses the
existing trained MLP from sub 53131296 to decide opp launches.
Default-off behind an env var. Single-knob threshold. A/B vs live
champion at n=32. Submit if Wilson 95% LB ≥ 0.50.

### Pre-flight (Rule 16 six-question check)

| # | Question | Pre-answer |
|---|---|---|
| Q1 | Already explored? | Top-K rate-cap (this session, FAIL). MLP-as-opp-model: NO, never tried. |
| Q2 | Rank-lock-vulnerable? | No — opt-in behind env var, default off. K=0-equivalent path preserved byte-identically. |
| Q3 | Standalone result prediction | +5-15pp vs live, p ≈ 0.3 chance of clean lift, p ≈ 0.4 chance of noise, p ≈ 0.3 chance of regression. |
| Q4 | Correlation to K-cap result | Weakly anti-correlated. K-cap failed on asymmetric opp model; learned predictor addresses that asymmetry by encoding *what the opp actually does*. |
| Q5 | Precedent | konbu17 (top-10 finisher last comp) used a similar per-shot validator for OWN-side filtering; +19pp panel winrate. No precedent for the OPP-model use specifically. |
| Q6 | **Training objective matches comp metric** | The MLP target ("did the launching player still own target 10 turns after arrival?") is a per-shot proxy for TrueSkill/Elo — they are correlated but not identical. ACCEPTABLE for the predictor role: we are not training, we are repurposing an existing classifier as a behaviour model. Q6 not blocking. |

### Steps

1. **Pull substrate from sibling branch.** From `origin/claude/competition-objective-alignment-hqNVM` at commit `4a8e4c0` (or whatever HEAD is iff the MLP layer is still intact — verify by inspecting the file, not by trusting the commit message):
   - Copy `lib/shot_features.py` to our branch.
   - Copy `lib/_validator_tree_walker.py` IF the MLP forward lives there. If `_validator_tree_walker.py` is the LightGBM walker only, find the MLP forward inside `agents/baseline_validated/main.py` and lift it to a new module (e.g. `lib/_validator_mlp.py`).
   - Copy `data/shot_validator/schema.json`.
   - Extract the inline base64 weights blob from `agents/baseline_validated/main.py` and store it in a new `lib/_validator_weights.py` so other modules can import it cleanly.
   - Do NOT cherry-pick the agent wrapper itself — we want the MLP for opp-model use, not own-side filtering. If we later layer the own-side filter on top, that is Step 5 below.

2. **Add a new opp-model tier.** Edit `lib/opp_model.py`:
   ```python
   def mlp_validated_policy(obs, threshold=0.5):
       """Tier 3 — query the trained validator MLP for each
       candidate (src→tgt) launch from the opponent's seat; fire
       only when P(success) ≥ threshold."""
       # 1. Enumerate every owned-by-opp source planet.
       # 2. For each, enumerate every plausible target
       #    (every other planet within env max eta).
       # 3. Build a 25-d feature vector via encode_shot_features
       #    with focal_seat=opp_seat.
       # 4. Stack and forward through the MLP ensemble (mean of
       #    3 sigmoids).
       # 5. Emit a launch action for every (src, tgt) with
       #    p_mean ≥ threshold.
       ...
   ```
   Wire `make_opp_policy(tier=3)` to return it. Read the threshold
   from env var `BASELINE_OPP_MLP_THRESHOLD` (default 0.5). Read
   tier selection from env var `BASELINE_OPP_MODEL` (`lite_greedy`
   default, `mlp` selects the new tier).

3. **Unit tests** (`tests/test_opp_model_mlp.py`):
   - Synthetic obs with one src, one tgt: encoder produces 25-d
     vector with correct value ranges; MLP forward returns a float
     in [0, 1].
   - Threshold gate: emits drop monotonically as threshold rises
     from 0.1 to 0.9.
   - Env var off: default policy is unchanged (byte-identical to
     `lite_greedy_policy` on a fixed-seed snapshot).
   - Bundle test: `pytest tests/test_bundle.py` stays green.

4. **Smoke + parity (Rule 2 two-tier smoke is for GPU; we are CPU):**
   - `python fast.py play submissions/<bundle>.py` runs one full
     game vs `v7_0` without crash or timeout. p95 turn-ms < 1000ms.

5. **A/B vs live champion (Rule 45 gate):**
   - `python fast.py eval submissions/<bundle>.py --vs submissions/baseline_leaf_pv_2p.py --max-seeds 32 --workers 4 --gate 0.50`
   - Wallclock estimate: ~50 min at workers=4 with no contention.
   - **Two threshold values to sweep:** τ=0.5 and τ=0.6. Run them
     sequentially (≤2 concurrent CPU jobs per Rule 31).

6. **If Wilson-lo ≥ 0.50:** ship at the next slot, respecting Rule 42
   (pre-submit cross-branch coordination — check `state/MULTI_BRANCH.md`
   push-claim board and rolling pair before submitting). Predicted μ
   band before pushing: 1080-1180 (wide, given the validator's μ=1086
   reference and the uncertain threshold transfer).

### Pitfalls catalogued in advance

- **Bundler import friction.** All cross-module imports must be
  single-line, not parenthesised multi-line — the bundler's per-line
  import stripper produces `IndentationError` at runtime otherwise.
  Friction tag: `bundler-modular-agent-namespace-access-breaks-bundle`.
- **Rule 47 (physics-primitive verification).** The encoder uses
  `predict_garrison_at`. Before any A/B against a strong opponent,
  run a single-game trace and confirm sun/OOB waste < 2%.
- **Rule 46 (bundle + parity smoke before submission).** Always run
  the full bundle test suite + a single `fast.py play` before
  pushing to Kaggle.
- **Rule 42 (cross-branch coordination).** Sibling branch is the
  one that owns sub 53131296. Submitting our own MLP variant will
  evict one of the rolling-pair submissions. Check that the evicted
  μ is BELOW our predicted μ before pushing.
- **Threshold transfer.** τ=0.30 was tuned for our-side rejection.
  For opp-model use, plan for τ in [0.4, 0.7]. Start at 0.5 and only
  go lower if opp produces too few launches per tick (visible in
  single-game traces — sanity check: opp should still emit at least
  occasionally; if it sits idle the whole game, threshold is too
  high).
- **3-MLP ensemble cost.** Three forward passes per query × ~5
  candidates per opp tick × ~30 rollout steps × ~30 root candidates
  ≈ 13,500 forward passes per move. At 25 → 32 → 16 → 1 ≈ 1000
  multiplies per pass, that's ~13M multiplies per move. At numpy
  speeds (~ns per multiply) that's ~30 ms per move. Inside the 1-second
  turn budget, but not negligible. If timing is tight, drop to one
  seed (lose ~0.5pp ensemble averaging — usually fine).

### Estimated cost

- Substrate pull and wiring: 1-2 hours.
- Tests + smoke: 30 min.
- A/B at n=32 (two thresholds, sequential): ~100 min wallclock.
- Total: one focused session, no submission used until the gate
  clears (Rule 1 — single-shot submits only).

### Acceptance criteria

| Gate | Threshold | Action |
|---|---|---|
| Bundle tests | 10/10 green | Required |
| Unit tests | 8/8+ green | Required |
| `fast.py play` smoke | runs to end, p95 < 1000 ms | Required |
| A/B vs live, n=32, τ=0.5 | Wilson 95% LB ≥ 0.50 | Sufficient to ship τ=0.5 |
| A/B vs live, n=32, τ=0.6 | Wilson 95% LB ≥ 0.50 | Sufficient to ship τ=0.6 |
| Neither threshold clears | n/a | Falsify the MLP-as-opp-model axis; pivot to per-planet cooldown OR garrison reserve OR a smarter feature set. |

---

## Quick references for next session (no scrolling needed)

- **Today's branch and commit:** `claude/kaggle-submission-review-gZsCu`,
  HEAD `4892a0e`.
- **Sibling branch carrying the MLP substrate:**
  `claude/competition-objective-alignment-hqNVM`, MLP commit `4a8e4c0`.
  Phase 2 v2 LightGBM lives after `d2608e3` — avoid for this build.
- **Current live rolling pair (per Kaggle history today):**
  - 53131296 `baseline_validated.py` μ=1086.1 (3-MLP shot validator)
  - 53117942 `baseline_leaf_pv_2p.py` μ=1102.5 (current target to beat)
- **Live peak still on ladder:** sub 53111837 `baseline_pv_eta.py`
  μ=1157.2. **Not in rolling pair** — already evicted. Cannot be
  reactivated.
- **Daily submission budget:** 5/day, today 0/5 used. Next session
  starts fresh per UTC midnight.
- **Submission slot strategy:** push only when A/B clears. If
  neither τ clears, do not push — the rolling-pair floor is too
  valuable to spend on a coin-flip submission (Rule 12 + Rule 42).
