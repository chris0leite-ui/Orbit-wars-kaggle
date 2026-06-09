# HANDOVER.md — next-session brief

_Refreshed 2026-06-02 by `claude/champion-ml-graft-majestic-storm`.
Read this first (Rule 15). Also read `state/MULTI_BRANCH.md` (live rolling pair / track registry) per Rule 44._

## Where we are

**Branch:** `claude/champion-ml-graft-majestic-storm` (cut from `origin/claude/champion-strategy-rules-00JzI` this session).

**Live rolling pair (pulled 15:04 UTC, 2026-06-02):**

| Sub | Agent | Settled μ | Role |
|---|---|---:|---|
| 53295205 | baseline_champion_tier2 | **PENDING** | newest — our submit this session |
| 53280733 | baseline_state_driven_k | 1153.6 | backstop |
| 53277693 | baseline_launch_rules_universal (evicted) | 1110.6 | historical peak agent, μ=1183.7 prior |
| 53265480 | **champ_adaptiveK_on** (evicted) | **1170.4** | **NEXT SESSION'S TARGET BASE** |

**Days remaining: 21** (deadline 2026-06-23 23:59 UTC). Daily submit budget: 5.

## Today's progress (load-bearing)

This session's unlock: **first end-to-end graft of hqNVM's ML stack onto the 00JzI champion base.** Two ML pieces were lifted off hqNVM where they had previously failed on the wrong foundation (`baseline_pv_eta` settled μ=526-688). Hypothesis tested today: ML wasn't the problem — the chooser substrate it sat on was. Grafted Tier-2 opp model onto `launch_rules_universal` config; submitted as `baseline_champion_tier2` (#53295205).

- **Files cherry-picked from hqNVM (8):** `lib/_validator_tree_walker.py`, `lib/opp_features_lite.py`, `lib/value_head_features.py`, `agents/baseline/_value_head.py`, two trained-model blobs, `scripts/train_value_head.py`, `scripts/bundle_pv_eta_vh_dist_composite.py`.
- **In-place patches:** `lib/opp_model.py` Tier-2 stub replaced with distilled `trained_logreg_policy` (lite encoder, ~1.6 ms/call); `agents/baseline/chooser_trajectory.py` got Block A (prerank VH featurization) + Block B solo path; `scripts/bundle_agent.py` `DEFAULT_LIB_ORDER` extended; composite bundler now skips `shot_features` (not present on 00JzI, lite encoder bypasses it).
- **Local A/B (Rule 43 probe):** champion+Tier-2 vs un-ported champion at n=32 = **21/32 = 65.6%**, Wilson [0.483, 0.796], 1770 s. **Wilson-lo 0.483 short of Rule 43b 0.50 — submitted on PI override.**
- **Rule 46 GREEN:** bundle ✓, `pytest tests/test_bundle.py` 15/15 ✓, full game 275 steps WIN vs v7_0 max 878 ms ✓.
- **TLE check:** 1/~8000 turns at 1308 ms in the n=32 panel (0.012% rate, ~0.06 dropped moves/Kaggle game). Cold-load 0.9 ms opp + 21 ms VH — not the culprit. Per-turn p95 = 959 ms in eval, 855 ms single-game — sustained near-cap but rare overrun. NOT systemic.
- **Value head shipped at λ=0 (dormant)** — single-variable A/B for Tier-2 attribution. The shipped value-head model was trained on `baseline_pv_eta` self-play and would systematically under-rate the joint-aggressive candidates the champion chooser fires; activating it without retraining was deliberately deferred.
- **Joint-path VH patch deferred** — `vh_featurize_prerank` keys on solo prerank rows; joint legs would miss the cache. Documented in code comments at `agents/baseline/chooser_trajectory.py:1140-1147`. No-op at VH_LAMBDA=0; **mandatory fix before any VH-enabled submit.**

Commits this session (top of branch):
- `ac07376` claim(rule42): submit baseline_champion_tier2 (Tier-2 on champion base)
- `a1ee64d` graft hqNVM ML stack onto 00JzI champion base (Tier-2 wrapper, VH off)
- `d3168ba` (on hqNVM) plugins/ml: portable ML toolkit for non-ML target branches

## Falsified-or-dead (this branch)

Nothing falsified yet — we're awaiting Tier-2 settle (24-48 h). Decision tree for next session:

- **Settled μ ≥ 1130:** Tier-2 transfer to champion base CONFIRMED. Green light for value-head retrain + ship.
- **Settled μ 1050-1130:** ambiguous — Tier-2 lift is marginal. Consider re-A/B at n=64 before any VH work, OR skip VH and revert to plain adaptive-K as base.
- **Settled μ < 1050:** Tier-2 does NOT transfer to champion base. ML-on-champion hypothesis falsified. Pivot away from ML; revisit other unlock candidates from the 3-features re-test plan inherited from 00JzI's prior handover.

## NEXT-SESSION PLAN — PI-directed: value head on the adaptive-K champion

**The PI direction (2026-06-02 15:10 UTC, this session's wrap):**

> "We want the champion with the adaptive predictive horizon K. We want to build on that. And then I want to really retrain the value head and use the value head for this champion. This will be next step."

**Why adaptive-K specifically:** `champ_adaptiveK_on` settled at **μ=1170.4** (Kaggle sub #53265480, branch champion-strategy-rules-00JzI commit 9985e98) — the strongest recent 00JzI agent, +60 μ over `launch_rules_universal` (the historical-peak config we grafted Tier-2 onto this session). Adaptive-K opens K=20 (wider opening capture horizon) and decays to K=10 by step 30. Single env-var flip: `BASELINE_ADAPTIVE_K=1` (+ defaults `BASELINE_ADAPTIVE_K_OPEN=20`, `BASELINE_ADAPTIVE_K_TSETTLE=30`).

**Execution order (one lever at a time — Rule 37):**

0. **Pull `kaggle competitions submissions orbit-wars` first.** Read the settled μ of `baseline_champion_tier2` (#53295205). Branch on the decision tree above.

1. **Build `agents/baseline_champion_adaptiveK_tier2/main.py`** (the new base). Same env block as `baseline_champion_tier2` PLUS `BASELINE_ADAPTIVE_K=1`. Re-bundle via `scripts/bundle_pv_eta_vh_dist_composite.py --wrapper <abs-path>`. Re-run Rule 46 + an n=32 directional probe vs `submissions/baseline_launch_rules_universal_local.py`. Expect μ ≥ 1150 if Tier-2 still transfers under adaptive-K. **If probe shows directional regression vs adaptiveK-without-Tier-2, the ML graft is config-specific; escalate.**

2. **Retrain the value head on adaptive-K self-play** (~3-4 h CPU). Recipe at `scripts/train_value_head.py`:
   ```
   python scripts/train_value_head.py --collect-only \
     --agent agents/baseline_champion_adaptiveK_tier2 \
     --opponent agents/baseline_champion_adaptiveK_tier2 \
     --games 2000 --seeds-per-pair 4 \
     --out data/value_head/adaptiveK_corpus.jsonl
   python scripts/train_value_head.py \
     --corpus data/value_head/adaptiveK_corpus.jsonl --K 10 \
     --num-leaves 31 --num-iterations 200 \
     --out data/value_head/value_head_model.txt
   ```
   The encoder (`lib/value_head_features.py`) does NOT need retraining — only the regressor.

3. **Fix the joint-path VH featurization BEFORE any VH-enabled submit.** Currently `vh_featurize_prerank` (in `agents/baseline/_value_head.py:107`) keys on solo prerank rows. Joints at `chooser_trajectory.py:1250` and `:1414` miss the cache → VH biases away from 00JzI's defining attack class. Two options:
   - **(a) Add joint enumeration to `vh_featurize_prerank`** — featurize each joint pair at prerank time with averaged or summed features per leg. Larger change but semantically correct.
   - **(b) Compute on-the-fly VH for joints inside `score_candidate_v4_joint`** — smaller code change but pays the encode cost per joint.
   Either way, document the choice in the chooser comments and confirm with a parity smoke at VH_LAMBDA=0 (must be byte-equivalent to no VH).

4. **Build `agents/baseline_champion_adaptiveK_ml_vh/main.py`** (adaptive-K + Tier-2 + retrained VH). Add `BASELINE_VH_LAMBDA=1.0`. Re-bundle (the composite bundler patches both blobs).

5. **Full Rule 43 gate before submit:**
   - `fast.py eval <agent> --vs-panel default --require-h2h <rolling_champion>` — Wilson-lo ≥ 0.55 per opponent.
   - `fast.py eval <agent> --vs <rolling_champion> --geometry-panel` at n ≥ 32, Wilson-lo ≥ 0.50.
   - Rule 46 (bundle + test_bundle 15/15 + full game) clean.
   - Rule 42 push-claim row in `state/MULTI_BRANCH.md`.

6. **Submit only if predicted μ-gain > σ over the current rolling-pair-floor.** Pull `kaggle competitions submissions orbit-wars` immediately before push to confirm what gets evicted.

**Estimated total wall-clock:** ~5-6 hours (3-4 h retrain dominates).

**Hidden-in-plain-sight items I'd also check, but secondary:**
- **How often does Tier-2 disagree with the lite-greedy fallback?** If rarely (<10% of calls), most of the +14 μ "Tier-2 lift" is statistical noise. Cheap to instrument; should have happened before today's submit. 5-min check via a counter inside `trained_logreg_policy`.
- **Bundle size pass** — the legacy slow-path inside `trained_logreg_policy` (when `_DIST_USE_LITE_ENCODER=False`) is dead code in production. Trimming saves ~5 KB and slightly reduces parse-time; do this if VH submit timing is tight.
- **Sweep `BASELINE_OPP_FILTER_THRESHOLD`** (currently 0.15) — quick A/B at 0.10/0.20/0.25/0.30 to find the operating point. Hand-picked on hqNVM; might not be optimal on champion base.

## Pointers (new this session)

- `knowledge-base/thoughts/2026-06-02-ml-graft-on-champion-base-hypothesis.md` — design rationale, alternatives ruled out, Plan-agent critiques resolved.
- `state/MULTI_BRANCH.md` push-claim row dated 2026-06-02 14:55 — full Rule 42 disclosure for the Tier-2 submit.
- `submissions/baseline_champion_tier2.py` (976 KB) — the bundle as shipped.
- `agents/baseline_champion_tier2/` — wrapper source for re-bundling / reference.
- `scripts/bundle_pv_eta_vh_dist_composite.py` — the two-blob bundler for any future ML-composite agents on this branch (`shot_features` requirement removed; works on 00JzI).
- `/root/.claude/plans/go-majestic-storm.md` — the approved plan file from session start; contains the full execution sequence + risk register.
