# ML next-steps — locked in for later sessions

**Date locked:** 2026-06-05.
**Source conversation:** session at sub 53384340 (μ=947.7 settled, regression vs predicted ~1310).
**Status:** captured per PI instruction "lock these ideas into next steps. We will address them in later sessions." Not to be worked on this session.

## Existing ML infrastructure (already trained, NOT wired into producer_plus)

These models exist in `data/` but were built for the previous `baseline_adaptive_k` strategy track. They are NOT integrated into our current `producer_plus_multi_opp_def` agent. Each weak on its current metrics; would need retraining on current-strategy self-play data.

- `data/value_head/` — gradient-booster value head (LightGBM-style). Current metrics: regression_l1, R² = 0.046, RMSE 420.7, MAE 214.7, spearman_rho 0.39, feature_dim 15. R² = 4.6 % means it explains very little outcome variance. Probably trained for a different state representation.
- `data/value_head_distill/` — distilled version of the above.
- `data/opp_distill/distill_booster_lite` — opponent action distillation. Current metrics: 30 features, only 4 boosting iterations, val_acc_at_half = 0.95 but val_precision_at_thr = 0 and val_recall_at_thr = 0 → the model predicts "no launch" for everything at the chosen threshold. Effectively untrained.
- `data/shot_validator/` — predicts whether a launch will succeed. Metrics not inspected this session.

## Where ML helps (ranked by ROI)

### 1. Learned value function past the planner's horizon (highest ROI)
Our scorer simulates exactly for 18 ticks and assumes "the game is over" after that. Real games last 100-400 turns. The planner makes 200-turn decisions from an 18-turn lookahead. Canonical fix: learned value head taking the post-horizon `garrison_status` projection + side-info, predicting expected final `competitive_score`. Train via self-play. Add as one term to the existing leaf score. Realistic lift: 30-80 μ. **Directly addresses the "myopic greedy" weakness identified in the architecture review.**

### 2. Learned opponent policy (medium ROI)
Replaces `predict_opp_launches_via_mirror` (currently runs Producer's planner from opp's seat, ~20 ms/turn). A small classifier — given opp's observable state, predict their launches — would be faster AND more accurate against non-Producer opponents (the ladder is diverse). Compute saved: ~15 ms/turn. Realistic lift: 10-30 μ.

### 3. Learned defensive urgency (small but principled)
Current urgency = `production × time_remaining + ships_at_planet`. A learned urgency function trained on "which planets did good players prioritise defending, what was the outcome?" lets us pick the right 4-8 defensive targets per turn instead of cheap-heuristic ones. Modest lift: 5-15 μ. Easy A/B.

### 4. Direct action chooser / policy network (highest ceiling, highest risk)
Replace greedy entirely with a learned policy that picks from the candidate set. Throws out a lot of engineered structure for a black box. Only after 1-3 are exhausted.

## Where ML does NOT help
- Exact combat simulation (`sparse_launch_flow_delta`'s arrival math + survivor rule): correct by construction; replacing with NN would add error.
- Compute speedups: we run at p50 ~ 50 ms on a 1000 ms budget. No point compressing.
- Greedy mutex and candidate enumeration: deterministic and cheap; ML adds no value.

## Hard constraints for any ML in this competition
- **Submission is one Python file.** Models inline as base64 weights. Gradient boosters fit easily (10-100 KB). Tiny MLPs (<100 K params, <1 MB) fit. Large NNs don't.
- **CPU-only inference at submission time.** No GPU on the Kaggle eval. Caps model size hard. Tabular gradient boosters (LightGBM / XGBoost) are the natural fit.
- **1000 ms TLE cap.** Per-turn inference budget if we want 200 ms headroom: ~100 small-model evaluations × small model = fine. A few inference calls per candidate is fine.
- **Training data from self-play.** Each game is ~100-400 turns at ~50 ms = 10-30 seconds. 10K training games = 1-3 hours. Feasible.

## Recommended sequence (later sessions)
1. **Wait for sub 53384340 to fully settle (~24 h).** It started at μ=947.7 -- understand the regression first before adding more mechanism.
2. **Retrain the value head on current-strategy self-play data.** Use the same gradient-booster pipeline as `data/value_head/` but with the current `garrison_status` + side-info feature set. Target: R² > 0.3 on held-out games.
3. **Integrate as a leaf-evaluation term** in `competitive_score` — additive, weighted, gated behind an env knob so OFF is bit-identical.
4. **n=32 A/B vs the no-value-head sibling.** Standard gate.
5. **Only if (2-4) lifts:** consider the opponent policy distillation (#2) next.

## Why this matters
The biggest remaining weakness identified in the architecture review is the planner's 18-tick horizon vs 200-tick game length. A trained value head specifically addresses that gap. It's the right ML bet for this codebase BECAUSE the existing infrastructure (gradient-booster pipeline) is the natural fit for a single-file submission, and the feature extraction layer (`garrison_status`) is exactly what a value head wants as input.

ML is not a magic lift here -- the bigger wins this session came from modelling correctness (opp-mirror + opp-aware defence). But a value head is the next-logical extension of "make the scorer see further," which is the same family of improvements.
