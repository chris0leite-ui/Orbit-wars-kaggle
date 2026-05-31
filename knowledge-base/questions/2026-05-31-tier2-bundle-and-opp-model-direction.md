# 2026-05-31 — Open questions after Tier 2 falsification

1. **Which direction to test first — event-driven horizon or distilled opp
   model?** Both unblock heavy opp models in the chooser; the former is
   architecturally bigger, the latter cheaper to try. Lean: distilled opp
   model first (smaller swing, faster verdict).

2. **For distilled opp model: per-state features or per-emit features?**
   The 45-d `shot_features` encoder is per-emit (good for filter, awkward
   for "predict next action"). The 40-d `value_head_features` is per-state
   (good for whole-board snapshot). New feature set may be needed.

3. **How to pull top-leaderboard replay data from Kaggle?** PI flagged
   replays are available. Need to map the access path:
   `kaggle competitions submissions` returns metadata, but the actual
   per-tick observation/action stream may need a separate API or scrape.

4. **Should we rerun Tier 1 (`BASELINE_OPP_TIER=1`) at n=16 to confirm
   the structural-cost theory?** Predicted ~15-30% based on candidate
   validation counts (211/turn vs Tier 0's 1209). Killed by CPU
   contention today; cheap to rerun (15-20 min).

5. **Is the chooser's `if score > 0` filter (chooser_trajectory.py:978)
   too aggressive under heavy opp models?** Under Tier 2 the
   positive-rate per validated candidate is HIGHER than under Tier 0
   (24% vs 7%), but absolute count is lower. A "top-K by score
   regardless of sign" fallback could rescue more candidates when
   `scored` is small.
