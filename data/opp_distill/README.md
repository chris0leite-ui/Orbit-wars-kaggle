# Opp-distill corpus + booster (Tier 2 v2)

Distilled-ladder opponent action predictor. Replaces the falsified
filter-based Tier 2 (see `audit/2026-05-31-postmortem-tier2-falsification.md`,
Rule 37 axis closed).

## Files

| File | Purpose | Tracked in repo? |
|---|---|---|
| `distill_booster.txt` | LightGBM `model_to_string()` dump of the trained binary classifier; loaded by `lib.opp_model.trained_logreg_policy` at inference time via the pure-Python `lib._validator_tree_walker`. | YES |
| `distill_booster.meta.json` | Threshold, pos_rate, val metrics. | YES |
| `manifest.json` | Source day(s) + episode IDs + decoder knobs (top_k, min_ships) + corpus summary. Pointer for the private Kaggle dataset. | YES |
| `labels.jsonl` | Decoded (45-d feature, binary label) rows. Large (~60 MB at 1 day). | NO — gitignored. Lives in private Kaggle dataset `chris0leite/orbit-wars-opp-distill-corpus`. |
| `labels.summary.json` | Per-corpus aggregate (n_rows, pos_rate, etc). | NO — gitignored. |

## Reproducing

```bash
# 1. Download top-rated 2P episodes (~4 GB → /tmp/ow_replays)
python scripts/download_ladder_replays.py --date 2026-05-30 --target 940

# 2. Decode replays → labels.jsonl (~60 MB)
python scripts/decode_replays_to_labels.py --workers 4

# 3. Push compact corpus to private Kaggle dataset for persistence
kaggle datasets create -p data/opp_distill/ --dir-mode zip   # first time
# kaggle datasets version -p data/opp_distill/ -m "..." --dir-mode zip   # updates

# 4. Train LightGBM booster (~5 min CPU)
python scripts/train_opp_distill.py

# 5. Bench (verify ≤1 ms median per call)
python scripts/bench_opp_policy.py --tier 2 --n 200

# 6. Bundle the wrapper agent
python scripts/bundle_pv_eta_vh_dist.py
```

## Architecture

See module docstring in `lib/opp_model.py` (function `trained_logreg_policy`),
the plan in `/root/.claude/plans/wiggly-swimming-glacier.md`, and the
postmortem of the falsified design in
`audit/2026-05-31-postmortem-tier2-falsification.md`.
