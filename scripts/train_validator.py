"""Train a single LightGBM Booster shot validator (Phase 2 v2).

Loads `data/shot_validator/labels.jsonl` (produced by
`scripts/gen_validator_corpus.py`), does a GAME-LEVEL 80/20 split (row-
level split leaks ~15pp val acc per konbu17's notebook), trains one
binary-objective Booster with `scale_pos_weight=(1-p)/p`, and writes
the model text to `data/shot_validator/validator_booster.txt`.

Replaces the 3-MLP ensemble that shipped in PM5. Pure-Python inference
goes through `lib._validator_tree_walker.predict_proba`; the walker is
parity-tested against `Booster.predict(raw_score=True)` to ~1e-6 in
`tests/test_validator_smoke.py`.

Pos_rate calibration is the load-bearing decision; aborts if pos_rate
is outside [0.40, 0.85].

Usage:
    python -m scripts.train_validator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_LABELS = REPO / "data" / "shot_validator" / "labels.jsonl"
DEFAULT_BOOSTER = REPO / "data" / "shot_validator" / "validator_booster.txt"

THRESHOLD = 0.30
FEATURE_DIM = 45

# LightGBM hyperparams — tuned conservatively for ~16-20k labeled rows.
# Phase 2 v2 plan target: val_acc >= 0.85, Brier <= 0.12.
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 30
LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "binary_error"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.0,
    "lambda_l2": 0.1,
    "verbose": -1,
    "deterministic": True,
}


def _load_corpus(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats, labels, game_ids = [], [], []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            feats.append(r["features"])
            labels.append(r["label"])
            game_ids.append(r["game_id"])
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, game_ids


def _game_level_split(
    game_ids: list[str], val_frac: float, rng: np.random.Generator,
) -> np.ndarray:
    """True=train, False=val. Splits at the game level (all examples
    from one game go to one side)."""
    unique = sorted(set(game_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_games = set(unique[:n_val])
    return np.asarray([gid not in val_games for gid in game_ids])


def main(argv=None) -> int:
    import lightgbm as lgb  # heavy import; keep inside main

    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_BOOSTER))
    p.add_argument("--val-frac", type=float, default=0.20)
    args = p.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f"ERROR: labels not found: {labels_path}", file=sys.stderr)
        return 1
    X, y, game_ids = _load_corpus(labels_path)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        print(f"ERROR: expected (_,{FEATURE_DIM}) features, got {X.shape}",
              file=sys.stderr)
        return 1

    pos_rate = float(y.mean())
    print(f"corpus: n={len(y)}  pos_rate={pos_rate:.3f}  "
          f"unique_games={len(set(game_ids))}")
    if not (0.40 <= pos_rate <= 0.85):
        print(f"ERROR: pos_rate {pos_rate:.3f} outside healthy [0.40, 0.85] — "
              f"adjust opponent mix in gen step before training",
              file=sys.stderr)
        return 2

    scale_pos_weight = (1 - pos_rate) / pos_rate
    print(f"scale_pos_weight = (1-{pos_rate:.3f})/{pos_rate:.3f} = "
          f"{scale_pos_weight:.3f}")

    rng_split = np.random.default_rng(0)
    train_mask = _game_level_split(game_ids, args.val_frac, rng_split)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[~train_mask], y[~train_mask]
    print(f"split: train_n={len(y_tr)}  val_n={len(y_va)}  "
          f"(val_games={args.val_frac:.0%}, GAME-LEVEL, not row-level)")

    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    ds_train = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    ds_val = lgb.Dataset(X_va, label=y_va, reference=ds_train,
                         free_raw_data=False)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    bst = lgb.train(
        params, ds_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    best_iter = bst.best_iteration
    print(f"\nbest_iteration = {best_iter}")

    # Val metrics at the deployment threshold
    pred_va = bst.predict(X_va, num_iteration=best_iter)
    pred_va_bin = (pred_va >= 0.5).astype(np.float32)
    acc = float((pred_va_bin == y_va).mean())
    brier = float(((pred_va - y_va) ** 2).mean())
    pred_thr = (pred_va >= THRESHOLD).astype(np.float32)
    acc_thr = float((pred_thr == y_va).mean())
    tp = int(((pred_thr == 1) & (y_va == 1)).sum())
    fp = int(((pred_thr == 1) & (y_va == 0)).sum())
    tn = int(((pred_thr == 0) & (y_va == 0)).sum())
    fn = int(((pred_thr == 0) & (y_va == 1)).sum())
    print(f"\n=== val ===")
    print(f"  val_acc@0.5={acc:.3f}  Brier={brier:.4f}")
    print(f"  val_acc@thr={THRESHOLD}={acc_thr:.3f}  "
          f"TP={tp} FP={fp} TN={tn} FN={fn}")
    if tp + fp > 0:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  precision@thr={prec:.3f}  recall@thr={rec:.3f}")

    # Parity check the walker BEFORE we save — catches any silent format
    # change in the Booster text that the walker hasn't been updated for.
    from lib._validator_tree_walker import (
        parse_booster_text,
        predict_proba as walker_proba,
    )
    text = bst.model_to_string(num_iteration=best_iter)
    parsed = parse_booster_text(text)
    pred_walker = walker_proba(parsed, X_va[:200])
    pred_booster_200 = bst.predict(X_va[:200], num_iteration=best_iter)
    max_diff = float(np.max(np.abs(pred_walker - pred_booster_200)))
    print(f"\nwalker parity: max abs diff vs Booster on 200 val rows = "
          f"{max_diff:.3e}")
    if max_diff > 1e-5:
        print(f"ERROR: walker parity exceeds 1e-5 — refusing to save",
              file=sys.stderr)
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\nwrote booster -> {out_path}  ({len(text):,} chars, "
          f"{len(parsed.trees)} trees)")

    # Also save a small metadata sidecar — threshold + pos_rate the agent
    # will need at inference time.
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "threshold": THRESHOLD,
        "pos_rate": pos_rate,
        "best_iteration": int(best_iter),
        "val_acc_at_half": acc,
        "val_brier": brier,
        "feature_dim": FEATURE_DIM,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote meta    -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
