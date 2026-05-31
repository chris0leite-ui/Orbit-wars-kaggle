"""Train a LightGBM binary classifier for the distilled-ladder opp policy.

Loads `data/opp_distill/labels.jsonl` (produced by
`scripts/decode_replays_to_labels.py`), uses the `split` field already in
the data for train/val partition (already game-disjoint), trains one
binary-objective Booster with `is_unbalance=True`, writes the model text to
`data/opp_distill/distill_booster.txt`.

Differences vs `train_validator.py`:
  - Schema: rows have keys `feat` (45 floats), `label`, `episode`, `split`
    (not `features` / `label` / `game_id`).
  - Split: already in the data; no resplit needed.
  - pos_rate gate is loose: real opp emits are sparse (~0.5–2%) because we
    enumerate all (src, tgt) candidates including idle-turn ones. Gate at
    [0.002, 0.50].
  - Uses `is_unbalance=True` for class weighting (handles extreme imbalance).

Output:
  - `distill_booster.txt` — LightGBM `model_to_string()` dump.
  - `distill_booster.meta.json` — threshold, pos_rate, val metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_LABELS = REPO / "data" / "opp_distill" / "labels.jsonl"
DEFAULT_BOOSTER = REPO / "data" / "opp_distill" / "distill_booster.txt"

THRESHOLD = 0.30
FEATURE_DIM = 45

# Lite-mode slice: 30 cheap features from the 45-d corpus (matches
# lib.opp_features_lite.LITE_KEEP_INDICES). Drops:
# - 11 WorldModel-dependent features (F2/F3/F4/F6/F8)
# - F9 src threat (37, 38), post-capture geom (41, 42) — also need WM.
LITE_FEATURE_DIM = 30

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
    "min_data_in_leaf": 50,
    "lambda_l1": 0.0,
    "lambda_l2": 0.1,
    "is_unbalance": True,
    "verbose": -1,
    "deterministic": True,
}


def _load_corpus(
    path: Path, neg_per_pos: float | None = None, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load JSONL rows; optionally reservoir-sample negatives to bound memory.

    If `neg_per_pos` is set, walks the file once, keeps ALL positives, and
    reservoir-samples enough negatives to hit the target ratio. Avoids
    loading 12M rows into memory when only ~60k are positive.

    Reservoir sampling is split-aware (separate reservoirs for train/val) so
    val pos_rate matches train pos_rate after sampling.
    """
    if neg_per_pos is None:
        feats, labels, episodes, splits = [], [], [], []
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                feats.append(r["feat"])
                labels.append(r["label"])
                episodes.append(r["episode"])
                splits.append(r["split"])
        X = np.asarray(feats, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        return X, y, episodes, splits

    # Two-pass approach: count positives per split first, then size the
    # negative reservoir.
    n_pos_per_split: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["label"]) == 1:
                n_pos_per_split[r["split"]] = n_pos_per_split.get(r["split"], 0) + 1
    print(f"  positives per split: {n_pos_per_split}", file=sys.stderr)

    neg_quota_per_split: dict[str, int] = {
        s: int(n * neg_per_pos) for s, n in n_pos_per_split.items()
    }
    print(f"  neg quota per split (ratio={neg_per_pos}): {neg_quota_per_split}",
          file=sys.stderr)

    rng = random.Random(seed)
    # Reservoirs per split for negatives. Positives kept in full.
    pos_rows: list[dict] = []
    neg_reservoirs: dict[str, list[dict]] = {s: [] for s in n_pos_per_split}
    neg_seen_per_split: dict[str, int] = {s: 0 for s in n_pos_per_split}
    n_lines = 0
    with path.open() as fh:
        for line in fh:
            n_lines += 1
            r = json.loads(line)
            if int(r["label"]) == 1:
                pos_rows.append(r)
                continue
            split = r["split"]
            quota = neg_quota_per_split.get(split, 0)
            if quota <= 0:
                continue
            seen = neg_seen_per_split[split]
            res = neg_reservoirs[split]
            if len(res) < quota:
                res.append(r)
            else:
                j = rng.randint(0, seen)  # 0..seen inclusive
                if j < quota:
                    res[j] = r
            neg_seen_per_split[split] = seen + 1
            if n_lines % 1_000_000 == 0:
                print(f"  scanned {n_lines:,} rows ...", file=sys.stderr)

    print(f"  scanned {n_lines:,} rows total", file=sys.stderr)
    rows = pos_rows + [r for res in neg_reservoirs.values() for r in res]
    rng.shuffle(rows)

    feats = [r["feat"] for r in rows]
    labels = [r["label"] for r in rows]
    episodes = [r["episode"] for r in rows]
    splits = [r["split"] for r in rows]
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, episodes, splits


def main(argv=None) -> int:
    import lightgbm as lgb

    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_BOOSTER))
    p.add_argument(
        "--min-pos-rate", type=float, default=0.002,
        help="Lower bound on pos_rate; abort below this. Real opp emits "
             "are sparse — default 0.002 = 1 emit per 500 candidates",
    )
    p.add_argument("--max-pos-rate", type=float, default=0.50)
    p.add_argument(
        "--neg-per-pos", type=float, default=10.0,
        help="Reservoir-sample N×n_pos negatives to bound memory. None=load all "
             "(default: 10.0 → ~600k train rows at 60k positives)",
    )
    p.add_argument(
        "--lite", action="store_true",
        help="Slice corpus to the 34-d cheap feature subset + zero the "
             "lite-encoder-under-approximated slots. Required when the "
             "agent uses the vectorized lite encoder at inference time.",
    )
    args = p.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f"ERROR: labels not found: {labels_path}", file=sys.stderr)
        return 1
    print(f"loading corpus from {labels_path} "
          f"(neg_per_pos={args.neg_per_pos}) ...", file=sys.stderr)
    X, y, episodes, splits = _load_corpus(labels_path, neg_per_pos=args.neg_per_pos)

    if args.lite:
        from lib.opp_features_lite import LITE_KEEP_INDICES
        X = X[:, LITE_KEEP_INDICES].copy()
        print(f"  lite-mode: sliced corpus to {X.shape[1]} dims "
              f"(28-d cheap-features-only schema; model never sees the "
              f"WM-dependent or ray-cast-dependent features)",
              file=sys.stderr)
        effective_dim = LITE_FEATURE_DIM
    else:
        effective_dim = FEATURE_DIM

    # Save the post-downsample corpus as compressed npz for the private
    # Kaggle dataset (much smaller than the original JSONL, and skips the
    # 10-min JSONL parse on cross-session reload).
    if args.neg_per_pos is not None:
        out_npz = labels_path.with_suffix(".downsampled.npz")
        np.savez_compressed(
            out_npz,
            X=X, y=y,
            episodes=np.asarray(episodes),
            splits=np.asarray(splits),
        )
        print(f"  saved downsampled corpus → {out_npz} "
              f"({out_npz.stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)
    if X.ndim != 2 or X.shape[1] != effective_dim:
        print(f"ERROR: expected (_,{effective_dim}) features, got {X.shape}",
              file=sys.stderr)
        return 1

    pos_rate = float(y.mean())
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    n_eps = len(set(episodes))
    print(f"corpus: n={len(y)}  pos={n_pos}  neg={n_neg}  "
          f"pos_rate={pos_rate:.4f}  unique_episodes={n_eps}")
    if not (args.min_pos_rate <= pos_rate <= args.max_pos_rate):
        print(f"ERROR: pos_rate {pos_rate:.4f} outside "
              f"[{args.min_pos_rate}, {args.max_pos_rate}] — "
              f"adjust top_k or label-match logic",
              file=sys.stderr)
        return 2

    split_arr = np.asarray(splits)
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    print(f"split: train_n={len(y_tr)} (pos={int(y_tr.sum())}) "
          f"val_n={len(y_va)} (pos={int(y_va.sum())})")
    if len(y_tr) == 0 or len(y_va) == 0:
        print("ERROR: train or val set is empty", file=sys.stderr)
        return 1
    if int(y_va.sum()) == 0:
        print("ERROR: zero positives in val — split is unworkable",
              file=sys.stderr)
        return 1

    params = dict(LGB_PARAMS)
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
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    # AUC-style: separate distribution of positive vs negative predictions
    if y_va.sum() > 0 and (1 - y_va).sum() > 0:
        pos_pred_mean = float(pred_va[y_va == 1].mean())
        neg_pred_mean = float(pred_va[y_va == 0].mean())
    else:
        pos_pred_mean = neg_pred_mean = float("nan")

    print(f"\n=== val ===")
    print(f"  val_acc@0.5={acc:.3f}  Brier={brier:.4f}")
    print(f"  val_acc@thr={THRESHOLD}={acc_thr:.3f}  "
          f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  precision@thr={prec:.3f}  recall@thr={rec:.3f}")
    print(f"  E[pred|y=1]={pos_pred_mean:.4f}  "
          f"E[pred|y=0]={neg_pred_mean:.4f}  "
          f"separation={pos_pred_mean - neg_pred_mean:+.4f}")

    # Walker parity gate — same as train_validator
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

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "threshold": THRESHOLD,
        "pos_rate": pos_rate,
        "best_iteration": int(best_iter),
        "val_acc_at_half": acc,
        "val_brier": brier,
        "val_precision_at_thr": prec,
        "val_recall_at_thr": rec,
        "pos_pred_mean": pos_pred_mean,
        "neg_pred_mean": neg_pred_mean,
        "feature_dim": effective_dim,
        "lite_mode": bool(args.lite),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote meta    -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
