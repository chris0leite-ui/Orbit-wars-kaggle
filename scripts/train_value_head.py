"""Train a single LightGBM regressor for Reframe B.2's value head.

Loads `data/value_head/corpus_runs/<run>/corpus.jsonl` (produced by
`scripts/gen_b2_corpus.py`), does a GAME-LEVEL 80/20 split (row-level
split leaks ~15pp val accuracy per konbu17's notebook; the same logic
applies to regression), trains one regression-objective Booster, and
writes the model text to `data/value_head/value_head_model.txt`.

Pure-Python inference goes through `lib._validator_tree_walker.predict_raw`
(no sigmoid for regression). Parity-tested vs `Booster.predict(...)` to
~1e-6 before save.

Decision gates (mandatory before bundling):
  - Label variance sanity: σ ≈ 200-1500 ships. σ < 200 → likely game-end
    truncation; σ > 1500 → likely seat-accounting bug. Aborts.
  - Training-time Spearman rank-order gate (PM3 — `2026-05-28-pm-
    distillation-action-rank-collapse.md`): Spearman ρ of (head_output,
    actual_K10_delta) on held-out val must be > 0.10 with a margin
    above zero. Distillation R²=0.998 without rank-order preservation
    collapsed Phase A; this gate catches the same failure mode.

Usage:
    python scripts/train_value_head.py \\
        --corpus data/value_head/corpus_runs/<run>/corpus.jsonl \\
        --out data/value_head/value_head_model.txt
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

FEATURE_DIM = 15  # 14 base + leaf_delta

# Conservative LightGBM hyperparams for ~10-40k rows, 15 features.
NUM_BOOST_ROUND = 600
EARLY_STOPPING_ROUNDS = 40
LGB_PARAMS = {
    # regression_l1 is robust to the long-tailed ship-delta residual
    # (B.1 saw σ up to 900 for big-launch enemy captures). L2 would let
    # those outliers dominate the loss.
    "objective": "regression_l1",
    "metric": ["mae", "rmse"],
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

# Decision gates.
# B.2 absolute-delta scale: σ(K=10 ship-delta) lands ~600-1000 ships.
# B.3 CRN-paired advantage is much smaller (rollouts cancel most of the
# absolute drift): σ(label) ~ 10-60 ships on 14k smoke. Band widened
# accordingly. Failing the band still aborts (catches game-end
# truncation or seat-accounting bugs); the band must just admit both
# label scales.
LABEL_SIGMA_MIN = 5.0
LABEL_SIGMA_MAX = 1500.0
SPEARMAN_RHO_MIN = 0.10


def _load_corpus(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats, labels, game_ids = [], [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            f = list(r["features"])
            # B.2 corpus writes features as the full 15-d vector with
            # leaf_delta (a.k.a. delta_pred) already appended at index
            # 14. B.3 corpus writes only the 14-d base features and
            # carries leaf_delta as a separate field; append it here so
            # the trainer sees the same 15-d schema in both cases.
            if len(f) == FEATURE_DIM - 1 and "leaf_delta" in r:
                f.append(float(r["leaf_delta"]))
            feats.append(f)
            labels.append(r["label"])
            game_ids.append(r["game_id"])
    if not feats:
        raise RuntimeError(f"empty corpus: {path}")
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, game_ids


def _game_level_split(
    game_ids: list[str], val_frac: float, rng: np.random.Generator,
) -> np.ndarray:
    """True=train, False=val. All examples from one game go to one side."""
    unique = sorted(set(game_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_games = set(unique[:n_val])
    return np.asarray([gid not in val_games for gid in game_ids])


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns 0.0 on degenerate input."""
    if len(a) < 3 or float(a.std()) == 0 or float(b.std()) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    import lightgbm as lgb  # heavy import; keep inside main

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True,
                   help="Path to corpus.jsonl from gen_b2_corpus.py")
    p.add_argument(
        "--out",
        default=str(REPO / "data" / "value_head" / "value_head_model.txt"),
        help="Output model text path",
    )
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--allow-low-rho", action="store_true",
                   help="Save the model even if Spearman ρ < "
                        f"{SPEARMAN_RHO_MIN} (for diagnostics only; "
                        "do NOT bundle a model that fails this gate)")
    args = p.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"ERROR: corpus not found: {corpus_path}", file=sys.stderr)
        return 1
    X, y, game_ids = _load_corpus(corpus_path)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        print(f"ERROR: expected (_,{FEATURE_DIM}) features, got {X.shape}",
              file=sys.stderr)
        return 1

    sigma_label = float(y.std())
    print(f"corpus: n={len(y)}  σ(label)={sigma_label:.1f}  "
          f"mean(label)={float(y.mean()):+.1f}  "
          f"unique_games={len(set(game_ids))}")
    if not (LABEL_SIGMA_MIN <= sigma_label <= LABEL_SIGMA_MAX):
        print(f"ERROR: σ(label) {sigma_label:.1f} outside healthy "
              f"[{LABEL_SIGMA_MIN}, {LABEL_SIGMA_MAX}] — investigate "
              f"label computation (game-end truncation, seat accounting)",
              file=sys.stderr)
        return 2

    rng_split = np.random.default_rng(0)
    train_mask = _game_level_split(game_ids, args.val_frac, rng_split)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[~train_mask], y[~train_mask]
    print(f"split: train_n={len(y_tr)}  val_n={len(y_va)}  "
          f"(val_games={args.val_frac:.0%}, GAME-LEVEL)")

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

    # Val metrics.
    pred_va = bst.predict(X_va, num_iteration=best_iter)
    rmse = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
    mae = float(np.mean(np.abs(pred_va - y_va)))
    # R² (coefficient of determination).
    ss_res = float(np.sum((y_va - pred_va) ** 2))
    ss_tot = float(np.sum((y_va - y_va.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    rho = _spearman_rho(np.asarray(pred_va), y_va)
    print(f"\n=== val ===")
    print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:+.4f}  "
          f"Spearman ρ={rho:+.4f}")
    print(f"  σ(y_va)={float(y_va.std()):.1f}  "
          f"σ(residual)={float((y_va - pred_va).std()):.1f}  "
          f"σ(pred)={float(pred_va.std()):.1f}")

    # Spearman ρ gate (mandatory unless --allow-low-rho).
    if rho < SPEARMAN_RHO_MIN and not args.allow_low_rho:
        print(f"\nERROR: Spearman ρ {rho:.3f} < required {SPEARMAN_RHO_MIN}",
              file=sys.stderr)
        print("  PM3 distillation collapse: high R² without rank-order "
              "preservation is a false positive. Pass --allow-low-rho to "
              "save anyway (diagnostics only).",
              file=sys.stderr)
        return 4

    # Walker parity check on val rows — confirm pure-numpy inference
    # matches `Booster.predict(...)` before we ship the artifact.
    from lib._validator_tree_walker import (
        parse_booster_text,
        predict_raw as walker_raw,
    )
    text = bst.model_to_string(num_iteration=best_iter)
    parsed = parse_booster_text(text)
    n_check = min(500, len(X_va))
    pred_walker = walker_raw(parsed, X_va[:n_check])
    pred_booster_raw = bst.predict(X_va[:n_check],
                                   num_iteration=best_iter,
                                   raw_score=True)
    max_diff = float(np.max(np.abs(pred_walker - pred_booster_raw)))
    print(f"\nwalker parity: max abs diff vs Booster on {n_check} val rows = "
          f"{max_diff:.3e}")
    if max_diff > 1e-4:
        print(f"ERROR: walker parity exceeds 1e-4 — refusing to save",
              file=sys.stderr)
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\nwrote model -> {out_path}  ({len(text):,} chars, "
          f"{len(parsed.trees)} trees)")

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "best_iteration": int(best_iter),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_rho": rho,
        "sigma_label": sigma_label,
        "feature_dim": FEATURE_DIM,
        "objective": LGB_PARAMS["objective"],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote meta  -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
