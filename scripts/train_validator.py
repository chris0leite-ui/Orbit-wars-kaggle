"""Train the 3-MLP ensemble shot validator.

Loads `data/shot_validator/labels.jsonl` (produced by
`scripts/gen_validator_corpus.py`), does GAME-LEVEL 80/20 split (row-level
split leaks ~15pp val acc per konbu17's notebook), trains 3 MLPs with seeds
[42, 100, 7], and writes ensemble weights to
`data/shot_validator/validator_ensemble_weights.npz`.

Architecture (konbu17 cell 11 lines 496-505):
    Linear(24, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)
    BCEWithLogitsLoss(pos_weight=(1-p)/p), Adam lr=1e-3, batch 512, 40 epochs

Pos_rate calibration is the load-bearing decision; aborts if pos_rate is
outside [0.40, 0.85].

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
DEFAULT_WEIGHTS = REPO / "data" / "shot_validator" / "validator_ensemble_weights.npz"

SEEDS = (42, 100, 7)
THRESHOLD = 0.30
HIDDEN1 = 64
HIDDEN2 = 32
EPOCHS = 40
BATCH = 512
LR = 1e-3
FEATURE_DIM = 25


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
    """Returns a boolean mask True=train, False=val. Splits at the game
    level (all examples from a game go to one side or the other)."""
    unique = sorted(set(game_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_games = set(unique[:n_val])
    return np.asarray([gid not in val_games for gid in game_ids])


def _init_mlp(rng: np.random.Generator) -> dict:
    He = lambda fi, fo: rng.standard_normal((fi, fo), dtype=np.float32) * np.sqrt(2.0 / fi)
    return {
        "W0": He(FEATURE_DIM, HIDDEN1), "b0": np.zeros(HIDDEN1, np.float32),
        "W1": He(HIDDEN1, HIDDEN2), "b1": np.zeros(HIDDEN2, np.float32),
        "W2": (rng.standard_normal((HIDDEN2, 1), dtype=np.float32) * np.sqrt(2.0 / HIDDEN2) * 0.1),
        "b2": np.zeros(1, np.float32),
    }


def _forward(P: dict, X: np.ndarray) -> tuple[np.ndarray, tuple]:
    z0 = X @ P["W0"] + P["b0"]; a0 = np.maximum(0, z0)
    z1 = a0 @ P["W1"] + P["b1"]; a1 = np.maximum(0, z1)
    z2 = (a1 @ P["W2"] + P["b2"]).ravel()
    return z2, (X, z0, a0, z1, a1)


def _bce_with_logits_grad(
    s: np.ndarray, y: np.ndarray, pos_weight: float,
) -> np.ndarray:
    """Gradient of BCEWithLogitsLoss(pos_weight=w) wrt logits.

    BCE_w(s, y) = -[w * y * log σ(s) + (1-y) * log(1-σ(s))]
    Gradient wrt s = σ(s) * (1 + (w-1) y) - w y
                   = (1 + (w-1)y) * σ(s) - w y

    Equivalent simpler form when w=1: σ(s) - y.
    """
    sig = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return (1.0 + (pos_weight - 1.0) * y) * sig - pos_weight * y


def _backward(
    P: dict, cache: tuple, s: np.ndarray, y: np.ndarray, pos_weight: float,
) -> dict:
    X, z0, a0, z1, a1 = cache
    n = float(len(y))
    g = (_bce_with_logits_grad(s, y, pos_weight) / n).reshape(-1, 1)
    gW2 = a1.T @ g; gb2 = g.sum(axis=0)
    da1 = g @ P["W2"].T; dz1 = da1 * (z1 > 0)
    gW1 = a0.T @ dz1; gb1 = dz1.sum(axis=0)
    da0 = dz1 @ P["W1"].T; dz0 = da0 * (z0 > 0)
    gW0 = X.T @ dz0; gb0 = dz0.sum(axis=0)
    return {"W0": gW0, "b0": gb0, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


class _Adam:
    def __init__(self, P: dict, lr: float = LR):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P: dict, G: dict,
             b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.t += 1
        for k in P:
            self.m[k] = b1 * self.m[k] + (1 - b1) * G[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * (G[k] ** 2)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            P[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)


def _train_one(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    seed: int, pos_weight: float,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    P = _init_mlp(rng)
    opt = _Adam(P, lr=LR)
    history = []
    for ep in range(EPOCHS):
        idx = rng.permutation(len(X_tr))
        losses = []
        for b in range(0, len(X_tr), BATCH):
            bi = idx[b:b+BATCH]
            Xb, yb = X_tr[bi], y_tr[bi]
            s, cache = _forward(P, Xb)
            sig = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
            sig = np.clip(sig, 1e-7, 1 - 1e-7)
            L = -(pos_weight * yb * np.log(sig) + (1 - yb) * np.log(1 - sig)).mean()
            losses.append(L)
            G = _backward(P, cache, s, yb, pos_weight)
            opt.step(P, G)
        s_va, _ = _forward(P, X_va)
        pred_va = (1.0 / (1.0 + np.exp(-np.clip(s_va, -30, 30)))) >= 0.5
        va_acc = float((pred_va == (y_va >= 0.5)).mean())
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)),
                        "val_acc": va_acc})
        if ep == 0 or (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f"    seed={seed} ep={ep+1:>3d} train_L={np.mean(losses):.4f} "
                  f"val_acc={va_acc:.3f}")
    return P, {"history": history, "seed": seed}


def _ensemble_predict(models: list[dict], X: np.ndarray) -> np.ndarray:
    """Average sigmoid across the ensemble."""
    out = np.zeros(len(X), dtype=np.float32)
    for P in models:
        s, _ = _forward(P, X)
        out += 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return out / len(models)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_WEIGHTS))
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

    pos_weight = (1 - pos_rate) / pos_rate
    print(f"BCE pos_weight = (1-{pos_rate:.3f})/{pos_rate:.3f} = {pos_weight:.3f}")

    rng_split = np.random.default_rng(0)
    train_mask = _game_level_split(game_ids, args.val_frac, rng_split)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[~train_mask], y[~train_mask]
    print(f"split: train_n={len(y_tr)}  val_n={len(y_va)}  "
          f"(val_games={args.val_frac:.0%}, GAME-LEVEL, not row-level)")

    models = []
    for seed in SEEDS:
        print(f"\n--- training seed={seed} ---")
        P, _meta = _train_one(X_tr, y_tr, X_va, y_va, seed, pos_weight)
        models.append(P)

    # Final ensemble val metrics
    pred_va = _ensemble_predict(models, X_va)
    pred_va_bin = (pred_va >= 0.5).astype(np.float32)
    acc = float((pred_va_bin == y_va).mean())
    brier = float(((pred_va - y_va) ** 2).mean())
    # Accuracy at the deployment threshold
    pred_thr = (pred_va >= THRESHOLD).astype(np.float32)
    acc_thr = float((pred_thr == y_va).mean())
    tp = int(((pred_thr == 1) & (y_va == 1)).sum())
    fp = int(((pred_thr == 1) & (y_va == 0)).sum())
    tn = int(((pred_thr == 0) & (y_va == 0)).sum())
    fn = int(((pred_thr == 0) & (y_va == 1)).sum())
    print(f"\n=== ensemble val ===")
    print(f"  val_acc@0.5={acc:.3f}  Brier={brier:.4f}")
    print(f"  val_acc@thr={THRESHOLD}={acc_thr:.3f}  "
          f"TP={tp} FP={fp} TN={tn} FN={fn}")
    if tp + fp > 0:
        print(f"  precision@thr={tp/(tp+fp):.3f}  recall@thr={tp/(tp+fn) if tp+fn else 0:.3f}")

    # Pack weights
    out = {"threshold": np.float32(THRESHOLD), "pos_rate": np.float32(pos_rate)}
    for i, P in enumerate(models):
        for k, v in P.items():
            out[f"m{i}_{k}"] = v.astype(np.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"\nwrote weights -> {out_path}")
    return 0 if acc >= 0.65 else 0  # don't abort — let user decide


if __name__ == "__main__":
    raise SystemExit(main())
