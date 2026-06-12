"""Train the shot-success MLP and bake weights into agents/producer_plus/shot_mlp.py.

Input: data/shot_validator/labels.jsonl (from scripts/label_shot_outcomes.py).
Model: 24-32-16-8-1 MLP, ReLU hidden, sigmoid head, BCE loss (the May plan).
Split: GROUPED by source replay file (no episode leaks across train/val).

Outputs:
  - rewrites the BEGIN/END TRAINED WEIGHTS block in shot_mlp.py
  - data/shot_validator/model_meta.json (dims, metrics, provenance)
  - stdout: val AUC + threshold operating table (reject rate vs precision)

Usage:
    python -m scripts.train_shot_mlp [--labels PATH] [--epochs N]
                                     [--val-frac F] [--seed N] [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO / "data" / "shot_validator" / "labels.jsonl"
SHOT_MLP_PATH = REPO / "agents" / "producer_plus" / "shot_mlp.py"
META_OUT = REPO / "data" / "shot_validator" / "model_meta.json"

DIMS = [24, 32, 16, 8, 1]


def load_labels(path: Path):
    X, y, groups = [], [], []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            f = row["features"]
            if len(f) != DIMS[0]:
                continue
            X.append(f)
            y.append(int(row["label"]))
            groups.append(row.get("meta", {}).get("src_path", "?"))
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(groups))


def grouped_split(groups: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_set = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_set for g in groups])
    return ~is_val, is_val


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC (no sklearn dependency)."""
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    n = len(y_score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(len(DIMS) - 1):
            layers.append(torch.nn.Linear(DIMS[i], DIMS[i + 1]))
            if i < len(DIMS) - 2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, 0]


def export_weights(model: MLP) -> tuple[str, dict]:
    """Flatten Linear weights in shot_mlp's expected (W [in,out], b) order."""
    chunks = []
    with torch.no_grad():
        for mod in model.net:
            if isinstance(mod, torch.nn.Linear):
                chunks.append(mod.weight.T.contiguous().numpy().astype(np.float32).ravel())
                chunks.append(mod.bias.numpy().astype(np.float32).ravel())
    blob = np.concatenate(chunks)
    b64 = base64.b64encode(blob.tobytes()).decode("ascii")
    return b64, {"dims": DIMS}


def bake_weights(b64: str, meta: dict) -> None:
    src = SHOT_MLP_PATH.read_text()
    block = (
        "# === BEGIN TRAINED WEIGHTS (written by scripts/train_shot_mlp.py) ===\n"
        f"WEIGHTS_B64 = {b64!r}\n"
        f"WEIGHTS_META = {meta!r}\n"
        "# === END TRAINED WEIGHTS ===\n"
    )
    new = re.sub(
        r"# === BEGIN TRAINED WEIGHTS.*?# === END TRAINED WEIGHTS ===\n",
        block, src, flags=re.DOTALL,
    )
    if new == src:
        raise SystemExit("weights markers not found in shot_mlp.py")
    SHOT_MLP_PATH.write_text(new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--patience", type=int, default=20,
                    help="early stop after N epochs without val-AUC gain")
    ap.add_argument("--dry-run", action="store_true",
                    help="train + report, do NOT bake weights")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = load_labels(Path(args.labels))
    if len(X) == 0:
        print("ERROR: no labeled examples", file=sys.stderr)
        return 1
    tr, va = grouped_split(groups, args.val_frac, args.seed)
    base = float(y.mean())
    print(f"examples: {len(X)} (pos rate {base:.3f}); "
          f"train {int(tr.sum())} / val {int(va.sum())}; "
          f"episodes: {len(np.unique(groups))}")

    Xt = torch.from_numpy(X[tr]); yt = torch.from_numpy(y[tr])
    Xv = torch.from_numpy(X[va]); yv_np = y[va]

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch = -1.0, None, -1
    n = len(Xt)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).numpy()
        auc = auc_rank(yv_np, pv)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv)).numpy()

    print(f"val AUC = {best_auc:.4f}  (best epoch {best_epoch}, "
          f"stopped at {epoch})")
    print("\nthreshold | rejected | success-rate among rejected | among kept")
    for thr in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        rej = pv < thr
        if rej.sum() == 0:
            print(f"   {thr:.2f}   |    0.0%  |  —  |  {yv_np.mean():.3f}")
            continue
        print(f"   {thr:.2f}   |  {rej.mean()*100:5.1f}%  |  "
              f"{yv_np[rej].mean():.3f}  |  {yv_np[~rej].mean():.3f}")

    # Per-decile calibration
    print("\npredicted-decile calibration (val):")
    qs = np.quantile(pv, np.linspace(0, 1, 11))
    for d in range(10):
        m = (pv >= qs[d]) & (pv <= qs[d + 1])
        if m.sum() == 0:
            continue
        print(f"  p in [{qs[d]:.2f},{qs[d+1]:.2f}]: n={int(m.sum()):5d} "
              f"empirical success {yv_np[m].mean():.3f}")

    if args.dry_run:
        print("\n--dry-run: weights NOT baked")
        return 0

    b64, meta = export_weights(model)
    bake_weights(b64, meta)

    # Round-trip guard: the baked numpy forward must reproduce the torch
    # model (catches transpose/order bugs in export_weights).
    import importlib
    sys.path.insert(0, str(REPO / "agents" / "producer_plus"))
    import shot_mlp as _sm
    importlib.reload(_sm)
    sample = X[va][:256]
    np_p = _sm.predict_success(sample)
    with torch.no_grad():
        t_p = torch.sigmoid(model(torch.from_numpy(sample))).numpy()
    max_diff = float(np.abs(np_p - t_p).max())
    if max_diff > 1e-5:
        raise SystemExit(f"round-trip FAIL: numpy vs torch max diff {max_diff}")
    print(f"round-trip OK (numpy vs torch max diff {max_diff:.2e})")
    META_OUT.write_text(json.dumps({
        "dims": DIMS,
        "val_auc": best_auc,
        "n_examples": int(len(X)),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "pos_rate": base,
        "n_episodes": int(len(np.unique(groups))),
        "labels_path": str(args.labels),
        "seed": args.seed,
        "trained_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"\nbaked weights into {SHOT_MLP_PATH.name} "
          f"({len(b64)} b64 chars); meta -> {META_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
