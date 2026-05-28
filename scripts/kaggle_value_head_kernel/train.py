"""Orbit Wars value-head training kernel — runs on Kaggle T4 GPU.

Loads `training.npz` and `validation.npz` from
`/kaggle/input/orbit-wars-value-head-data/`, trains a 40 -> 128 -> 128
-> 1 MLP with MSE on final-margin labels, writes the trained weights as
NumPy arrays to `/kaggle/working/value_head_weights.npz` for the agent
bundler to consume.

CLAUDE.md Rule 2 two-tier smoke gate:
  Tier-1 (CPU, local)  -> `scripts/smoke_value_head_local.py` (must run
                          and pass before any kernel push).
  Tier-2 (GPU, inline) -> runs at the top of `main()` below; aborts with
                          a non-zero exit before the 20-epoch run begins
                          if the 1-epoch / 1k-example smoke takes
                          > 120 s or NaNs.

Output schema (`/kaggle/working/value_head_weights.npz`):
  W0 (40, 128) float32        b0 (128,) float32
  W1 (128, 128) float32       b1 (128,) float32
  W2 (128, 1)  float32        b2 (1,)   float32
  train_rmse  scalar  (final epoch)
  val_rmse    scalar  (final epoch)
  arch        bytes   ('40-128-128-1' utf-8)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KAGGLE_DATA_DIR = Path("/kaggle/input/orbit-wars-value-head-data")
_KAGGLE_OUT_DIR = Path("/kaggle/working")
_LOCAL_DATA_DIR = Path(os.environ.get(
    "VALUE_HEAD_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "value_head"),
))
_LOCAL_OUT_DIR = Path(os.environ.get(
    "VALUE_HEAD_OUT_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "value_head"),
))
# Kaggle path takes precedence if it exists (when running inside the
# kernel); otherwise fall back to local paths.
DATA_DIR = _KAGGLE_DATA_DIR if _KAGGLE_DATA_DIR.exists() else _LOCAL_DATA_DIR
OUT_DIR = _KAGGLE_OUT_DIR if _KAGGLE_OUT_DIR.exists() else _LOCAL_OUT_DIR

FEATURE_DIM = 40
HIDDEN = 128
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Tier-2 inline smoke
SMOKE_EXAMPLES = 1_000
SMOKE_EPOCHS = 1
SMOKE_DEADLINE_S = 120.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """40 -> 128 -> 128 -> 1 MLP. ReLU hidden, linear output (regression)."""

    def __init__(self, dim: int = FEATURE_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.fc0 = nn.Linear(dim, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — did the dataset upload?")
    d = np.load(path)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{name}: expected feature dim {FEATURE_DIM}, got {X.shape[1]}"
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: X/y row mismatch {X.shape} vs {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{name}: contains NaN/Inf")
    return X, y


def _load_or_split(val_frac: float = 0.1, seed: int = 1337) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Load training.npz; if validation.npz exists, use it; else
    random-split the training set 90/10."""
    X_train, y_train = _load_split("training.npz")
    val_path = DATA_DIR / "validation.npz"
    if val_path.exists():
        X_val, y_val = _load_split("validation.npz")
        return X_train, y_train, X_val, y_val
    print(
        f"no validation.npz; splitting training {1 - val_frac:.0%}/"
        f"{val_frac:.0%} (seed={seed})", flush=True,
    )
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return (
        X_train[train_idx], y_train[train_idx],
        X_train[val_idx], y_train[val_idx],
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(1, n)


@torch.no_grad()
def _eval(
    model: nn.Module, X: np.ndarray, y: np.ndarray,
    device: torch.device, batch_size: int = 4096,
) -> tuple[float, float]:
    """Returns (mse, rmse)."""
    model.eval()
    total = 0.0
    n = 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        yb = torch.from_numpy(y[i:i + batch_size]).to(device)
        pred = model(xb)
        total += float(((pred - yb) ** 2).sum().item())
        n += xb.shape[0]
    mse = total / max(1, n)
    return mse, float(np.sqrt(mse))


def _save_weights(model: nn.Module, train_rmse: float, val_rmse: float) -> Path:
    """Persist weights as NumPy arrays for the agent-side inference module."""
    out_path = OUT_DIR / "value_head_weights.npz"
    sd = model.state_dict()
    # nn.Linear stores weight as (out, in); we want (in, out) for a
    # `np.dot(x, W) + b` convention in the NumPy inference path.
    W0 = sd["fc0.weight"].detach().cpu().numpy().T.astype(np.float32)
    b0 = sd["fc0.bias"].detach().cpu().numpy().astype(np.float32)
    W1 = sd["fc1.weight"].detach().cpu().numpy().T.astype(np.float32)
    b1 = sd["fc1.bias"].detach().cpu().numpy().astype(np.float32)
    W2 = sd["fc2.weight"].detach().cpu().numpy().T.astype(np.float32)
    b2 = sd["fc2.bias"].detach().cpu().numpy().astype(np.float32)
    np.savez(
        out_path,
        W0=W0, b0=b0, W1=W1, b1=b1, W2=W2, b2=b2,
        train_rmse=np.float32(train_rmse),
        val_rmse=np.float32(val_rmse),
        arch=np.bytes_(f"{FEATURE_DIM}-{HIDDEN}-{HIDDEN}-1"),
    )
    return out_path


# ---------------------------------------------------------------------------
# Tier-2 inline smoke
# ---------------------------------------------------------------------------


def _tier2_smoke(
    X_train: np.ndarray, y_train: np.ndarray, device: torch.device,
) -> None:
    """1 epoch on first SMOKE_EXAMPLES examples; must finish in
    SMOKE_DEADLINE_S and produce a finite loss. Aborts early on failure
    so we don't burn 20-epoch worth of T4 quota on a broken kernel.
    """
    print(f"[smoke] tier-2 inline gate: {SMOKE_EXAMPLES} examples, "
          f"{SMOKE_EPOCHS} epoch(s), deadline {SMOKE_DEADLINE_S:.0f}s",
          flush=True)
    n = min(SMOKE_EXAMPLES, len(X_train))
    Xs = torch.from_numpy(X_train[:n])
    ys = torch.from_numpy(y_train[:n])
    loader = DataLoader(
        TensorDataset(Xs, ys), batch_size=BATCH_SIZE, shuffle=True,
    )
    model = ValueHead().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t0 = time.perf_counter()
    loss = _train_one_epoch(model, loader, opt, nn.MSELoss(), device)
    elapsed = time.perf_counter() - t0
    if not np.isfinite(loss):
        raise RuntimeError(f"[smoke] non-finite loss {loss}")
    if elapsed > SMOKE_DEADLINE_S:
        raise RuntimeError(
            f"[smoke] tier-2 took {elapsed:.1f}s > {SMOKE_DEADLINE_S:.0f}s "
            f"— aborting before full run"
        )
    print(f"[smoke] tier-2 OK: loss={loss:.2f}, {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"DATA_DIR: {DATA_DIR}", flush=True)
    print(f"OUT_DIR:  {OUT_DIR}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    X_train, y_train, X_val, y_val = _load_or_split()
    print(
        f"train: X={X_train.shape} y={y_train.shape} "
        f"y_mean={y_train.mean():+.1f} y_std={y_train.std():.1f}",
        flush=True,
    )
    print(
        f"val:   X={X_val.shape} y={y_val.shape} "
        f"y_mean={y_val.mean():+.1f} y_std={y_val.std():.1f}",
        flush=True,
    )

    _tier2_smoke(X_train, y_train, device)

    # Full training
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    model = ValueHead().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    history = []
    best_val_rmse = float("inf")
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_mse = _train_one_epoch(model, loader, opt, loss_fn, device)
        val_mse, val_rmse = _eval(model, X_val, y_val, device)
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "wallclock_s": time.perf_counter() - t0,
        })
        improved = val_rmse < best_val_rmse
        if improved:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in
                          model.state_dict().items()}
        print(
            f"epoch {epoch:2d}/{EPOCHS}: train_mse={train_mse:.2f} "
            f"val_mse={val_mse:.2f} val_rmse={val_rmse:.2f}"
            f"{' <- best' if improved else ''}",
            flush=True,
        )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    train_mse_final, train_rmse_final = _eval(model, X_train, y_train, device)
    val_mse_final, val_rmse_final = _eval(model, X_val, y_val, device)
    print(
        f"\nFINAL: train_rmse={train_rmse_final:.2f} "
        f"val_rmse={val_rmse_final:.2f} (best epoch by val_rmse)",
        flush=True,
    )

    out_path = _save_weights(model, train_rmse_final, val_rmse_final)
    print(f"wrote weights -> {out_path}", flush=True)

    # Persist training log alongside weights for postmortem.
    (OUT_DIR / "training_history.json").write_text(
        json.dumps({
            "history": history,
            "final": {
                "train_rmse": train_rmse_final,
                "val_rmse": val_rmse_final,
            },
            "config": {
                "feature_dim": FEATURE_DIM,
                "hidden": HIDDEN,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
