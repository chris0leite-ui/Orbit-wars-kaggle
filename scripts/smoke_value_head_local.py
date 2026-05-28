"""CLAUDE.md Rule 2 Tier-1: local CPU smoke for the value-head trainer.

Runs the exact training code path from `scripts/kaggle_value_head_kernel/
train.py` on a synthetic 100-example dataset on CPU, records peak memory
and wallclock. Must complete in < 5 minutes. Origin: 2026-05-14 cost
incident (90 min T4 burned on JIT compile that this would have caught).

What this catches BEFORE T4 push:
  - Tensor dtype mismatches (Python float vs torch float32)
  - Linear layer dimension mistakes (in/out swapped)
  - NaN gradients from bad LR / weight init
  - Missing torch / numpy versions on local dev

What this CANNOT catch (Tier-2 inline guard in train.py handles those):
  - CUDA-only failures (OOM at GPU batch sizes; cudnn version mismatch)
  - Datafile path resolution under /kaggle/input/

Usage:
  python scripts/smoke_value_head_local.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import the model directly from the kernel file via importlib so we run
# the EXACT same class definition the T4 will run. No duplication.
ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "scripts" / "kaggle_value_head_kernel"
sys.path.insert(0, str(KERNEL_DIR))
import train as kernel_train  # noqa: E402

SMOKE_EXAMPLES = 100
SMOKE_EPOCHS = 1
DEADLINE_S = 5 * 60


def _peak_mem_mib() -> float:
    """Peak RSS in MiB (Linux: ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    print(f"tier-1 smoke: {SMOKE_EXAMPLES} examples, "
          f"{SMOKE_EPOCHS} epoch(s), deadline {DEADLINE_S}s", flush=True)
    t0 = time.perf_counter()

    rng = np.random.default_rng(42)
    X = rng.standard_normal((SMOKE_EXAMPLES, kernel_train.FEATURE_DIM)
                            ).astype(np.float32)
    y = rng.standard_normal(SMOKE_EXAMPLES).astype(np.float32) * 30.0

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=32, shuffle=True,
    )
    device = torch.device("cpu")
    model = kernel_train.ValueHead().to(device)
    opt = optim.AdamW(
        model.parameters(),
        lr=kernel_train.LR,
        weight_decay=kernel_train.WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    for ep in range(SMOKE_EPOCHS):
        loss = kernel_train._train_one_epoch(model, loader, opt, loss_fn, device)
        if not np.isfinite(loss):
            print(f"FAIL: non-finite loss at epoch {ep}: {loss}", flush=True)
            return 1
        print(f"epoch {ep + 1}: loss={loss:.2f}", flush=True)

    # Run the eval path too — catches mse/rmse bugs.
    mse, rmse = kernel_train._eval(model, X, y, device)
    print(f"eval: mse={mse:.2f} rmse={rmse:.2f}", flush=True)

    # Persist a dummy weights file to confirm the save path works.
    out_path = Path("/tmp/value_head_weights_smoke.npz")
    sd = model.state_dict()
    np.savez(
        out_path,
        W0=sd["fc0.weight"].detach().cpu().numpy().T.astype(np.float32),
        b0=sd["fc0.bias"].detach().cpu().numpy().astype(np.float32),
        W1=sd["fc1.weight"].detach().cpu().numpy().T.astype(np.float32),
        b1=sd["fc1.bias"].detach().cpu().numpy().astype(np.float32),
        W2=sd["fc2.weight"].detach().cpu().numpy().T.astype(np.float32),
        b2=sd["fc2.bias"].detach().cpu().numpy().astype(np.float32),
        train_rmse=np.float32(rmse),
        val_rmse=np.float32(rmse),
    )

    elapsed = time.perf_counter() - t0
    peak = _peak_mem_mib()
    print(
        f"OK: elapsed={elapsed:.1f}s, peak_rss={peak:.1f} MiB, "
        f"weights -> {out_path}",
        flush=True,
    )
    if elapsed > DEADLINE_S:
        print(f"WARN: exceeded {DEADLINE_S}s deadline", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
