"""Upload `data/value_head/training.npz` (+ optional validation.npz) as a
Kaggle dataset so the GPU training kernel can read it.

Kaggle dataset references the kernel via its metadata's
`dataset_sources: ["chrisleitescha/orbit-wars-value-head-data"]`.

First-run: creates the dataset.
Subsequent runs: pushes a new version with a descriptive message.

Usage:
  # Initial create (one-time)
  python scripts/push_value_training_dataset.py --create

  # Subsequent version bump
  python scripts/push_value_training_dataset.py --version-notes \\
      "10k lite_greedy self-play, 8000 train + 200 val OOD"

This is a single-shot, PI-approved action (CLAUDE.md Rule 1) since it
publishes data to a Kaggle remote. Confirms the dataset directory
exists and the npz files validate before invoking `kaggle datasets`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "value_head"
DATASET_SLUG = "chrisleitescha/orbit-wars-value-head-data"
DATASET_TITLE = "Orbit Wars value head training data"


def _validate_npz(path: Path) -> None:
    """Reject any .npz that doesn't have well-formed (X, y) arrays."""
    d = np.load(path)
    if "X" not in d.files or "y" not in d.files:
        raise ValueError(f"{path}: missing X or y array")
    if d["X"].shape[1] != 40:
        raise ValueError(f"{path}: X feature dim {d['X'].shape[1]} != 40")
    if d["X"].shape[0] != d["y"].shape[0]:
        raise ValueError(f"{path}: X/y row mismatch")
    if not np.isfinite(d["X"]).all() or not np.isfinite(d["y"]).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    print(f"  validated {path.name}: X={d['X'].shape} y={d['y'].shape}")


def _ensure_dataset_metadata() -> None:
    """Write the dataset-metadata.json that `kaggle datasets create/version`
    expects in the upload directory."""
    meta = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _run_kaggle(args: list[str]) -> int:
    print(f"$ kaggle {' '.join(args)}", flush=True)
    return subprocess.call(["kaggle", *args])


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--create", action="store_true",
                   help="first-time dataset create")
    g.add_argument("--version-notes",
                   help="push a new dataset version with this message")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + write metadata, skip kaggle invocation")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        print(f"error: {DATA_DIR} does not exist", file=sys.stderr)
        return 1

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"error: no .npz files under {DATA_DIR}", file=sys.stderr)
        return 1

    # Insist the merged training.npz is present; per-pairing chunks alone
    # are not usable by the kernel.
    if not (DATA_DIR / "training.npz").exists():
        print(
            "error: data/value_head/training.npz not found. "
            "Run `python scripts/gen_value_training_data.py --merge "
            "data/value_head` first.",
            file=sys.stderr,
        )
        return 1

    print(f"validating {len(npz_files)} .npz files under {DATA_DIR}")
    for p in npz_files:
        _validate_npz(p)

    _ensure_dataset_metadata()

    if args.dry_run:
        print("dry-run: skipping kaggle invocation")
        return 0

    if args.create:
        rc = _run_kaggle(["datasets", "create", "-p", str(DATA_DIR)])
    else:
        rc = _run_kaggle(
            ["datasets", "version", "-p", str(DATA_DIR), "-m",
             args.version_notes],
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
