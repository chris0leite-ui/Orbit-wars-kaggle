"""Embed trained value-head weights into `agents/baseline/value_learned.py`.

Takes a `value_head_weights.npz` produced by
`scripts/kaggle_value_head_kernel/train.py` (or by the local Tier-1
smoke), reads its arrays, base64-encodes the entire .npz blob verbatim,
and rewrites the `WEIGHTS_B64 = "..."` literal at the top of
`agents/baseline/value_learned.py`.

Idempotent — running twice with the same input produces the same source.

Usage:
  python scripts/embed_value_head_weights.py \\
      --weights /kaggle/working/value_head_weights.npz
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VALUE_LEARNED_PATH = ROOT / "agents" / "baseline" / "value_learned.py"

EXPECTED_KEYS = ["W0", "b0", "W1", "b1", "W2", "b2"]
EXPECTED_SHAPES = {
    "W0": (40, 128),
    "b0": (128,),
    "W1": (128, 128),
    "b1": (128,),
    "W2": (128, 1),
    "b2": (1,),
}


def _validate(weights_path: Path) -> bytes:
    """Validate the weights file shape, return the canonical .npz bytes."""
    d = np.load(weights_path)
    for k in EXPECTED_KEYS:
        if k not in d.files:
            raise ValueError(f"{weights_path}: missing key {k!r}")
        arr = d[k]
        if arr.shape != EXPECTED_SHAPES[k]:
            raise ValueError(
                f"{weights_path}: {k} shape {arr.shape}, "
                f"expected {EXPECTED_SHAPES[k]}"
            )
        if arr.dtype != np.float32:
            raise ValueError(
                f"{weights_path}: {k} dtype {arr.dtype}, expected float32"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{weights_path}: {k} contains NaN/Inf")
    # Re-pack a canonical .npz containing only the 6 weight arrays so
    # the embedded blob doesn't drag along metadata (training_history,
    # arch strings, etc.).
    buf = io.BytesIO()
    np.savez(
        buf,
        **{k: d[k].astype(np.float32) for k in EXPECTED_KEYS},
    )
    return buf.getvalue()


def _rewrite_literal(source: str, b64: str) -> str:
    """Replace the existing `WEIGHTS_B64 = "..."` literal."""
    pattern = re.compile(r'^WEIGHTS_B64\s*=\s*"[^"]*"', re.MULTILINE)
    if not pattern.search(source):
        raise RuntimeError(
            "could not find `WEIGHTS_B64 = \"...\"` literal in "
            f"{VALUE_LEARNED_PATH}"
        )
    return pattern.sub(f'WEIGHTS_B64 = "{b64}"', source, count=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True,
                    help="path to value_head_weights.npz")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"error: {args.weights} not found", file=sys.stderr)
        return 1

    print(f"validating {args.weights}", flush=True)
    blob = _validate(args.weights)
    b64 = base64.b64encode(blob).decode("ascii")
    print(
        f"blob size: {len(blob)} bytes -> base64 {len(b64)} chars",
        flush=True,
    )

    src = VALUE_LEARNED_PATH.read_text()
    new = _rewrite_literal(src, b64)
    VALUE_LEARNED_PATH.write_text(new)
    print(
        f"updated {VALUE_LEARNED_PATH.relative_to(ROOT)} "
        f"({len(new)} chars)", flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
