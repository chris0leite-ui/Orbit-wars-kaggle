"""Patch base64-encoded validator weights into agents/baseline_validated/main.py.

Mirrors `scripts/embed_value_head_weights.py`. Reads
`data/shot_validator/validator_ensemble_weights.npz`, validates shapes,
base64-encodes the entire npz blob, and replaces the `_WEIGHTS_B64 = ""`
literal in `agents/baseline_validated/main.py`.

Usage:
    python -m scripts.embed_validator_weights
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.shot_features import FEATURE_DIM  # noqa: E402

DEFAULT_NPZ = REPO / "data" / "shot_validator" / "validator_ensemble_weights.npz"
DEFAULT_AGENT = REPO / "agents" / "baseline_validated" / "main.py"

EXPECTED_KEYS = {"threshold", "pos_rate"}
for i in range(3):
    for k in ("W0", "b0", "W1", "b1", "W2", "b2"):
        EXPECTED_KEYS.add(f"m{i}_{k}")

EXPECTED_SHAPES = {}
for i in range(3):
    EXPECTED_SHAPES[f"m{i}_W0"] = (FEATURE_DIM, 64)
    EXPECTED_SHAPES[f"m{i}_b0"] = (64,)
    EXPECTED_SHAPES[f"m{i}_W1"] = (64, 32)
    EXPECTED_SHAPES[f"m{i}_b1"] = (32,)
    EXPECTED_SHAPES[f"m{i}_W2"] = (32, 1)
    EXPECTED_SHAPES[f"m{i}_b2"] = (1,)


def _validate(npz_path: Path) -> bytes:
    with np.load(npz_path) as z:
        keys = set(z.files)
        missing = EXPECTED_KEYS - keys
        if missing:
            raise SystemExit(f"missing keys in {npz_path}: {sorted(missing)}")
        for k, want in EXPECTED_SHAPES.items():
            got = z[k].shape
            if got != want:
                raise SystemExit(f"shape mismatch on {k}: got {got}, want {want}")
        # Re-pack to a clean canonical npz (no extra keys, predictable order).
        buf = io.BytesIO()
        out = {k: z[k].astype(np.float32) if z[k].dtype != np.float32 else z[k]
               for k in sorted(EXPECTED_KEYS)}
        np.savez(buf, **out)
        return buf.getvalue()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--agent", default=str(DEFAULT_AGENT))
    args = p.parse_args(argv)

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        print(f"ERROR: weights not found: {npz_path}", file=sys.stderr)
        return 1

    blob = _validate(npz_path)
    b64 = base64.b64encode(blob).decode("ascii")

    agent_path = Path(args.agent)
    src = agent_path.read_text()
    pattern = re.compile(r'^_WEIGHTS_B64 = "[^"]*"', re.MULTILINE)
    if not pattern.search(src):
        print(f"ERROR: could not find `_WEIGHTS_B64 = \"...\"` line in {agent_path}",
              file=sys.stderr)
        return 1
    new_src = pattern.sub(f'_WEIGHTS_B64 = "{b64}"', src, count=1)
    agent_path.write_text(new_src)
    print(f"wrote {len(b64):,} chars of base64 weights into {agent_path}")
    print(f"  npz size = {len(blob):,} bytes; base64 = {len(b64):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
