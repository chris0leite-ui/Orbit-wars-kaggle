"""Patch the trained LightGBM booster into agents/baseline_validated/main.py.

Replaces the prior 3-MLP-ensemble npz embed. Reads
`data/shot_validator/validator_booster.txt` (LightGBM model dump),
gzips + base64-encodes it, and writes into the `_BOOSTER_B64 = ""`
placeholder in `agents/baseline_validated/main.py`. The matching
`_THRESHOLD` constant is patched from the sidecar metadata.

Usage:
    python -m scripts.embed_validator_weights
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_BOOSTER = REPO / "data" / "shot_validator" / "validator_booster.txt"
DEFAULT_META = REPO / "data" / "shot_validator" / "validator_booster.meta.json"
DEFAULT_AGENT = REPO / "agents" / "baseline_validated" / "main.py"


def _validate_booster_text(text: str) -> None:
    """Catch obvious format breakage before the agent ships."""
    if "tree" not in text[:20]:
        raise SystemExit("booster text does not start with `tree` header")
    if "\nTree=0" not in text:
        raise SystemExit("no Tree=0 block in booster text")
    if "max_feature_idx=" not in text:
        raise SystemExit("no max_feature_idx in booster header")
    # Parity check via walker round-trip — fail loud if our parser can't
    # round-trip the text we're about to embed.
    from lib._validator_tree_walker import parse_booster_text
    parsed = parse_booster_text(text)
    if not parsed.trees:
        raise SystemExit("parsed booster has no trees")
    print(f"booster validated: {len(parsed.trees)} trees, "
          f"num_features={parsed.num_features}, "
          f"sigmoid_scale={parsed.sigmoid_scale}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--booster", default=str(DEFAULT_BOOSTER))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--agent", default=str(DEFAULT_AGENT))
    args = p.parse_args(argv)

    booster_path = Path(args.booster)
    meta_path = Path(args.meta)
    agent_path = Path(args.agent)
    if not booster_path.is_file():
        print(f"ERROR: booster not found: {booster_path}", file=sys.stderr)
        return 1
    if not meta_path.is_file():
        print(f"ERROR: meta sidecar not found: {meta_path}", file=sys.stderr)
        return 1

    text = booster_path.read_text()
    _validate_booster_text(text)
    meta = json.loads(meta_path.read_text())
    threshold = float(meta["threshold"])

    # gzip-then-base64 to shrink the text dump (typical 3-5x reduction).
    blob_gz = gzip.compress(text.encode("utf-8"), compresslevel=9)
    b64 = base64.b64encode(blob_gz).decode("ascii")

    src = agent_path.read_text()
    pat_b64 = re.compile(r'^_BOOSTER_B64 = "[^"]*"', re.MULTILINE)
    if not pat_b64.search(src):
        print(f"ERROR: no `_BOOSTER_B64 = \"...\"` line in {agent_path}",
              file=sys.stderr)
        return 1
    new_src = pat_b64.sub(f'_BOOSTER_B64 = "{b64}"', src, count=1)

    pat_thr = re.compile(r'^_THRESHOLD_DEFAULT = [0-9.]+', re.MULTILINE)
    if not pat_thr.search(new_src):
        print(f"ERROR: no `_THRESHOLD_DEFAULT = ...` line in {agent_path}",
              file=sys.stderr)
        return 1
    new_src = pat_thr.sub(f"_THRESHOLD_DEFAULT = {threshold:.4f}",
                          new_src, count=1)

    agent_path.write_text(new_src)
    print(f"wrote {len(b64):,} chars of base64 (from "
          f"{len(blob_gz):,} bytes gzipped from {len(text):,} chars "
          f"booster text) into {agent_path}")
    print(f"  threshold patched to {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
