"""Push the decoded opp-distill corpus to a private Kaggle dataset.

First-time setup: creates the dataset. Subsequent runs: pushes a new
version with an updated message.

Files pushed:
  - labels.jsonl (~60 MB, the actual corpus)
  - labels.summary.json
  - manifest.json (replays used, decoder knobs, source day)
  - README.md (carried from data/opp_distill/)

The dataset lives at `chris0leite/orbit-wars-opp-distill-corpus` and is
PRIVATE. Future sessions can pull it with:
  kaggle datasets download -p data/opp_distill/ chris0leite/orbit-wars-opp-distill-corpus

This script avoids re-decoding 20 GB of raw replays on container resume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_USER = "chris0leite"  # Kaggle username from bootstrap env
DEFAULT_SLUG = "orbit-wars-opp-distill-corpus"
DEFAULT_TITLE = "Orbit Wars Opp Distill Corpus"
DEFAULT_SUBTITLE = "Distilled-ladder opp predictor training corpus"


def _write_metadata(stage_dir: Path, user: str, slug: str,
                    title: str, subtitle: str) -> None:
    meta = {
        "title": title,
        "subtitle": subtitle,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return res.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument(
        "--source-dir", default=str(REPO / "data" / "opp_distill"),
        help="Directory holding labels.jsonl + summary + manifest + README",
    )
    ap.add_argument(
        "--message", default="initial corpus push",
        help="Version message (used on subsequent updates)",
    )
    ap.add_argument(
        "--create", action="store_true",
        help="First-time create (vs version-update). Default: version-update.",
    )
    args = ap.parse_args()

    src = Path(args.source_dir)
    labels = src / "labels.jsonl"
    if not labels.is_file():
        print(f"ERROR: {labels} not found — run decode_replays_to_labels.py first",
              file=sys.stderr)
        return 1

    # Stage in /tmp to avoid polluting repo (the kaggle CLI uploads the
    # whole staging dir).
    stage = Path("/tmp") / f"_kaggle_push_{args.slug}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # Copy the files we want in the dataset.
    files_to_include = ["labels.jsonl", "labels.summary.json",
                        "manifest.json", "README.md"]
    for fn in files_to_include:
        p = src / fn
        if p.is_file():
            shutil.copy(p, stage / fn)
            print(f"  staged {fn} ({p.stat().st_size:,} bytes)",
                  file=sys.stderr)
        else:
            print(f"  WARN: {p} missing — skipping", file=sys.stderr)

    _write_metadata(stage, args.user, args.slug,
                    DEFAULT_TITLE, DEFAULT_SUBTITLE)

    if args.create:
        # First-time create. Adds the dataset to Kaggle as private.
        _run(["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode",
              "zip"])
        print(f"\nCREATED: kaggle.com/datasets/{args.user}/{args.slug}",
              file=sys.stderr)
        print("If you want it private, log into kaggle.com and toggle "
              "visibility (CLI doesn't expose privacy flag).", file=sys.stderr)
    else:
        # Version update.
        _run(["kaggle", "datasets", "version", "-p", str(stage), "-m",
              args.message, "--dir-mode", "zip"])
        print(f"\nVERSIONED: kaggle.com/datasets/{args.user}/{args.slug}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
