"""Sanity-inspect a B.2 corpus before training.

Reports label and feature statistics for the corpus produced by
`scripts/gen_b2_corpus.py`. Helps catch:
  - σ(label) outside the healthy [200, 1500] band (the trainer aborts
    on this but it's nicer to see it explicitly).
  - Per-owner-at-launch and per-seat label distribution (sanity for
    the within-owner verdict's residual scale).
  - Feature-column variance (a column with σ=0 is dead weight and
    points to an encoder bug).

Usage:
    python scripts/inspect_value_head_corpus.py \\
        --corpus data/value_head/corpus_runs/<run>/corpus.jsonl
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

from lib.value_head_features import FEATURE_NAMES, FEATURE_DIM_FULL


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    args = p.parse_args(argv)

    rows = []
    with open(args.corpus) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print("corpus is empty", file=sys.stderr)
        return 1

    X = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    seats = np.asarray([int(r["seat"]) for r in rows])
    game_ids = [r["game_id"] for r in rows]

    print(f"== corpus: {args.corpus} ==")
    print(f"  n_rows: {len(rows)}")
    print(f"  n_games: {len(set(game_ids))}")
    print(f"  feature dim: {X.shape[1]}  (expected {FEATURE_DIM_FULL})")
    print()
    print(f"== label ==")
    print(f"  σ(label): {float(y.std()):.1f}  (healthy band [200, 1500])")
    print(f"  mean(label): {float(y.mean()):+.1f}")
    print(f"  min/max: {float(y.min()):+.0f} / {float(y.max()):+.0f}")
    print()
    print(f"== by seat ==")
    for s in (0, 1):
        m = seats == s
        if m.sum() == 0:
            continue
        print(f"  seat {s}: n={int(m.sum())}  "
              f"σ(label)={float(y[m].std()):.1f}  "
              f"mean(label)={float(y[m].mean()):+.1f}")
    print()

    # Owner-at-launch breakdown using indices 2/3/4 (me/neutral/enemy).
    own_me = X[:, 2] > 0.5
    own_n = X[:, 3] > 0.5
    own_e = X[:, 4] > 0.5
    print(f"== by owner_at_launch ==")
    for name, mask in (("me", own_me), ("neutral", own_n), ("enemy", own_e)):
        if mask.sum() == 0:
            continue
        print(f"  {name}: n={int(mask.sum())}  "
              f"σ(label)={float(y[mask].std()):.1f}  "
              f"mean(label)={float(y[mask].mean()):+.1f}")
    print()

    print(f"== feature variance (σ; 0 = dead column) ==")
    for i in range(X.shape[1]):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i}"
        col = X[:, i]
        print(f"  {i:2d} {name:<38s} "
              f"σ={float(col.std()):.3f}  "
              f"min={float(col.min()):+.2f}  "
              f"max={float(col.max()):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
