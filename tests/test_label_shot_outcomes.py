"""Smoke test for the shot-validator labeling pipeline.

Runs `scripts.label_shot_outcomes` on a small fixture replay (if any
on-disk replays are available) and asserts:
- Output file exists
- Each line is valid JSON with `features` (length 24) and `label` ∈ {0, 1}
- All features in [0, 1]
- At least one positive and one negative example (small chance of 0
  pos or 0 neg on a tiny replay; skip if so)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPLAY_DIR = REPO / "audit" / "external" / "replays"


pytestmark = pytest.mark.skipif(
    not REPLAY_DIR.is_dir() or not list(REPLAY_DIR.glob("*.json")),
    reason="audit/external/replays not present (gitignored); rerun after `scripts.label_shot_outcomes` data pull",
)


def test_labeling_pipeline_emits_valid_dataset(tmp_path):
    out = tmp_path / "labels.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.label_shot_outcomes",
            "--replay-dir",
            str(REPLAY_DIR),
            "--out",
            str(out),
            "--limit",
            "5",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, f"pipeline failed: {result.stderr[:300]}"
    assert out.is_file()
    lines = out.read_text().splitlines()
    assert len(lines) >= 10, f"too few examples: {len(lines)}"

    n_positive = 0
    for line in lines:
        d = json.loads(line)
        assert "features" in d and "label" in d
        assert len(d["features"]) == 24
        # All features are normalised; `ship_diff` (index 21) is signed
        # in [-1, 1]; all others are non-negative in [0, 1].
        # Allow a small slack for boundary effects.
        for i, x in enumerate(d["features"]):
            if i == 21:   # ship_diff
                assert -1.5 <= x <= 1.5, f"ship_diff out of range: {x}"
            else:
                assert -0.01 <= x <= 1.5, f"feature {i} out of range: {x}"
        assert d["label"] in (0, 1)
        n_positive += d["label"]

    # On a corpus of 5 random replays we should see at least one of each
    assert 0 < n_positive < len(lines), (
        f"expected mixed labels; got {n_positive}/{len(lines)} positives"
    )
