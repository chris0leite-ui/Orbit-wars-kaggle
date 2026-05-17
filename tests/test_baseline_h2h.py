"""H2H gate: baseline vs the strongest in-tree opponent (v7_0_drop_one).

Slow — runs n=16 games. Skipped by default. Invoke with:
    BASELINE_RUN_H2H=1 python -m pytest tests/test_baseline_h2h.py -q -s

For the deeper n=64 gate, use:
    python fast.py eval baseline --vs v7_0_drop_one --n 64
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not os.environ.get("BASELINE_RUN_H2H"),
    reason="set BASELINE_RUN_H2H=1 to enable (each test takes ~2-4 min)",
)


def _wilson_lo(wins: int, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half)


def _load_agent(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_h2h_{Path(path).stem}", path)
    mod = spec.loader.load_module()
    return mod.agent


def _play(p0, p1, seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def test_baseline_vs_v7_0_drop_one_n16_per_seat():
    """Functional parity gate: baseline scores reasonably against the
    v7-line champion. Soft threshold (Wilson lo >= 0.30) — the production
    gate is `fast.py eval --n 64` (Wilson lo >= 0.45)."""
    from agents.baseline.main import agent as baseline

    bundle = REPO / "submissions" / "v7_0_drop_one.py"
    assert bundle.exists(), f"v7_0_drop_one bundle missing: {bundle}"
    opp = _load_agent(str(bundle))

    n_per_seat = 8
    wins = 0
    for s in range(n_per_seat):
        r0, r1 = _play(baseline, opp, s)
        if r0 is not None and r0 > r1:
            wins += 1
        elif r0 == r1:
            wins += 0.5
    for s in range(n_per_seat):
        r0, r1 = _play(opp, baseline, s + 100)
        if r1 is not None and r1 > r0:
            wins += 1
        elif r0 == r1:
            wins += 0.5

    n = 2 * n_per_seat
    lo = _wilson_lo(int(wins), n)
    pct = wins / n * 100
    print(f"\nbaseline vs v7_0_drop_one: {wins}/{n} ({pct:.1f}%) Wilson_lo={lo:.2f}")
    assert lo >= 0.30, (
        f"baseline regressed vs v7_0_drop_one: {wins}/{n} Wilson_lo={lo:.2f}"
    )
