"""Tests for the 3-source bundle (triples) extension to the existing
pair-joint enumeration in `agents/baseline/chooser_trajectory.py`.

The triples block enumerates C(K, 3) triples for any target where the
pair-loop produced no positive score, subject to a per-turn budget cap.
The diagnostic probe (commit 5794566) showed ~3 enemy/neutral targets
per turn vs v7_0 with ≥3 candidates available and no positive pair —
this extension tries 3-source bundles on that subset.

Each test runs one short game and inspects the instrumentation JSONL
the chooser writes (gated on `BASELINE_JOINT_INSTRUMENT` / the
module-level `_JOINT_INSTRUMENT_PATH` attribute). We monkey-patch the
module-level gates rather than fiddle with env-vars because the
constants are evaluated at module import time.
"""
from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments import make

import agents.baseline.chooser_trajectory as ct


def _run_one_game(seed: int, triples: bool, instrument_path: Path) -> None:
    """Run a 2P game with baseline vs nearest, override gates as
    requested, route instrumentation to `instrument_path`, then flush.
    """
    ct._JOINT_INSTRUMENT_ROWS.clear()
    old_path = ct._JOINT_INSTRUMENT_PATH
    old_triples = ct.JOINT_TRIPLES_ENABLED
    ct._JOINT_INSTRUMENT_PATH = str(instrument_path)
    ct.JOINT_TRIPLES_ENABLED = triples
    try:
        # Late imports so monkey-patches above are in effect.
        from agents.baseline.main import agent as baseline_agent
        from agents.simple.nearest import agent as nearest_agent
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([baseline_agent, nearest_agent])
        ct._joint_instrument_dump()  # atexit doesn't fire mid-test
    finally:
        ct._JOINT_INSTRUMENT_PATH = old_path
        ct.JOINT_TRIPLES_ENABLED = old_triples


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_triples_gate_off_never_attempts_triples(tmp_path: Path) -> None:
    """With the triples gate off, every per-target row must show zero
    triple attempts. Guards against accidentally-on production state."""
    p = tmp_path / "off.jsonl"
    _run_one_game(seed=42, triples=False, instrument_path=p)
    rows = _load_rows(p)
    assert rows, "no instrumentation rows captured (game produced none)"
    for r in rows:
        assert r["n_triples_attempted"] == 0, (
            f"triples attempted with gate off: {r}"
        )
        assert r["n_triples_positive"] == 0
        assert r["n_triples_solo_gated"] == 0


def test_triples_gate_on_attempts_at_least_one_triple(
    tmp_path: Path,
) -> None:
    """With the triples gate on, the diagnostic shows the block runs at
    least once across the game. Lower-bound check — probe data showed
    every game against `nearest` had ≥4 such turns within 100 steps."""
    p = tmp_path / "on.jsonl"
    _run_one_game(seed=42, triples=True, instrument_path=p)
    rows = _load_rows(p)
    assert rows, "no instrumentation rows captured"
    total_attempted = sum(r["n_triples_attempted"] for r in rows)
    assert total_attempted > 0, (
        "triples block never executed; check pair_positive_for_tgt gate"
    )


def test_triples_only_fire_when_no_pair_positive(tmp_path: Path) -> None:
    """Invariant: for any target where the chooser ran triples, the
    pair-loop must have produced 0 positive scores. Triples are
    'after pairs failed', not 'in addition to'."""
    p = tmp_path / "invariant.jsonl"
    _run_one_game(seed=42, triples=True, instrument_path=p)
    rows = _load_rows(p)
    triples_rows = [r for r in rows if r["n_triples_attempted"] > 0]
    assert triples_rows, "no triples attempted across the game"
    for r in triples_rows:
        assert r["n_pairs_positive"] == 0, (
            f"triples ran despite positive pair: {r}"
        )


def test_triples_per_target_attempts_bounded_by_top_k(
    tmp_path: Path,
) -> None:
    """A single target's triples attempts cannot exceed C(K, 3) where
    K = JOINT_TRIPLES_TOP_K (default 4 → C(4, 3) = 4)."""
    p = tmp_path / "cap.jsonl"
    _run_one_game(seed=42, triples=True, instrument_path=p)
    rows = _load_rows(p)
    # Combinatorial upper bound: choose 3 from JOINT_TRIPLES_TOP_K.
    k = ct.JOINT_TRIPLES_TOP_K
    max_per_target = (k * (k - 1) * (k - 2)) // 6
    for r in rows:
        assert r["n_triples_attempted"] <= max_per_target, (
            f"target exceeded per-target triple cap {max_per_target}: "
            f"{r}"
        )
