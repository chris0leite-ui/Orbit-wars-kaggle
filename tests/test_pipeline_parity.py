"""Phase A — bit-parity test for the pipeline composer.

The composed agent (`lib.pipeline.default_composition()`) MUST emit
exactly the same moves as the legacy `lib.joint_solver.mpc.solve_turn`
entry point. This test drives a real game with a dual-call wrapper:
on each turn, both paths are invoked on the same observation; any
discrepancy aborts the test.

If this test ever fails, the pipeline scaffold has introduced behavior
drift somewhere — likely a stage-ordering bug or an argument mismatch
between the composer and the legacy code path. Fix before proceeding
to Phase B.
"""

from __future__ import annotations

import os

import pytest


def _set_analytical_env_overrides() -> dict:
    """Mimic agents/analytical/main.py's per-call env override."""
    saved: dict = {}
    for key in ("PROPOSER_DRAIN_FILTER", "PROPOSER_HOLD_FEASIBILITY"):
        saved[key] = os.environ.get(key)
        os.environ[key] = "off"
    return saved


def _restore_env(saved: dict) -> None:
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _make_dual_agent():
    """Wrapper agent that calls both solve_turn and the composed pipeline.

    Records any divergence in the closed-over `discrepancies` list.
    Returns the composed pipeline's moves (if parity holds, equivalent).
    """
    from lib.joint_solver.mpc import solve_turn
    from lib.pipeline import default_composition

    composed = default_composition()
    discrepancies: list[tuple] = []

    def dual_agent(obs, configuration=None):
        saved = _set_analytical_env_overrides()
        try:
            old_moves = solve_turn(obs, configuration)
            new_moves = composed(obs, configuration)
        finally:
            _restore_env(saved)
        # Normalize move shape for comparison.
        old_norm = [
            [int(m[0]), float(m[1]), int(m[2])] for m in (old_moves or [])
        ]
        new_norm = [
            [int(m[0]), float(m[1]), int(m[2])] for m in (new_moves or [])
        ]
        if old_norm != new_norm:
            step = int(obs.get("step", -1)) if isinstance(obs, dict) else -1
            discrepancies.append((step, old_norm, new_norm))
        # Return the new (composed) moves; if parity holds, identical.
        return new_moves

    return dual_agent, discrepancies


@pytest.mark.parametrize("seed", [42, 7])
def test_pipeline_parity_short_game(seed: int):
    """50-turn game vs simple/nearest; composed == solve_turn every turn."""
    from kaggle_environments import make
    nearest_path = "agents/simple/nearest.py"
    env = make("orbit_wars", configuration={
        "seed": seed,
        "episodeSteps": 50,  # bounded for fast test
    }, debug=False)
    dual, discrepancies = _make_dual_agent()
    try:
        env.run([dual, nearest_path])
    except Exception as e:
        pytest.fail(f"env.run raised: {e}; discrepancies so far: {discrepancies[:3]}")
    assert not discrepancies, (
        f"seed {seed}: {len(discrepancies)} divergent turn(s); "
        f"first 3: {discrepancies[:3]}"
    )


@pytest.mark.parametrize("seed", [42])
def test_pipeline_parity_full_game(seed: int):
    """Full-length game vs simple/nearest; bit-exact parity across all turns.

    Slower (~30s) — runs as part of the standard suite but separate from
    the short variant so a smoke pass is fast.
    """
    from kaggle_environments import make
    nearest_path = "agents/simple/nearest.py"
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    dual, discrepancies = _make_dual_agent()
    try:
        env.run([dual, nearest_path])
    except Exception as e:
        pytest.fail(f"env.run raised: {e}; discrepancies so far: {discrepancies[:3]}")
    assert not discrepancies, (
        f"seed {seed} full game: {len(discrepancies)} divergent turn(s); "
        f"first 3: {discrepancies[:3]}"
    )
