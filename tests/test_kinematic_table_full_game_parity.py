"""Full-game byte-identity gate for the Phase γ kinematic table swap.

Runs the same agent against itself with `KINEMATIC_TABLE_ENABLED=1` and
with the var unset, and asserts the move list at every step is
byte-identical. The agents are imported as Python callables (not file
paths) to bypass the `kaggle_environments` raw-loader's lux_ai_s3
relative-import bug — same harness shape as scripts/ab_*.py.

Three seeds cover the game-stage variety:
  - seed 42:  orbital-only opening + light combat (no comets early).
  - seed 7:   comet enters mid-game; classical bug surface for the
              old orbital-prediction bug (commit d9feee2).
  - seed 13:  edge case from prior session (mid-flight expiry was the
              residual after d9feee2; commit 1daec97).

`episodeSteps=200` is large enough to exercise comet spawn at tick 50
and the post-spawn dynamics; smaller than the env's 500-step default
so the test completes in CI time. Marked @pytest.mark.slow so the
fast smoke suite skips it; explicit-run gates Phase γ.

Plan: /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md
"""

from __future__ import annotations

import os

import pytest

from kaggle_environments import make

from agents.analytical_phase_c.main import agent as phase_c_agent
from agents.baseline.main import agent as baseline_agent
from lib.kinematic_table import clear_table as _kt_clear


def _capture_moves(seed: int, kt_on: bool, episode_steps: int) -> list:
    if kt_on:
        os.environ["KINEMATIC_TABLE_ENABLED"] = "1"
    else:
        os.environ.pop("KINEMATIC_TABLE_ENABLED", None)
    _kt_clear()  # fresh singleton per run

    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "episodeSteps": int(episode_steps)},
        debug=False,
    )
    env.run([phase_c_agent, baseline_agent])
    moves = []
    for step_data in env.steps:
        per_step = []
        for ag in step_data:
            action = ag.get("action") if isinstance(ag, dict) else getattr(ag, "action", None)
            per_step.append(action)
        moves.append(per_step)
    return moves


@pytest.mark.slow
@pytest.mark.parametrize("seed", [42, 7, 13])
def test_full_game_byte_identical_table_vs_inline(seed):
    """Game-stage-coverage gate: KINEMATIC_TABLE_ENABLED=1 vs unset
    must produce byte-identical move lists at every step. Anything
    else is a Phase γ bug."""
    moves_off = _capture_moves(seed, kt_on=False, episode_steps=200)
    moves_on = _capture_moves(seed, kt_on=True, episode_steps=200)

    assert len(moves_off) == len(moves_on), (
        f"seed {seed}: step count differs: off={len(moves_off)} on={len(moves_on)}"
    )
    if moves_off == moves_on:
        return  # byte-identical

    # Surface the first divergence.
    for i, (a, b) in enumerate(zip(moves_off, moves_on)):
        if a != b:
            raise AssertionError(
                f"seed {seed}: first divergence at step {i}:\n"
                f"  table_OFF: {a}\n"
                f"  table_ON : {b}"
            )
    # Lengths matched but no per-step diff — shouldn't reach here.
    raise AssertionError(f"seed {seed}: unreached")
