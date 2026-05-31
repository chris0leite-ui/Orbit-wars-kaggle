"""Seat-symmetry verification for the canonical-frame transform (2026-05-31).

Self-play with our deterministic bundle on both sides showed perfect
mirror symmetry for the first ~30 turns then divergent collapse to P0=30,
P1=1 at turn 500 — chaotic amplification of id-order / floating-point
tiebreaks systematically favoring lower-id ownership.

The fix in agents/baseline/main.py rotates the observation 180° through
board center when our seat is P1, runs the chooser in the rotated frame,
and rotates output angles back. Both seats then feed identical canonical
input into identical deterministic logic and produce mirror-symmetric
actions; the seat asymmetry inside our agent evaporates.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.baseline.main import _rotate_obs, _unrotate_actions
from lib.geometry import BOARD_SIZE
from lib.mirror import rotate_angle, rotate_xy


def _sample_obs() -> dict:
    """Minimal obs dict covering the fields _rotate_obs touches."""
    return {
        "player": 1,
        "step": 17,
        "angular_velocity": 0.012,
        "planets": [
            [0, -1, 10.0, 90.0, 1.0, 8, 3],
            [1, 0, 25.0, 25.0, 1.0, 12, 4],   # P0 home
            [2, -1, 90.0, 10.0, 1.0, 8, 3],
            [3, 1, 75.0, 75.0, 1.0, 12, 4],   # P1 home (180° of id=1)
        ],
        "fleets": [
            [0, 0, 30.0, 30.0, 0.5, 1, 5],
            [1, 1, 70.0, 70.0, 3.6, 3, 5],
        ],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [0, -1, 10.0, 90.0, 1.0, 8, 3],
            [1, -1, 25.0, 25.0, 1.0, 10, 4],
            [2, -1, 90.0, 10.0, 1.0, 8, 3],
            [3, -1, 75.0, 75.0, 1.0, 10, 4],
        ],
    }


def test_rotate_obs_is_involution():
    """Rotating twice through board center is the identity."""
    obs = _sample_obs()
    twice = _rotate_obs(_rotate_obs(obs))
    # Positions and angles should be bit-identical (mod 2π for angles).
    for p_orig, p_twice in zip(obs["planets"], twice["planets"]):
        assert p_orig[0] == p_twice[0]                  # id
        assert p_orig[1] == p_twice[1]                  # owner
        assert math.isclose(p_orig[2], p_twice[2], abs_tol=1e-12)  # x
        assert math.isclose(p_orig[3], p_twice[3], abs_tol=1e-12)  # y
        assert p_orig[5] == p_twice[5]                  # ships
        assert p_orig[6] == p_twice[6]                  # production
    for f_orig, f_twice in zip(obs["fleets"], twice["fleets"]):
        assert f_orig[0] == f_twice[0]                  # id
        assert f_orig[1] == f_twice[1]                  # owner
        assert math.isclose(f_orig[2], f_twice[2], abs_tol=1e-12)
        assert math.isclose(f_orig[3], f_twice[3], abs_tol=1e-12)
        # Angle is mod 2π — adding 2π is the identity.
        diff = (f_orig[4] - f_twice[4]) % (2 * math.pi)
        assert diff < 1e-12 or abs(diff - 2 * math.pi) < 1e-12
        assert f_orig[5] == f_twice[5]                  # from_planet_id
        assert f_orig[6] == f_twice[6]                  # ships


def test_rotate_obs_maps_home_pair():
    """P0 home and P1 home are 180°-rotations of each other; after the
    transform, P1 home sits where P0 home was (canonical-frame property)."""
    obs = _sample_obs()
    rotated = _rotate_obs(obs)
    # planet id=1 (P0 home) was at (25, 25); rotated to (75, 75).
    p0_home_rotated = next(p for p in rotated["planets"] if p[0] == 1)
    assert math.isclose(p0_home_rotated[2], BOARD_SIZE - 25.0)
    assert math.isclose(p0_home_rotated[3], BOARD_SIZE - 25.0)
    # planet id=3 (P1 home) was at (75, 75); rotated to (25, 25).
    p1_home_rotated = next(p for p in rotated["planets"] if p[0] == 3)
    assert math.isclose(p1_home_rotated[2], BOARD_SIZE - 75.0)
    assert math.isclose(p1_home_rotated[3], BOARD_SIZE - 75.0)


def test_unrotate_actions_round_trip():
    """Applying angle-rotation twice to an action's angle is identity (mod 2π)."""
    actions = [
        [1, 0.0, 30],
        [3, 1.234, 12],
        [5, math.pi, 7],
        [9, -2.5, 100],
    ]
    twice = _unrotate_actions(_unrotate_actions(actions))
    for orig, rt in zip(actions, twice):
        assert orig[0] == rt[0]   # src_id
        assert orig[2] == rt[2]   # ships
        diff = (orig[1] - rt[1]) % (2 * math.pi)
        assert diff < 1e-12 or abs(diff - 2 * math.pi) < 1e-12


def test_unrotate_actions_matches_rotate_angle():
    """The action-angle transform equals lib.mirror.rotate_angle."""
    actions = [[0, 1.5, 10]]
    rotated = _unrotate_actions(actions)
    expected = rotate_angle(1.5)
    assert math.isclose(rotated[0][1], expected, abs_tol=1e-12)


# --------------------------------------------------------------------------
# Integration: self-play seat balance.
# --------------------------------------------------------------------------

BUNDLE = str(REPO / "submissions" / "baseline.py")


def _selfplay_one_seed(seed: int) -> dict:
    """Run one self-play game with our bundle on both sides; return outcome."""
    code = (
        "import json, sys, time;"
        f"sys.path.insert(0, {str(REPO)!r});"
        "from kaggle_environments import make;"
        f"env = make('orbit_wars', configuration={{'seed': {seed}, 'episodeSteps': 500}}, debug=False);"
        f"env.run([{BUNDLE!r}, {BUNDLE!r}]);"
        "final = env.steps[-1];"
        "r0 = final[0]['reward']; r1 = final[1]['reward'];"
        "print(json.dumps({'r0': r0, 'r1': r1, 'n_steps': len(env.steps)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=900,
    )
    line = next(
        (l for l in reversed((proc.stdout or "").splitlines()) if l.startswith("{")),
        "",
    )
    import json
    if not line:
        return {"r0": None, "r1": None, "n_steps": 0,
                "stderr": (proc.stderr or "")[:400]}
    return json.loads(line)


@pytest.mark.timeout(2400)
def test_selfplay_seat_balance():
    """With the canonical-frame transform, our bundle's self-play should
    not systematically favor either seat.

    Before fix: seed=0 self-play landed P0=30, P1=1 — 100% P0 dominance.
    After fix: outcomes should be split, ties, or at least not systematic.

    We gate at |p0_wins - p1_wins| <= 1 across 5 seeds. With perfect
    canonical-frame play we expect mostly ties (engine ship-count tiebreak
    of two perfectly-mirrored end states). The point is to PROVE the
    structural P0 bias is gone, not to chase a specific win-rate target.
    """
    if not Path(BUNDLE).is_file():
        pytest.skip(f"bundle not found: {BUNDLE}")
    seeds = [0, 1, 2, 3, 4]
    p0_wins = 0
    p1_wins = 0
    ties = 0
    for s in seeds:
        result = _selfplay_one_seed(s)
        if result["r0"] is None:
            pytest.fail(f"self-play crashed at seed={s}: {result.get('stderr', '')}")
        if result["r0"] > result["r1"]:
            p0_wins += 1
        elif result["r1"] > result["r0"]:
            p1_wins += 1
        else:
            ties += 1
        print(f"  seed={s}: r0={result['r0']} r1={result['r1']} n_steps={result['n_steps']}")
    print(f"  TOTAL: P0_wins={p0_wins}  P1_wins={p1_wins}  ties={ties}")
    assert abs(p0_wins - p1_wins) <= 1, (
        f"Seat asymmetry persists: P0={p0_wins}, P1={p1_wins}, ties={ties}. "
        f"Expected |P0 − P1| ≤ 1 across 5 seeds."
    )
