"""Tests for lib/fingerprint.py — behavioural fingerprint over a K-turn prefix.

Two flavours of test:
  - hand-built replays: deterministic, fast, exercise each feature individually
  - end-to-end smoke: a real env run captured via tournament._build_replay,
    asserts that two distinct strategies produce distinguishable fingerprints
"""

from __future__ import annotations

from pathlib import Path

import importlib.util
import sys

import numpy as np
import pytest
from kaggle_environments import make

from lib.fingerprint import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    batch_fingerprints,
    fingerprint,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Hand-built replay helpers
# ---------------------------------------------------------------------------


def _planet(pid: int, owner: int, x: float, y: float, ships: int, prod: int = 1) -> list:
    """Build a planet tuple: [id, owner, x, y, radius, ships, prod]."""
    return [pid, owner, float(x), float(y), 1.0, int(ships), int(prod)]


def _step(
    i: int,
    planets: list[list],
    fleets: list[list],
    action_p0: list[list],
    action_p1: list[list],
) -> dict:
    return {
        "step": i,
        "planets": planets,
        "fleets": fleets,
        "action_p0": action_p0,
        "action_p1": action_p1,
    }


def _replay_from_steps(steps: list[dict], agent_p0: str = "A", agent_p1: str = "B") -> dict:
    return {
        "seed": 42,
        "agent_p0": agent_p0,
        "agent_p1": agent_p1,
        "n_steps": len(steps),
        "rewards": [0.0, 0.0],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Shape / contract
# ---------------------------------------------------------------------------


def test_fingerprint_returns_fixed_length_vector():
    rep = _replay_from_steps([_step(0, [_planet(0, 0, 10, 10, 100)], [], [], [])])
    fp = fingerprint(rep, player_id=0, prefix_turns=10)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (len(FEATURE_NAMES),)
    assert fp.dtype == np.float64


def test_fingerprint_empty_replay_returns_zero_vector():
    rep = {"steps": []}
    fp = fingerprint(rep, player_id=0, prefix_turns=10)
    assert fp.shape == (len(FEATURE_NAMES),)
    assert np.all(fp == 0.0)


def test_fingerprint_invalid_player_raises():
    rep = _replay_from_steps([])
    with pytest.raises(ValueError):
        fingerprint(rep, player_id=2, prefix_turns=10)


def test_feature_version_pinned():
    """Bumping FEATURE_VERSION is a load-bearing change — it invalidates
    any classifier trained against an earlier version. Catch accidental bumps.
    """
    assert FEATURE_VERSION == 1


# ---------------------------------------------------------------------------
# Per-feature behaviour
# ---------------------------------------------------------------------------


def test_launches_per_turn_counts_only_player_actions():
    # P0 launches once per turn for 5 turns; P1 launches twice per turn.
    steps = []
    for i in range(5):
        steps.append(_step(
            i,
            planets=[_planet(0, 0, 10, 10, 100), _planet(1, 1, 90, 90, 100)],
            fleets=[],
            action_p0=[[0, 0.0, 10]],
            action_p1=[[1, 3.14, 5], [1, 1.57, 5]],
        ))
    rep = _replay_from_steps(steps)
    fp_p0 = fingerprint(rep, 0, prefix_turns=5)
    fp_p1 = fingerprint(rep, 1, prefix_turns=5)
    idx = FEATURE_NAMES.index("launches_per_turn")
    assert fp_p0[idx] == pytest.approx(1.0)
    assert fp_p1[idx] == pytest.approx(2.0)


def test_mean_fleet_size_matches_action_ships():
    steps = [_step(0, [_planet(0, 0, 10, 10, 200)], [],
                   action_p0=[[0, 0.0, 30], [0, 0.0, 50]],
                   action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx = FEATURE_NAMES.index("mean_fleet_size")
    assert fp[idx] == pytest.approx(40.0)


def test_targets_neutral_fraction():
    # P0 always launches at planet 1 (neutral). Expected fraction = 1.0.
    src = _planet(0, 0, 10.0, 10.0, 100)
    neutral = _planet(1, -1, 90.0, 10.0, 50, prod=3)
    # angle from src (10,10) to (90,10) is 0.0 (straight east).
    steps = [_step(0, [src, neutral], [], action_p0=[[0, 0.0, 10]], action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx_neutral = FEATURE_NAMES.index("targets_neutral_fraction")
    idx_enemy = FEATURE_NAMES.index("targets_enemy_fraction")
    assert fp[idx_neutral] == pytest.approx(1.0)
    assert fp[idx_enemy] == pytest.approx(0.0)


def test_targets_enemy_fraction():
    src = _planet(0, 0, 10.0, 10.0, 100)
    enemy = _planet(1, 1, 90.0, 10.0, 50, prod=3)
    steps = [_step(0, [src, enemy], [], action_p0=[[0, 0.0, 10]], action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx_neutral = FEATURE_NAMES.index("targets_neutral_fraction")
    idx_enemy = FEATURE_NAMES.index("targets_enemy_fraction")
    assert fp[idx_neutral] == pytest.approx(0.0)
    assert fp[idx_enemy] == pytest.approx(1.0)


def test_mean_target_production_uses_inferred_target():
    src = _planet(0, 0, 10.0, 10.0, 100)
    # Two candidate targets at the same eastward direction; closer one wins.
    near = _planet(1, -1, 30.0, 10.0, 50, prod=2)
    far = _planet(2, -1, 80.0, 10.0, 50, prod=5)
    steps = [_step(0, [src, near, far], [], action_p0=[[0, 0.0, 10]], action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx = FEATURE_NAMES.index("mean_target_production")
    # The ray hits both; closer-with-positive-forward-projection wins (near).
    assert fp[idx] == pytest.approx(2.0)


def test_sun_clip_launch_rate_when_aimed_at_sun():
    # Source at top-left; aim straight at the centre (sun) — must clip.
    src = _planet(0, 0, 10.0, 10.0, 100)
    # Angle from (10,10) to (50,50) ≈ pi/4.
    import math
    steps = [_step(0, [src], [], action_p0=[[0, math.pi / 4, 10]], action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx = FEATURE_NAMES.index("sun_clip_launch_rate")
    assert fp[idx] == pytest.approx(1.0)


def test_sun_clip_zero_when_path_clears():
    # Source below the sun, fire eastward — path is far from sun.
    src = _planet(0, 0, 10.0, 90.0, 100)
    steps = [_step(0, [src], [], action_p0=[[0, 0.0, 10]], action_p1=[])]
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=1)
    idx = FEATURE_NAMES.index("sun_clip_launch_rate")
    assert fp[idx] == pytest.approx(0.0)


def test_state_features_track_my_planets():
    # P0 owns 2 planets in step 0, then 3 in step 1 (a capture).
    s0 = _step(
        0,
        planets=[_planet(0, 0, 10, 10, 50), _planet(1, 0, 20, 20, 50), _planet(2, -1, 80, 80, 30)],
        fleets=[],
        action_p0=[],
        action_p1=[],
    )
    s1 = _step(
        1,
        planets=[_planet(0, 0, 10, 10, 50), _planet(1, 0, 20, 20, 50), _planet(2, 0, 80, 80, 30)],
        fleets=[],
        action_p0=[],
        action_p1=[],
    )
    rep = _replay_from_steps([s0, s1])
    fp = fingerprint(rep, 0, prefix_turns=2)
    idx = FEATURE_NAMES.index("mean_planets_owned")
    assert fp[idx] == pytest.approx((2 + 3) / 2)


def test_ships_growth_is_positive_when_total_ships_rise():
    # P0's total ships grow linearly: 50, 60, 70 over 3 steps.
    steps = []
    for i, ships in enumerate([50, 60, 70]):
        steps.append(_step(
            i,
            planets=[_planet(0, 0, 10, 10, ships)],
            fleets=[],
            action_p0=[],
            action_p1=[],
        ))
    rep = _replay_from_steps(steps)
    fp = fingerprint(rep, 0, prefix_turns=3)
    idx = FEATURE_NAMES.index("ships_growth_per_turn")
    assert fp[idx] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Discrimination — distinct strategies must produce distinct fingerprints
# ---------------------------------------------------------------------------


def _load_tournament_module():
    """Load scripts/tournament.py under the stable name `tournament` so its
    dataclasses resolve consistently. Mirrors scripts/strategy_panel.py.
    """
    spec = importlib.util.spec_from_file_location(
        "tournament", REPO / "scripts" / "tournament.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tournament"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_distinct_strategies_have_distinct_fingerprints():
    """End-to-end: real env, two different agents (`weakest` vs `production`).
    Their fingerprints over the first 100 turns should differ noticeably."""
    tournament = _load_tournament_module()
    from agents.simple import production as prod_strat
    from agents.simple import weakest as weak_strat

    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run([prod_strat.agent, weak_strat.agent])
    replay = tournament._build_replay(env, 42, "production", "weakest")
    fp_prod = fingerprint(replay, player_id=0, prefix_turns=100)
    fp_weak = fingerprint(replay, player_id=1, prefix_turns=100)
    # L2 distance must be substantial — features are on different scales,
    # so this is a generous threshold.
    diff = float(np.linalg.norm(fp_prod - fp_weak))
    assert diff > 1.0, (
        f"production and weakest fingerprints should differ; got L2={diff:.3f}, "
        f"prod={fp_prod}, weak={fp_weak}"
    )


def test_batch_fingerprints_shapes_align():
    """`batch_fingerprints` should yield rows for both seats per replay."""
    tournament = _load_tournament_module()
    from agents.simple import nearest as near_strat

    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run([near_strat.agent, near_strat.agent])
    replay = tournament._build_replay(env, 42, "nearest", "nearest")
    X, labels, seeds, players = batch_fingerprints([replay], prefix_turns=100)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert labels == ["nearest", "nearest"]
    assert seeds == [42, 42]
    assert players == [0, 1]
