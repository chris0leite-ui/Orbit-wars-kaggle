"""Unit tests for lib/candidate_portfolios — portfolio generator."""

from __future__ import annotations

import pytest

from lib.candidate_portfolios import (
    Portfolio,
    _drop_weakest_source,
    _per_source_swap,
    generate_portfolios,
)
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel


def _mission(src_id, target_id, ships, score):
    return Mission(
        mission_class="snipe",
        src_id=src_id,
        target_id=target_id,
        ships=ships,
        score=score,
        eta=5,
    )


# ---------------------------------------------------------------------------
# _per_source_swap
# ---------------------------------------------------------------------------


def test_per_source_swap_empty_returns_none():
    assert _per_source_swap([]) is None


def test_per_source_swap_single_mission_per_source_returns_none():
    # Each source has exactly 1 mission → no swap possible.
    ms = [_mission(0, 10, 5, 100.0), _mission(1, 11, 5, 50.0)]
    assert _per_source_swap(ms) is None


def test_per_source_swap_drops_top1_from_smallest_gap_source():
    # Source 0: top1 score=100, top2=99 (gap 1) ← smallest gap
    # Source 1: top1 score=200, top2=100 (gap 100)
    ms = [
        _mission(0, 10, 5, 100.0),
        _mission(0, 11, 5, 99.0),
        _mission(1, 20, 5, 200.0),
        _mission(1, 21, 5, 100.0),
    ]
    out = _per_source_swap(ms)
    assert out is not None
    # Source 0's top-1 (target 10, score 100) is dropped.
    src0_targets = {m.target_id for m in out if m.src_id == 0}
    assert 10 not in src0_targets
    assert 11 in src0_targets
    # Source 1 is untouched.
    src1_targets = {m.target_id for m in out if m.src_id == 1}
    assert src1_targets == {20, 21}


# ---------------------------------------------------------------------------
# _drop_weakest_source
# ---------------------------------------------------------------------------


def test_drop_weakest_source_empty_returns_none():
    assert _drop_weakest_source([]) is None


def test_drop_weakest_source_single_source_returns_none():
    """One source — dropping it is the no-op portfolio, not this one."""
    ms = [_mission(0, 10, 5, 100.0), _mission(0, 11, 5, 50.0)]
    assert _drop_weakest_source(ms) is None


def test_drop_weakest_source_filters_lowest_top_score_source():
    # Source 0's best is 200; source 1's best is 50 → source 1 is weakest.
    ms = [
        _mission(0, 10, 5, 200.0),
        _mission(0, 11, 5, 150.0),
        _mission(1, 20, 5, 50.0),
        _mission(1, 21, 5, 30.0),
    ]
    out = _drop_weakest_source(ms)
    assert out is not None
    assert all(m.src_id != 1 for m in out)
    assert {m.src_id for m in out} == {0}


# ---------------------------------------------------------------------------
# generate_portfolios — end-to-end on fixture obs
# ---------------------------------------------------------------------------


def _make_world_and_model(obs_dict):
    """Build a World + WorldModel from an obs dict for end-to-end tests."""
    world = World.from_obs(obs_dict)
    model = WorldModel.from_world(world)
    return world, model


def _two_planet_2p_obs():
    """Minimal 2-player obs: one home each, one neutral target."""
    return {
        "player": 0,
        "angular_velocity": 0.0,
        "step": 10,
        "planets": [
            # [id, owner, x, y, radius, ships, production]
            [0, 0, 10.0, 10.0, 3.0, 50, 3.0],   # our home
            [1, 1, 90.0, 90.0, 3.0, 50, 3.0],   # opp home
            [2, -1, 50.0, 50.0, 3.0, 5, 2.0],    # neutral
        ],
        "fleets": [],
        "comet_planet_ids": [],
    }


def test_generate_portfolios_incumbent_is_first():
    obs = _two_planet_2p_obs()
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    assert portfolios[0].label == "incumbent"


def test_generate_portfolios_includes_noop():
    obs = _two_planet_2p_obs()
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    labels = [p.label for p in portfolios]
    assert "noop" in labels
    noop = next(p for p in portfolios if p.label == "noop")
    assert noop.missions == []


def test_generate_portfolios_caps_at_5():
    """No matter what the world looks like, we generate ≤ 5 portfolios."""
    obs = _two_planet_2p_obs()
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    assert len(portfolios) <= 5


def test_generate_portfolios_empty_world():
    """No planets → only the incumbent (empty) and the noop, which collapse
    to a single emit. We still return ≥ 1 portfolio so the agent has
    something to score."""
    obs = {
        "player": 0,
        "angular_velocity": 0.0,
        "step": 0,
        "planets": [],
        "fleets": [],
        "comet_planet_ids": [],
    }
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    assert len(portfolios) >= 1
    assert portfolios[0].label == "incumbent"


def test_generate_portfolios_labels_are_unique():
    obs = _two_planet_2p_obs()
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    labels = [p.label for p in portfolios]
    assert len(labels) == len(set(labels))


def test_generate_portfolios_conservative_differs_when_source_is_fat():
    """When a source has > AGGRESSIVE_MIN_GARRISON ships, snipe(aggressive=
    True) emits a fat-fleet mission and snipe(aggressive=False) emits the
    minimum-viable size. They should produce different portfolios."""
    obs = _two_planet_2p_obs()
    # Bump home garrison from 50 to ensure aggressive picks the fat path.
    obs["planets"][0] = [0, 0, 10.0, 10.0, 3.0, 100, 3.0]
    world, model = _make_world_and_model(obs)
    portfolios = generate_portfolios(world, model)
    labels = [p.label for p in portfolios]
    assert "conservative" in labels
