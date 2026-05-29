"""Unit tests for `lib.value_head_features.encode_features` (Reframe B.2).

The encoder is a pure function — `(src, tgt, ships, eta, me, world,
world_model) → ndarray(14,)`. These tests pin:

  - feature dim
  - owner one-hot mapping (me / neutral / enemy)
  - combat margin clipping at ±1
  - sun-distance computation (lib.geometry.CENTER = 50)
  - opp-centroid handling when no enemy planets exist
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.value_head_features import (
    FEATURE_DIM_BASE,
    FEATURE_NAMES,
    SUN_CENTER,
    encode_features,
)
from lib.world_model import WorldModel


def _make_world(planets, *, focal=0, step=10):
    obs = {
        "player": focal,
        "planets": [tuple(p) for p in planets],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    w = World.from_obs(obs)
    w.obs_raw = obs
    return w


def test_feature_dim_is_14():
    p_src = Planet(0, 0, 30.0, 50.0, 1.5, 100, 3)
    p_tgt = Planet(1, -1, 50.0, 50.0, 1.5, 50, 2)
    w = _make_world([p_src, p_tgt])
    m = WorldModel.from_world(w)
    feats = encode_features(p_src, p_tgt, ships=80, eta=5, me=0,
                            world=w, world_model=m)
    assert feats.shape == (FEATURE_DIM_BASE,) == (14,)
    assert feats.dtype == np.float32
    # leaf_delta is feats[14] in the FULL vector — encoder owns [0..13].
    assert "leaf_delta" == FEATURE_NAMES[14]


def test_owner_at_launch_one_hot_me_neutral_enemy():
    p_me = Planet(0, 0, 30.0, 50.0, 1.5, 100, 3)
    p_n = Planet(1, -1, 50.0, 50.0, 1.5, 50, 2)
    p_e = Planet(2, 1, 70.0, 50.0, 1.5, 80, 2)
    w = _make_world([p_me, p_n, p_e])
    m = WorldModel.from_world(w)

    f_me = encode_features(p_me, p_me, ships=50, eta=0, me=0,
                           world=w, world_model=m)
    f_n = encode_features(p_me, p_n, ships=50, eta=5, me=0,
                          world=w, world_model=m)
    f_e = encode_features(p_me, p_e, ships=200, eta=8, me=0,
                          world=w, world_model=m)

    # Indices 2/3/4 = launch one-hot me/neutral/enemy.
    assert (f_me[2], f_me[3], f_me[4]) == (1.0, 0.0, 0.0)
    assert (f_n[2], f_n[3], f_n[4]) == (0.0, 1.0, 0.0)
    assert (f_e[2], f_e[3], f_e[4]) == (0.0, 0.0, 1.0)


def test_combat_margin_clips_at_plus_minus_one():
    p_src = Planet(0, 0, 30.0, 50.0, 1.5, 100, 3)
    # Tiny enemy garrison: 10 ships. Send 1000. Raw margin = (1000-10)/10
    # = 99. Must clip to +1.
    p_tgt_tiny = Planet(1, 1, 50.0, 50.0, 1.5, 10, 0)
    # Huge enemy garrison: 5000 ships. Send 1. Raw margin = (1-5000)/5000
    # = -0.9998. Won't clip — confirms the +1 clip side is the load-
    # bearing one. Send 0 ships against a 5000 garrison → margin <= -1.
    p_tgt_huge = Planet(2, 1, 70.0, 50.0, 1.5, 5000, 0)
    w = _make_world([p_src, p_tgt_tiny, p_tgt_huge])
    m = WorldModel.from_world(w)

    f_overshoot = encode_features(p_src, p_tgt_tiny, ships=1000, eta=3,
                                  me=0, world=w, world_model=m)
    assert f_overshoot[8] == pytest.approx(1.0)

    f_undershoot = encode_features(p_src, p_tgt_huge, ships=0, eta=3,
                                   me=0, world=w, world_model=m)
    assert f_undershoot[8] == pytest.approx(-1.0)


def test_src_distance_to_sun_uses_center_50():
    # Src at (50, 50) = sun position → distance 0.
    p_at_sun = Planet(0, 0, SUN_CENTER, SUN_CENTER, 1.5, 100, 3)
    p_tgt = Planet(1, -1, 30.0, 50.0, 1.5, 30, 2)
    w = _make_world([p_at_sun, p_tgt])
    m = WorldModel.from_world(w)
    feats = encode_features(p_at_sun, p_tgt, ships=40, eta=4, me=0,
                            world=w, world_model=m)
    assert feats[11] == pytest.approx(0.0, abs=1e-5)
    # Tgt at (30, 50): distance = 20.
    assert feats[12] == pytest.approx(20.0, abs=1e-4)


def test_opp_centroid_zero_when_no_enemy_planets():
    # Only me + neutral. Opp centroid undefined → distance set to 0.
    p_me = Planet(0, 0, 30.0, 50.0, 1.5, 100, 3)
    p_n = Planet(1, -1, 50.0, 50.0, 1.5, 50, 2)
    w = _make_world([p_me, p_n])
    m = WorldModel.from_world(w)
    feats = encode_features(p_me, p_n, ships=60, eta=5, me=0,
                            world=w, world_model=m)
    assert feats[13] == pytest.approx(0.0)


def test_production_features_are_raw():
    p_src = Planet(0, 0, 30.0, 50.0, 1.5, 100, 4)  # production=4
    p_tgt = Planet(1, -1, 50.0, 50.0, 1.5, 50, 2)  # production=2
    w = _make_world([p_src, p_tgt])
    m = WorldModel.from_world(w)
    feats = encode_features(p_src, p_tgt, ships=60, eta=5, me=0,
                            world=w, world_model=m)
    assert feats[9] == pytest.approx(4.0)
    assert feats[10] == pytest.approx(2.0)


def test_ships_and_eta_are_raw():
    p_src = Planet(0, 0, 30.0, 50.0, 1.5, 100, 3)
    p_tgt = Planet(1, -1, 50.0, 50.0, 1.5, 50, 2)
    w = _make_world([p_src, p_tgt])
    m = WorldModel.from_world(w)
    feats = encode_features(p_src, p_tgt, ships=137, eta=22, me=0,
                            world=w, world_model=m)
    assert feats[0] == pytest.approx(137.0)
    assert feats[1] == pytest.approx(22.0)
