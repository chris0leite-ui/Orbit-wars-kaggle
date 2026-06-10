"""Tests for the reactive floor (PRODUCER_PLUS_REACTIVE_FLOOR): enemy floors
include the defense the opponent can ROUTE to the target within the flight."""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO_ROOT, "agents", "producer"),
          os.path.join(REPO_ROOT, "agents", "producer_plus")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_rf",
        os.path.join(REPO_ROOT, "agents", "producer_plus", "main.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_rf"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(pp_main, reserve_dist=6.0):
    """P=3: planet 0 mine, planet 1 enemy target, planet 2 enemy reserve
    (100 ships) at distance `reserve_dist` from the target."""
    obs = SimpleNamespace(
        player_id=0, P=3, device=torch.device("cpu"),
        ships=torch.tensor([30.0, 5.0, 100.0]),
        owned=torch.tensor([True, False, False]),
        is_enemy=torch.tensor([False, True, True]),
        alive=torch.tensor([True, True, True]),
    )
    K = 8
    d = torch.zeros(K + 1, 3, 3)
    d[:, 2, 1] = reserve_dist
    d[:, 1, 2] = reserve_dist
    d[:, 0, 1] = 10.0
    cache = SimpleNamespace(cross_dist=d)
    return obs, cache, K


def test_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_REACTIVE_FLOOR", raising=False)
    assert pp_main._reactive_floor_weight() == 0.0


def test_margin_counts_reachable_reserve(pp_main):
    # Reserve of 100 at distance 6, speed(100) ~ 3.36 -> eta ~1.8; with lag 2
    # it reaches the target for arrivals k >= 4. Early arrivals (k <= 3) see
    # no support; later ones see the full 100.
    obs, cache, K = _fixture(pp_main)
    m = pp_main._reactive_reinforcement_margin(
        obs, cache, torch.tensor([1]), K, weight=1.0)
    assert m is not None and m.shape == (1, K)
    assert m[0, 0].item() == 0.0 and m[0, 2].item() == 0.0
    assert m[0, K - 1].item() == 100.0


def test_margin_excludes_target_itself(pp_main):
    # Target = the reserve planet (q == t): its own garrison must not be
    # double-counted as routed support.
    obs, cache, K = _fixture(pp_main)
    m = pp_main._reactive_reinforcement_margin(
        obs, cache, torch.tensor([2]), K, weight=1.0)
    # Only the OTHER enemy planet (5 ships at distance 6) can support it.
    assert m[0, K - 1].item() == 5.0


def test_weight_scales(pp_main):
    obs, cache, K = _fixture(pp_main)
    m = pp_main._reactive_reinforcement_margin(
        obs, cache, torch.tensor([1]), K, weight=0.5)
    assert m[0, K - 1].item() == 50.0
