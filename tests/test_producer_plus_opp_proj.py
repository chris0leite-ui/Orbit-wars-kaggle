"""Unit tests for the opp-projection registry + translator in ``producer_plus``.

Covers Step 3 of ``state/MIGRATION_PLAN.md``: the bit-identical default,
the registry lookup, and the tuple-to-bucket-delta translation.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def opp_projector():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_opp_projector_test",
        os.path.join(PRODUCER_PLUS_DIR, "opp_projector.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_opp_projector_test"] = module
    spec.loader.exec_module(module)
    return module


def test_none_projector_returns_empty(opp_projector):
    obs = {"player": 0, "planets": [], "fleets": [], "step": 0}
    assert opp_projector._none_projector(obs, my_id=0, num_seats=2, horizon=8) == []


def test_get_projector_default_is_none(opp_projector):
    assert opp_projector.get_projector(None) is opp_projector._none_projector
    assert opp_projector.get_projector("none") is opp_projector._none_projector


def test_get_projector_unknown_falls_back_to_none(opp_projector):
    """Misconfigured env var must never raise — degrade to no projection."""
    assert opp_projector.get_projector("garbage_name") is opp_projector._none_projector
    assert opp_projector.get_projector("") is opp_projector._none_projector


def test_get_projector_lite_greedy_registered(opp_projector):
    assert opp_projector.get_projector("lite_greedy") is opp_projector._lite_greedy_projector
    assert opp_projector.get_projector("LITE_GREEDY") is opp_projector._lite_greedy_projector
    assert opp_projector.get_projector("  lite_greedy  ") is opp_projector._lite_greedy_projector


def test_lite_greedy_projector_swallows_errors(opp_projector):
    """Malformed obs must return [] rather than crash the agent."""
    assert opp_projector._lite_greedy_projector(
        None, my_id=0, num_seats=2, horizon=8,
    ) == []
    assert opp_projector._lite_greedy_projector(
        {"player": 0}, my_id=0, num_seats=2, horizon=8,
    ) == []


def _planet_ids(*ids: int) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.long)


def test_translator_empty_inputs(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta.shape == (3, 8, 2)
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_basic_mapping(opp_projector):
    # tgt_id=7 lives at slot index 1; eta=3 → bucket index 2.
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta[1, 2, 1].item() == pytest.approx(12.0)
    # Everything else must remain zero.
    delta[1, 2, 1] = 0.0
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_out_of_window(opp_projector):
    """eta > H is outside the scoring forecast — must drop, not overflow."""
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 99, 1, 12.0), (7, 0, 1, 12.0), (7, -1, 1, 12.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_unknown_planet(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(42, 3, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_bad_owner(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, -1, 12.0), (7, 3, 5, 12.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_nonpositive_ships(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 0.0), (7, 3, 1, -5.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_sums_duplicates(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 4.0), (7, 3, 1, 6.0), (7, 3, 1, 2.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta[1, 2, 1].item() == pytest.approx(12.0)


def test_translator_h_zero(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 1, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=0,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta.shape == (3, 0, 2)


def test_translator_handles_invalid_planet_id_in_obs(opp_projector):
    """planet_ids may contain ``-1`` padding slots — those must be ignored."""
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 12.0)], _planet_ids(5, -1, 7, 9, -1), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    # tgt_id=7 → slot index 2 (skipping the -1 padding row, but the dict still uses raw slot indices).
    assert delta[2, 2, 1].item() == pytest.approx(12.0)
