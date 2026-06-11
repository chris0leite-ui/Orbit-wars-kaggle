"""Tests for the in-agent opening searcher (PRODUCER_PLUS_OPENING_SEARCH).

Beam search over neutral-capture schedules during the pre-contact opening
(ported from scripts/opening_optimum.py), emitting launches due now. The
search itself is pure Python over the current observation; these tests
exercise it on synthetic boards.
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

for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_opening",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_opening"] = module
    spec.loader.exec_module(module)
    return module


def _obs_tensors(planet_rows):
    """planet_rows: [pid, owner, x, y, r, ships, prod] per planet."""
    return {
        "planets": torch.tensor(planet_rows, dtype=torch.float32),
        "angular_velocity": torch.tensor([0.0]),
        "step": torch.tensor([0]),
        "player": torch.tensor([0]),
    }


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_OPENING_SEARCH", raising=False)
    assert pp_main._opening_search_window() == 0


def test_captures_affordable_neutral_now(pp_main):
    # My planet (30 ships, prod 1) at (40,50); neutral garrison 10 nearby at
    # (60,50); enemy far away at (90,50). Search should launch now: 11 ships
    # (garrison+1) from planet 0 at planet 1.
    obs_tensors = _obs_tensors([
        [0, 0, 40.0, 50.0, 1.0, 30.0, 1.0],
        [1, -1, 60.0, 50.0, 1.0, 10.0, 2.0],
        [2, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
    ])
    due = pp_main._opening_search_plan(
        obs_tensors, pid=0, claimed=set(), horizon=40, beam_width=32,
    )
    assert due, "expected an immediate capture launch"
    src, tgt, size = due[0]
    assert (src, tgt) == (0, 1)
    assert size == pytest.approx(11.0)


def test_skips_neutrals_enemy_reaches_first(pp_main):
    # The only neutral is right next to the enemy: race lost, no launch.
    obs_tensors = _obs_tensors([
        [0, 0, 10.0, 50.0, 1.0, 30.0, 1.0],
        [1, -1, 85.0, 50.0, 1.0, 10.0, 2.0],
        [2, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
    ])
    due = pp_main._opening_search_plan(
        obs_tensors, pid=0, claimed=set(), horizon=40, beam_width=32,
    )
    assert due == []


def test_claimed_neutrals_excluded(pp_main):
    obs_tensors = _obs_tensors([
        [0, 0, 40.0, 50.0, 1.0, 30.0, 1.0],
        [1, -1, 60.0, 50.0, 1.0, 10.0, 2.0],
        [2, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
    ])
    due = pp_main._opening_search_plan(
        obs_tensors, pid=0, claimed={1}, horizon=40, beam_width=32,
    )
    assert due == []


def test_waits_when_garrison_insufficient(pp_main):
    # Garrison 5 < needed 12: the optimal first launch is later, nothing due
    # NOW.
    obs_tensors = _obs_tensors([
        [0, 0, 40.0, 50.0, 1.0, 5.0, 1.0],
        [1, -1, 60.0, 50.0, 1.0, 10.0, 2.0],
        [2, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
    ])
    due = pp_main._opening_search_plan(
        obs_tensors, pid=0, claimed=set(), horizon=40, beam_width=32,
    )
    assert due == []


def test_prefers_high_production_when_competing(pp_main):
    # Two equally-near neutrals, one prod 4 / one prod 1, garrison such that
    # only one is affordable now: the schedule's first launch goes at the
    # high-production one.
    obs_tensors = _obs_tensors([
        [0, 0, 50.0, 50.0, 1.0, 12.0, 1.0],
        [1, -1, 60.0, 50.0, 1.0, 10.0, 1.0],
        [2, -1, 40.0, 50.0, 1.0, 10.0, 4.0],
        [3, 1, 50.0, 95.0, 1.0, 30.0, 1.0],
    ])
    due = pp_main._opening_search_plan(
        obs_tensors, pid=0, claimed=set(), horizon=40, beam_width=64,
    )
    assert due
    assert due[0][1] == 2          # the prod-4 neutral


def test_hold_filter_skips_unaffordable_launches(pp_main):
    # A capture wave is all-or-nothing: sources whose safe drain can't fund
    # the full size are skipped, never clamped.
    rows = [(0, 5, 11.0), (2, 6, 20.0), (3, 7, 8.0)]
    drain = {0: 15.0, 2: 12.0, 3: 8.0}
    kept = pp_main._opening_hold_filter(rows, drain)
    assert kept == [(0, 5, 11.0), (3, 7, 8.0)]


def test_hold_filter_unknown_source_skipped(pp_main):
    assert pp_main._opening_hold_filter([(9, 5, 3.0)], {}) == []


def test_reserve_filter_blocks_strippable_sources(pp_main):
    # Source 0 has 30 ships, reserve 25: an 11-ship launch would dip to 19
    # < 25 -> blocked. Source 3 has 30, reserve 5 -> 11-ship launch fine.
    rows = [(0, 5, 11.0), (3, 7, 11.0)]
    ships = {0: 30.0, 3: 30.0}
    reserve = {0: 25.0, 3: 5.0}
    assert pp_main._opening_reserve_filter(rows, ships, reserve) == [(3, 7, 11.0)]


def test_reserve_k_default(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_OPENING_RESERVE_K", raising=False)
    assert pp_main._opening_reserve_k() == 8
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_RESERVE_K", "0")
    assert pp_main._opening_reserve_k() == 0
