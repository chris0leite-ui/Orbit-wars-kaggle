"""Tests for the neutral shortlist quota (PRODUCER_PLUS_NEUTRAL_SHORTLIST)."""
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
        "producer_plus_main_test_nq",
        os.path.join(REPO_ROOT, "agents", "producer_plus", "main.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_nq"] = m
    spec.loader.exec_module(m)
    return m


def test_quota_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_NEUTRAL_SHORTLIST", raising=False)
    assert pp_main._neutral_shortlist_quota() == 0


def test_appends_missing_neutrals(monkeypatch, pp_main):
    # P=5: 0 mine, 1-2 enemy (shortlisted), 3-4 neutral (3 near, 4 far).
    obs = SimpleNamespace(
        P=5, device=torch.device("cpu"),
        is_neutral=torch.tensor([False, False, False, True, True]),
        alive=torch.ones(5, dtype=torch.bool),
    )
    K = 6
    d = torch.full((K + 1, 5, 5), 50.0)
    d[:, 0, 3] = 5.0   # neutral 3 near our planet 0
    d[:, 0, 4] = 30.0  # neutral 4 far
    cache = SimpleNamespace(cross_dist=d, P=5, K=K, alive_by_step=torch.ones(K + 1, 5, dtype=torch.bool), device=torch.device("cpu"))
    source_mask = torch.tensor([True, False, False, False, False])
    tgt = torch.tensor([1, 2])
    ex = torch.tensor([True, True])
    out_idx, out_ex = pp_main._append_neutral_quota(
        tgt, ex, obs=obs, cache=cache, source_mask=source_mask, K_eta=K, quota=1)
    assert out_idx.tolist() == [1, 2, 3]      # nearest neutral appended
    assert out_ex.tolist() == [True, True, True]


def test_dedupes_already_shortlisted(monkeypatch, pp_main):
    obs = SimpleNamespace(
        P=4, device=torch.device("cpu"),
        is_neutral=torch.tensor([False, False, True, True]),
        alive=torch.ones(4, dtype=torch.bool),
    )
    K = 6
    d = torch.full((K + 1, 4, 4), 50.0)
    d[:, 0, 2] = 5.0
    cache = SimpleNamespace(cross_dist=d, P=4, K=K, alive_by_step=torch.ones(K + 1, 4, dtype=torch.bool), device=torch.device("cpu"))
    source_mask = torch.tensor([True, False, False, False])
    tgt = torch.tensor([2])                    # nearest neutral already listed
    ex = torch.tensor([True])
    out_idx, out_ex = pp_main._append_neutral_quota(
        tgt, ex, obs=obs, cache=cache, source_mask=source_mask, K_eta=K, quota=1)
    assert out_idx.tolist() == [2, 2]
    assert out_ex.tolist() == [True, False]    # duplicate appended as INVALID
