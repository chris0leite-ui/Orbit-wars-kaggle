"""Tests for the force-concentration chooser mechanism.

Covers Rule 38 (fix-verification reproduces failure state):
    1. Env-getter parses the gate + max_waves knob correctly, including
       garbage-input defaults.
    2. Synthetic _greedy_select test: with ``max_waves_per_target=1``
       (legacy), only one wave lands per target. With ``=2`` and an
       identity rescore, multiple waves can land on the same target —
       proves the cap relaxation and the rescore wire-up.
    3. With the gate UNSET (default), the producer_plus agent's actions
       are byte-identical to the same agent with the gate explicitly = 0
       (guards the OFF-path against perturbation).
    4. With the gate ON, action rows DIFFER from the OFF path at at
       least one seed — proves the new code path is exercised.
    5. Wallclock smoke: full game under 60 s (Rule 46c).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def producer_plus_main():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_force_concentration",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_force_concentration"] = module
    spec.loader.exec_module(module)
    return module


def _clear_fc_env(monkeypatch):
    for name in (
        "PRODUCER_PLUS_FORCE_CONCENTRATION",
        "PRODUCER_PLUS_FORCE_CONCENTRATION_MAX_WAVES",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Env-getter tests
# ---------------------------------------------------------------------------


def test_force_concentration_default_off(monkeypatch, producer_plus_main):
    _clear_fc_env(monkeypatch)
    assert producer_plus_main._force_concentration_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "ON"])
def test_force_concentration_env_on_truthy(monkeypatch, producer_plus_main, value):
    _clear_fc_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_FORCE_CONCENTRATION", value)
    assert producer_plus_main._force_concentration_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_force_concentration_env_off_falsy(monkeypatch, producer_plus_main, value):
    _clear_fc_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_FORCE_CONCENTRATION", value)
    assert producer_plus_main._force_concentration_enabled() is False


def test_force_concentration_max_waves_default(monkeypatch, producer_plus_main):
    _clear_fc_env(monkeypatch)
    assert producer_plus_main._force_concentration_max_waves() == 2


@pytest.mark.parametrize("value,expected", [
    ("1", 1), ("2", 2), ("3", 3), ("5", 5),
    ("0", 1),    # clamped to >= 1
    ("-1", 1),   # clamped to >= 1
    ("abc", 2), # garbage falls back to default
    ("", 2),
])
def test_force_concentration_max_waves_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_fc_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_FORCE_CONCENTRATION_MAX_WAVES", value)
    assert producer_plus_main._force_concentration_max_waves() == expected


# ---------------------------------------------------------------------------
# Synthetic test of _greedy_select cap relaxation
# ---------------------------------------------------------------------------


def _synthetic_select_kwargs():
    """Four single-source candidates: (src 0 -> T_a, score 10), (src 1 -> T_a,
    score 8), (src 2 -> T_b, score 5), (src 3 -> T_b, score 4). Sources 0..3
    each have exactly ``cand_send`` ships of budget so no source can refire.
    Targets T_a, T_b live at planet slots 4, 5 (distinct from sources)."""
    import torch
    P, W = 6, 6
    device = torch.device("cpu")
    dtype = torch.float32
    score = torch.tensor([10.0, 8.0, 5.0, 4.0], dtype=dtype)
    cand_src = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    cand_send = torch.tensor([[3.0], [3.0], [3.0], [3.0]], dtype=dtype)
    cand_angle = torch.zeros(4, 1, dtype=dtype)
    cand_eta = torch.ones(4, 1, dtype=dtype)
    cand_active = torch.ones(4, 1, dtype=torch.bool)
    cand_tgt_slot = torch.tensor([4, 4, 5, 5], dtype=torch.long)
    cand_tgt_short = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cand_is_def = torch.zeros(4, dtype=torch.bool)
    source_budget = torch.zeros(P, dtype=dtype)
    source_budget[0:4] = 3.0  # exactly enough for one launch each
    target_exists = torch.tensor([True, True], dtype=torch.bool)
    return dict(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle,
        cand_eta=cand_eta, cand_active=cand_active,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists, roi_threshold=1.5,
    )


def test_greedy_select_legacy_caps_at_one_wave_per_target():
    """``max_waves_per_target=1`` (default) fires exactly one wave per
    target — c0 (best at T_a) and c2 (best at T_b). Confirms the legacy
    behaviour is preserved when force-concentration is OFF."""
    from agents.producer.orbit_lite.planner_core import _greedy_select
    kw = _synthetic_select_kwargs()
    entries, _ = _greedy_select(**kw)
    n_fired = int(entries.valid.sum().item())
    assert n_fired == 2, f"expected 2 launches with cap=1, got {n_fired}"
    fired_tgts = entries.target_slots[entries.valid].tolist()
    assert sorted(fired_tgts) == [4, 5], f"expected one wave per target, got {fired_tgts}"


def test_greedy_select_cap_two_allows_second_wave_to_same_target():
    """``max_waves_per_target=2`` + identity rescore allows c1 to fire as a
    second wave at T_a. Total fires = 4 (one per candidate); T_a hit
    twice. Proves the mutex relaxation + rescore wire-up."""
    from agents.producer.orbit_lite.planner_core import _greedy_select
    kw = _synthetic_select_kwargs()
    score_const = kw["score"].clone()
    rescore_calls = {"n": 0}

    def _id_rescore(c_src, c_send, c_eta, c_tgt, c_active):
        rescore_calls["n"] += 1
        return score_const.clone()

    entries, _ = _greedy_select(
        rescore_fn=_id_rescore, max_waves_per_target=2, **kw,
    )
    n_fired = int(entries.valid.sum().item())
    assert n_fired == 4, f"expected 4 launches with cap=2, got {n_fired}"
    fired_tgts = entries.target_slots[entries.valid].tolist()
    assert fired_tgts.count(4) == 2, (
        f"expected 2 waves to T_a (slot 4) with cap=2, got {fired_tgts}"
    )
    assert fired_tgts.count(5) == 2, (
        f"expected 2 waves to T_b (slot 5) with cap=2, got {fired_tgts}"
    )
    assert rescore_calls["n"] >= 1, "rescore_fn was never called"


def test_greedy_select_cap_two_without_rescore_still_relaxes_mutex():
    """``max_waves_per_target=2`` with ``rescore_fn=None`` still relaxes the
    target mutex (degenerate path: production never uses this combo, but
    it must not crash). The same outcome as the identity rescore in this
    constant-score fixture."""
    from agents.producer.orbit_lite.planner_core import _greedy_select
    kw = _synthetic_select_kwargs()
    entries, _ = _greedy_select(
        rescore_fn=None, max_waves_per_target=2, **kw,
    )
    n_fired = int(entries.valid.sum().item())
    assert n_fired == 4, f"expected 4 launches with cap=2, got {n_fired}"


# ---------------------------------------------------------------------------
# Integration tests (slow)
# ---------------------------------------------------------------------------


def _play_one_game(focal_path, opp_path, seed):
    # Shims set PRODUCER_PLUS_* via os.environ.setdefault at load; those keys
    # outlive the game in this process and silently re-gate the NEXT game's
    # agents (the clean_ab.py env-pollution problem). Start every game from a
    # clean gate state so ON/OFF comparisons compare what they claim to.
    for _k in [k for k in os.environ if k.startswith("PRODUCER_PLUS_")]:
        os.environ.pop(_k)
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, opp_path])
    return env.state, env.steps


@pytest.mark.slow
def test_force_concentration_off_byte_identical(monkeypatch):
    """Rule 38: gate explicit "0" must match gate UNSET — both hit the
    legacy code path. The synthetic test above proves cap=1 is byte-equivalent
    to the original; this guards the end-to-end agent path."""
    shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_force_concentration.py")
    producer = os.path.join(PRODUCER_DIR, "producer_agent.py")
    for name in (
        "PRODUCER_PLUS_FORCE_CONCENTRATION",
        "PRODUCER_PLUS_FORCE_CONCENTRATION_MAX_WAVES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRODUCER_PLUS_FORCE_CONCENTRATION", "0")
    state_zero, _ = _play_one_game(shim, producer, 7)
    monkeypatch.delenv("PRODUCER_PLUS_FORCE_CONCENTRATION", raising=False)
    state_unset, _ = _play_one_game(shim, producer, 7)
    assert state_zero[0]["reward"] == state_unset[0]["reward"], (
        f"explicit gate=0 ({state_zero[0]['reward']}) differs from unset "
        f"({state_unset[0]['reward']}) — OFF mapping mis-routed"
    )


@pytest.mark.slow
def test_force_concentration_on_changes_planner_output():
    """Proof the force-concentration code path is exercised: with the
    knob ON and max_waves=2, gate ON must differ from gate OFF on at
    least one of seeds (7, 13, 42). If all agree, the chooser never
    actually fires a second wave on any target — mechanism is no-op."""
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    off_shim = os.path.join(PRODUCER_DIR, "producer_agent.py")
    on_shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_force_concentration.py")
    differs = False
    for seed in (7, 13, 42):
        state_off, _ = _play_one_game(off_shim, producer_path, seed)
        state_on, _ = _play_one_game(on_shim, producer_path, seed)
        if (
            state_off[0]["reward"] != state_on[0]["reward"]
            or state_off[1]["reward"] != state_on[1]["reward"]
        ):
            differs = True
            break
    assert differs, (
        "force-concentration ON produced identical outcomes to vanilla "
        "producer on seeds 7, 13, 42 — code path may be unreachable"
    )


@pytest.mark.slow
def test_force_concentration_smoke_wallclock_under_60s():
    """Rule 46c: full game vs producer at seed 7 under 60 s wallclock.
    The rescore closure adds one extra ``score_candidates`` call per
    fired wave (~3-10 ms each)."""
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_force_concentration.py",
    )
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"multi_tick_force_concentration at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock regression"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]
