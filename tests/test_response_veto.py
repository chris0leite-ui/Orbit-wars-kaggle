"""Tests for the response veto (PRODUCER_PLUS_RESPONSE_VETO).

The mechanism: after our waves are chosen, one extra mirror pass with OUR
launches as background yields the opponent's predicted reply; attack waves
whose flow score under that reply is worse than doing nothing are dropped.
Default OFF leaves entries untouched (run_turn never calls the helper).

The mirror itself is monkeypatched here — these tests cover the veto's
scoring/filtering logic against the exact sparse scorer on synthetic boards.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")

for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass(frozen=True)
class _Cfg:
    roi_threshold: float = 1.5


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_veto",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_veto"] = module
    spec.loader.exec_module(module)
    return module


def test_env_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_RESPONSE_VETO", raising=False)
    assert pp_main._response_veto_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_RESPONSE_VETO", value)
    assert pp_main._response_veto_enabled() is True


def _board(pp_main):
    """P=3, A=2, H=4. Planet 0 mine (20 ships, prod 1), planet 1 enemy
    (garrison 5, prod 2), planet 2 enemy reserve (30 ships, prod 1)."""
    from orbit_lite.movement import PlanetGarrisonStatus

    H = 4
    dtype = torch.float32
    owner = torch.tensor(
        [[0] * (H + 1), [1] * (H + 1), [1] * (H + 1)], dtype=torch.long)
    ships = torch.tensor(
        [[20.0] * (H + 1), [5.0] * (H + 1), [30.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner, ships=ships,
        pre_combat_owner=owner.clone(), pre_combat_ships=ships.clone(),
        arrivals_by_owner=torch.zeros(3, H + 1, 2, dtype=dtype),
    )
    obs = SimpleNamespace(
        player_id=0, P=3, device=torch.device("cpu"),
        ships=torch.tensor([20.0, 5.0, 30.0], dtype=dtype),
        owned=torch.tensor([True, False, False]),
        is_enemy=torch.tensor([False, True, True]),
        alive=torch.tensor([True, True, True]),
    )
    prod = torch.tensor([1.0, 2.0, 1.0], dtype=dtype)
    alive_by_step = torch.ones(H + 1, 3, dtype=torch.bool)
    return status, obs, prod, alive_by_step, H


def _entries(pp_main, ships, tgt=1):
    """One attack wave P0 -> tgt, eta 2."""
    from orbit_lite.movement_step import LaunchEntries
    dtype = torch.float32
    return LaunchEntries(
        source_slots=torch.tensor([0]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        angle=torch.tensor([0.0], dtype=dtype),
        eta=torch.tensor([2.0], dtype=dtype),
        valid=torch.tensor([True]),
    )


def _stub_reply(pp_main, ships, eta=2.0, tgt=1):
    """Opponent reply: planet 2 reinforces `tgt` with `ships` at `eta`."""
    from orbit_lite.garrison_launch import LaunchSet
    dtype = torch.float32
    L = max(1, 1)
    return LaunchSet(
        source_slots=torch.tensor([2]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        eta=torch.tensor([float(eta)], dtype=dtype),
        owner=torch.tensor([1]),
        valid=torch.tensor([ships > 0]),
    )


def _run_veto(pp_main, monkeypatch, attack_ships, reply_ships, margin=None):
    status, obs, prod, alive_by_step, H = _board(pp_main)
    entries = _entries(pp_main, attack_ships)
    reply = _stub_reply(pp_main, reply_ships)
    monkeypatch.setattr(
        pp_main, "predict_opp_launches_via_mirror", lambda **kw: reply)
    if margin is not None:
        monkeypatch.setenv("PRODUCER_PLUS_RESPONSE_VETO_MARGIN", str(margin))
    out = pp_main._apply_response_veto(
        entries,
        movement=None, obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )
    return out


def test_unparried_capture_survives(monkeypatch, pp_main):
    # 10 vs garrison 5, no reply: profitable capture stays valid.
    out = _run_veto(pp_main, monkeypatch, attack_ships=10.0, reply_ships=0.0)
    assert out.valid.tolist() == [True]


def test_parried_attack_vetoed(monkeypatch, pp_main):
    # 10 vs garrison 5 but a 30-ship reply lands the same tick: our fleet
    # annihilates against the reply (top1−top2) and the capture fails —
    # strictly worse than holding the 10 ships home. Veto.
    out = _run_veto(pp_main, monkeypatch, attack_ships=10.0, reply_ships=30.0)
    assert out.valid.tolist() == [False]


def test_attack_through_parry_survives(monkeypatch, pp_main):
    # 20 vs garrison 5 with only a 6-ship reply: we still flip the planet
    # and keep its production; better than doing nothing. Keep.
    out = _run_veto(pp_main, monkeypatch, attack_ships=20.0, reply_ships=6.0)
    assert out.valid.tolist() == [True]


def test_own_target_transfers_untouched(monkeypatch, pp_main):
    # Waves at OUR OWN planets (reinforce/regroup lane) are never vetoed.
    status, obs, prod, alive_by_step, H = _board(pp_main)
    entries = _entries(pp_main, 10.0, tgt=0)
    called = {"n": 0}

    def _no_mirror(**kw):
        called["n"] += 1
        return _stub_reply(pp_main, 0.0)

    monkeypatch.setattr(pp_main, "predict_opp_launches_via_mirror", _no_mirror)
    out = pp_main._apply_response_veto(
        entries,
        movement=None, obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )
    assert out.valid.tolist() == [True]
    assert called["n"] == 0  # no attack waves -> mirror never even runs


# ---------------------------------------------------------------------------
# Upsize ("beat the parry", PRODUCER_PLUS_RESPONSE_VETO_UPSIZE)
# ---------------------------------------------------------------------------


def _board_rich(pp_main, my_ships=60.0, tgt_prod=5.0):
    """Like _board but with a configurable home garrison and target prod."""
    from orbit_lite.movement import PlanetGarrisonStatus

    H = 4
    dtype = torch.float32
    owner = torch.tensor(
        [[0] * (H + 1), [1] * (H + 1), [1] * (H + 1)], dtype=torch.long)
    ships = torch.tensor(
        [[my_ships] * (H + 1), [5.0] * (H + 1), [30.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner, ships=ships,
        pre_combat_owner=owner.clone(), pre_combat_ships=ships.clone(),
        arrivals_by_owner=torch.zeros(3, H + 1, 2, dtype=dtype),
    )
    obs = SimpleNamespace(
        player_id=0, P=3, device=torch.device("cpu"),
        ships=torch.tensor([my_ships, 5.0, 30.0], dtype=dtype),
        owned=torch.tensor([True, False, False]),
        is_enemy=torch.tensor([False, True, True]),
        alive=torch.tensor([True, True, True]),
    )
    prod = torch.tensor([1.0, tgt_prod, 1.0], dtype=dtype)
    alive_by_step = torch.ones(H + 1, 3, dtype=torch.bool)
    return status, obs, prod, alive_by_step, H


def _stub_aim(angle=0.0, eta=2.0, viable=True):
    def aim(movement, src, tgt, sizes, **kw):
        n = int(src.shape[0])
        return {
            "angle": torch.full((n,), float(angle)),
            "eta": torch.full((n,), float(eta)),
            "viable": torch.full((n,), bool(viable), dtype=torch.bool),
        }
    return aim


def _run_upsize(pp_main, monkeypatch, my_ships, reply_ships):
    status, obs, prod, alive_by_step, H = _board_rich(pp_main, my_ships=my_ships)
    entries = _entries(pp_main, 10.0)
    reply = _stub_reply(pp_main, reply_ships)
    monkeypatch.setattr(
        pp_main, "predict_opp_launches_via_mirror", lambda **kw: reply)
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim())
    monkeypatch.setenv("PRODUCER_PLUS_RESPONSE_VETO_UPSIZE", "1")
    return pp_main._apply_response_veto(
        entries,
        movement=object(), obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )


def test_upsize_beats_parry(monkeypatch, pp_main):
    # Wave of 10 vs garrison 5 dies to a 30-ship same-tick reply (neutral
    # trade -> vetoed). Source holds 60: the full-budget retry sends 60,
    # beats the parry (survivor 30 vs garrison 5) and keeps the planet's
    # prod-5 stream -> clears the margin. Wave survives, upsized.
    out = _run_upsize(pp_main, monkeypatch, my_ships=60.0, reply_ships=30.0)
    assert out.valid.tolist() == [True]
    assert out.ships.tolist() == [60.0]


def test_upsize_insufficient_still_drops(monkeypatch, pp_main):
    # Source only holds 12: the retry sends 12 and still annihilates
    # against the 30-ship reply -> dropped.
    out = _run_upsize(pp_main, monkeypatch, my_ships=12.0, reply_ships=30.0)
    assert out.valid.tolist() == [False]


def test_upsize_off_by_default(monkeypatch, pp_main):
    # Without the upsize gate the parried wave is simply dropped even
    # though the source could afford the bigger send.
    monkeypatch.delenv("PRODUCER_PLUS_RESPONSE_VETO_UPSIZE", raising=False)
    status, obs, prod, alive_by_step, H = _board_rich(pp_main, my_ships=60.0)
    entries = _entries(pp_main, 10.0)
    reply = _stub_reply(pp_main, 30.0)
    monkeypatch.setattr(
        pp_main, "predict_opp_launches_via_mirror", lambda **kw: reply)
    out = pp_main._apply_response_veto(
        entries,
        movement=object(), obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )
    assert out.valid.tolist() == [False]
