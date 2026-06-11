"""Tests for synchronized multi-source arrivals (PRODUCER_PLUS_SYNC).

Two-source pair candidates on targets neither source cracks alone; the
nearer leg is scored at the far leg's arrival tick, diverted post-veto into
a memory hold, and launched on the last turn that still makes the shared
arrival date. These tests exercise the hold lifecycle and the post-veto
diversion with a stubbed intercept solver; full-pipeline behaviour is
covered by the bundle smoke (scripts/sync_probe.py + Rule 46).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

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
        "producer_plus_main_test_sync",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_sync"] = module
    spec.loader.exec_module(module)
    return module


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_SYNC", raising=False)
    assert pp_main._sync_enabled() is False


# --- fixtures ---------------------------------------------------------------


def _obs(ships, owned, alive=None):
    ships_t = torch.tensor(ships, dtype=torch.float32)
    P = int(ships_t.shape[0])
    return SimpleNamespace(
        P=P,
        device=ships_t.device,
        ships=ships_t,
        owned=torch.tensor(owned, dtype=torch.bool),
        alive=torch.ones(P, dtype=torch.bool) if alive is None
        else torch.tensor(alive, dtype=torch.bool),
    )


def _obs_tensors(planet_ids):
    rows = [[float(pid), 0.0, 0.0, 0.0, 1.0, 0.0, 1.0] for pid in planet_ids]
    return {"planets": torch.tensor(rows, dtype=torch.float32)}


def _stub_aim(eta, viable=True, angle=0.5):
    def aim(movement, src, tgt, ships, **kw):
        n = int(src.shape[0])
        return {
            "eta": torch.full((n,), float(eta)),
            "angle": torch.full((n,), float(angle)),
            "viable": torch.full((n,), bool(viable), dtype=torch.bool),
        }
    return aim


def _hold(src_id=10, tgt_id=30, ships=20.0, arrival_step=8):
    return {"src_id": src_id, "tgt_id": tgt_id, "ships": ships,
            "arrival_step": arrival_step}


def _entries(pp_main, rows):
    """rows: (src_slot, tgt_slot, ships, eta, valid)."""
    from orbit_lite.movement_step import LaunchEntries
    return LaunchEntries(
        source_slots=torch.tensor([r[0] for r in rows], dtype=torch.long),
        target_slots=torch.tensor([r[1] for r in rows], dtype=torch.long),
        ships=torch.tensor([r[2] for r in rows], dtype=torch.float32),
        angle=torch.zeros(len(rows), dtype=torch.float32),
        eta=torch.tensor([r[3] for r in rows], dtype=torch.float32),
        valid=torch.tensor([r[4] for r in rows], dtype=torch.bool),
    )


# --- hold lifecycle ----------------------------------------------------------


def test_hold_kept_and_reserved_while_early(monkeypatch, pp_main):
    # Arrival date 8, now 3 (remaining 5), fresh eta 3 -> waiting is correct.
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=3.0))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, False])
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=3)
    assert entries is None
    assert len(mem.sync_holds) == 1
    assert debit is not None and float(debit[0]) == pytest.approx(20.0)


def test_hold_executes_on_last_makeable_turn(monkeypatch, pp_main):
    # Remaining 3, fresh eta 2.4 -> ceil 3 >= 3: launch NOW with fresh aim.
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=2.4, angle=1.25))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, False])
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=5)
    assert mem.sync_holds == []
    assert entries is not None and int(entries.valid.sum()) == 1
    assert int(entries.source_slots[0]) == 0 and int(entries.target_slots[0]) == 2
    assert float(entries.ships[0]) == pytest.approx(20.0)
    assert float(entries.angle[0]) == pytest.approx(1.25)
    assert debit is not None and float(debit[0]) == pytest.approx(20.0)


def test_hold_released_when_date_unreachable(monkeypatch, pp_main):
    # Remaining 2, fresh eta 5 -> even launching now misses by >1 tick: release.
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=5.0))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, False])
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=6)
    assert entries is None and mem.sync_holds == [] and debit is None


def test_hold_launches_one_tick_late_within_slack(monkeypatch, pp_main):
    # Remaining 2, fresh eta 2.6 -> ceil 3 = remaining+1: still fire (1 tick slack).
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=2.6))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, False])
    entries, _ = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=6)
    assert entries is not None and int(entries.valid.sum()) == 1


def test_hold_canceled_when_source_lost(monkeypatch, pp_main):
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=3.0))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [False, False, False])   # source not ours
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=3)
    assert entries is None and mem.sync_holds == [] and debit is None


def test_hold_canceled_when_reserve_drained(monkeypatch, pp_main):
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=3.0))
    mem = SimpleNamespace(sync_holds=[_hold(ships=20.0, arrival_step=8)])
    obs = _obs([12.0, 5.0, 40.0], [True, False, False])    # combat ate the reserve
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=3)
    assert entries is None and mem.sync_holds == [] and debit is None


def test_hold_canceled_when_target_already_ours(monkeypatch, pp_main):
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=3.0))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, True])     # target flipped to us
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=3)
    assert entries is None and mem.sync_holds == [] and debit is None


def test_holds_reset_at_step_zero(monkeypatch, pp_main):
    monkeypatch.setattr(pp_main, "intercept_angle", _stub_aim(eta=3.0))
    mem = SimpleNamespace(sync_holds=[_hold(arrival_step=8)])
    obs = _obs([50.0, 5.0, 40.0], [True, False, False])
    entries, debit = pp_main._process_sync_holds(
        mem, obs=obs, obs_tensors=_obs_tensors([10, 20, 30]),
        movement=None, current_step=0)
    assert entries is None and mem.sync_holds == [] and debit is None


# --- post-veto diversion -------------------------------------------------------


def _sink_row(near_src=0, far_src=1, tgt=2, eta=6.2, near_ships=15.0,
              far_ships=25.0, arrival_dt=7):
    return {"near_src": near_src, "far_src": far_src, "tgt": tgt, "eta": eta,
            "near_ships": near_ships, "far_ships": far_ships,
            "arrival_dt": arrival_dt}


def test_divert_near_leg_to_hold_when_far_survives(pp_main):
    # Entry 0 = far leg (launch now), entry 1 = near leg (eta scored at far's).
    entries = _entries(pp_main, [
        (1, 2, 25.0, 6.2, True),
        (0, 2, 15.0, 6.2, True),
    ])
    mem = SimpleNamespace(sync_holds=[])
    out = pp_main._divert_sync_entries(
        entries, sink=[_sink_row()], obs_tensors=_obs_tensors([10, 20, 30]),
        current_step=4, memory=mem)
    assert bool(out.valid[0]) is True       # far leg launches
    assert bool(out.valid[1]) is False      # near leg held, not launched
    assert len(mem.sync_holds) == 1
    h = mem.sync_holds[0]
    assert h["src_id"] == 10 and h["tgt_id"] == 30
    assert h["ships"] == pytest.approx(15.0)
    assert h["arrival_step"] == 4 + 7


def test_divert_drops_near_leg_when_far_vetoed(pp_main):
    entries = _entries(pp_main, [
        (1, 2, 25.0, 6.2, False),           # far leg vetoed
        (0, 2, 15.0, 6.2, True),
    ])
    mem = SimpleNamespace(sync_holds=[])
    out = pp_main._divert_sync_entries(
        entries, sink=[_sink_row()], obs_tensors=_obs_tensors([10, 20, 30]),
        current_step=4, memory=mem)
    assert bool(out.valid[1]) is False
    assert mem.sync_holds == []             # no hold without the partner


def test_divert_leaves_unrelated_entries_alone(pp_main):
    entries = _entries(pp_main, [
        (1, 2, 25.0, 4.0, True),            # ordinary wave, eta differs
        (0, 1, 9.0, 2.0, True),
    ])
    mem = SimpleNamespace(sync_holds=[])
    out = pp_main._divert_sync_entries(
        entries, sink=[_sink_row()], obs_tensors=_obs_tensors([10, 20, 30]),
        current_step=4, memory=mem)
    assert bool(out.valid[0]) and bool(out.valid[1])
    assert mem.sync_holds == []


def test_divert_respects_max_holds_cap(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_SYNC_MAX_HOLDS", "1")
    entries = _entries(pp_main, [
        (1, 2, 25.0, 6.2, True),
        (0, 2, 15.0, 6.2, True),
    ])
    mem = SimpleNamespace(sync_holds=[_hold()])   # cap already filled
    out = pp_main._divert_sync_entries(
        entries, sink=[_sink_row()], obs_tensors=_obs_tensors([10, 20, 30]),
        current_step=4, memory=mem)
    assert bool(out.valid[1]) is False      # near leg still never launches now
    assert len(mem.sync_holds) == 1         # but no new hold beyond the cap
