"""Unit + integration tests for producer_plus opp-foresight mechanisms.

Three mechanisms layered on top of the lite-greedy opp projection:
- Source-exposure penalty (Mechanism 1)
- Race-loss penalty (Mechanism 2)
- Counter-capture target seeding (Mechanism 3)

Each helper is pure-tensor and unit-testable in isolation; one
integration test verifies the gates wire into run_turn correctly.
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
def main_module():
    """Load producer_plus/main.py under an isolated sys.modules name."""
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_foresight_test",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_foresight_test"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _opp_owner_mask
# ---------------------------------------------------------------------------


def test_opp_owner_mask_2p(main_module):
    m = main_module._opp_owner_mask(
        my_id=0, num_seats=2, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert m.tolist() == [False, True]


def test_opp_owner_mask_4p_my_id_2(main_module):
    m = main_module._opp_owner_mask(
        my_id=2, num_seats=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert m.tolist() == [True, True, False, True]


def test_opp_owner_mask_handles_oob_my_id(main_module):
    # my_id outside [0, A) means all seats are "opp" — no self bit to flip off.
    m = main_module._opp_owner_mask(
        my_id=9, num_seats=2, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert m.tolist() == [True, True]


# ---------------------------------------------------------------------------
# _fresh_opp_capture_mask + _append_counter_capture_targets
# ---------------------------------------------------------------------------


def _mk_status(owner_timeline):
    """Build a lightweight stand-in for PlanetGarrisonStatus with just .owner."""
    class _S:
        owner = torch.tensor(owner_timeline, dtype=torch.long)
    return _S()


def test_fresh_opp_capture_detects_capture_in_window(main_module):
    # 2 planets, H+1 = 4 ticks. Planet 0 stays ours (0). Planet 1 flips to opp (1) at tick 2.
    status = _mk_status([[0, 0, 0, 0], [0, 0, 1, 1]])
    mask = main_module._fresh_opp_capture_mask(status, my_id=0, K_eta=3)
    assert mask.tolist() == [False, True]


def test_fresh_opp_capture_ignores_planets_already_opp_at_step_0(main_module):
    # Planet 0 was opp from the start — NOT a fresh capture.
    status = _mk_status([[1, 1, 1, 1], [0, 0, 0, 0]])
    mask = main_module._fresh_opp_capture_mask(status, my_id=0, K_eta=3)
    assert mask.tolist() == [False, False]


def test_fresh_opp_capture_clips_to_K_eta(main_module):
    # Capture happens at tick 3 only; K_eta=2 means window is [1, 2]. Miss it.
    status = _mk_status([[0, 0, 0, 1]])
    mask = main_module._fresh_opp_capture_mask(status, my_id=0, K_eta=2)
    assert mask.tolist() == [False]
    # K_eta=3 brings it into the window.
    mask = main_module._fresh_opp_capture_mask(status, my_id=0, K_eta=3)
    assert mask.tolist() == [True]


def test_fresh_opp_capture_K_eta_zero_returns_empty(main_module):
    status = _mk_status([[0, 0, 1, 1]])
    mask = main_module._fresh_opp_capture_mask(status, my_id=0, K_eta=0)
    assert mask.tolist() == [False]


def test_append_counter_capture_appends_only_new_slots(main_module):
    target_idx = torch.tensor([5, 7], dtype=torch.long)
    target_exists = torch.tensor([True, True])
    fresh_mask = torch.tensor([False, False, False, False, False, True, False, True, True, False])
    #                            slots 0-9: slot 5 already present, 7 already present, 8 is new
    new_idx, new_exists = main_module._append_counter_capture_targets(
        target_idx, target_exists, fresh_capture_mask=fresh_mask,
    )
    assert new_idx.tolist() == [5, 7, 8]
    assert new_exists.tolist() == [True, True, True]


def test_append_counter_capture_noop_when_no_fresh(main_module):
    target_idx = torch.tensor([5, 7], dtype=torch.long)
    target_exists = torch.tensor([True, True])
    fresh_mask = torch.zeros(10, dtype=torch.bool)
    new_idx, new_exists = main_module._append_counter_capture_targets(
        target_idx, target_exists, fresh_capture_mask=fresh_mask,
    )
    assert torch.equal(new_idx, target_idx)
    assert torch.equal(new_exists, target_exists)


# ---------------------------------------------------------------------------
# _apply_source_exposure_penalty
# ---------------------------------------------------------------------------


def test_source_exposure_safe_residual_unchanged(main_module):
    # 1 candidate, src=0, send=5 ships, eta=2.
    # source_ships[0] = 30 → residual = 25. Opp arrivals at src 0 in window [0,2) = 5+5 = 10.
    # margin=1.0 → 25 >= 10 → safe → score unchanged.
    fleet_buckets = torch.zeros(2, 3, 2, dtype=torch.float32)
    fleet_buckets[0, 0, 1] = 5.0  # opp axis 1 arrives at tick 1 (bucket 0)
    fleet_buckets[0, 1, 1] = 5.0  # opp arrives at tick 2 (bucket 1)
    score_in = torch.tensor([2.5], dtype=torch.float32)
    score_out = main_module._apply_source_exposure_penalty(
        score_in,
        cand_src=torch.tensor([[0]], dtype=torch.long),
        cand_send=torch.tensor([[5.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        source_ships_per_planet=torch.tensor([30.0, 0.0], dtype=torch.float32),
        safety_margin=1.0,
    )
    assert score_out.tolist() == [pytest.approx(2.5)]


def test_source_exposure_exposed_residual_becomes_neginf(main_module):
    # Same arrivals (10 opp ships incoming). Send 8 → residual = 30-8 = 22.
    # If we shrink source to 10 ships: residual = 2 < 10 → exposed.
    fleet_buckets = torch.zeros(2, 3, 2, dtype=torch.float32)
    fleet_buckets[0, 0, 1] = 5.0
    fleet_buckets[0, 1, 1] = 5.0
    score_in = torch.tensor([2.5], dtype=torch.float32)
    score_out = main_module._apply_source_exposure_penalty(
        score_in,
        cand_src=torch.tensor([[0]], dtype=torch.long),
        cand_send=torch.tensor([[8.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        source_ships_per_planet=torch.tensor([10.0, 0.0], dtype=torch.float32),
        safety_margin=1.0,
    )
    assert score_out[0].item() == float("-inf")


def test_source_exposure_margin_zero_disables(main_module):
    fleet_buckets = torch.zeros(2, 3, 2, dtype=torch.float32)
    fleet_buckets[0, 0, 1] = 100.0  # huge threat
    score_in = torch.tensor([5.0], dtype=torch.float32)
    score_out = main_module._apply_source_exposure_penalty(
        score_in,
        cand_src=torch.tensor([[0]], dtype=torch.long),
        cand_send=torch.tensor([[1.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        source_ships_per_planet=torch.tensor([2.0, 0.0], dtype=torch.float32),
        safety_margin=0.0,
    )
    # residual=1, margin=0 → threshold=0 → 1 >= 0 → safe → unchanged.
    assert score_out.tolist() == [pytest.approx(5.0)]


# ---------------------------------------------------------------------------
# _apply_race_loss_penalty
# ---------------------------------------------------------------------------


def test_race_loss_no_opp_at_target_unchanged(main_module):
    fleet_buckets = torch.zeros(2, 4, 2, dtype=torch.float32)  # P=2, H=4, A=2
    score_in = torch.tensor([3.0], dtype=torch.float32)
    score_out = main_module._apply_race_loss_penalty(
        score_in,
        cand_tgt_slot=torch.tensor([1], dtype=torch.long),
        cand_send=torch.tensor([[10.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        multiplier=0.2,
    )
    assert score_out.tolist() == [pytest.approx(3.0)]


def test_race_loss_opp_outraces_us_score_multiplied(main_module):
    # Opp lands 15 ships at planet 1 by tick 2; we send 10. opp >= us → penalty.
    fleet_buckets = torch.zeros(2, 4, 2, dtype=torch.float32)
    fleet_buckets[1, 0, 1] = 15.0  # opp arrival at tick 1 (bucket 0)
    score_in = torch.tensor([3.0], dtype=torch.float32)
    score_out = main_module._apply_race_loss_penalty(
        score_in,
        cand_tgt_slot=torch.tensor([1], dtype=torch.long),
        cand_send=torch.tensor([[10.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        multiplier=0.2,
    )
    assert score_out.tolist() == [pytest.approx(0.6)]


def test_race_loss_multiplier_one_disables(main_module):
    fleet_buckets = torch.zeros(2, 4, 2, dtype=torch.float32)
    fleet_buckets[1, 0, 1] = 99.0
    score_in = torch.tensor([3.0], dtype=torch.float32)
    score_out = main_module._apply_race_loss_penalty(
        score_in,
        cand_tgt_slot=torch.tensor([1], dtype=torch.long),
        cand_send=torch.tensor([[1.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        multiplier=1.0,
    )
    assert score_out.tolist() == [pytest.approx(3.0)]


def test_race_loss_minus_inf_stays_minus_inf(main_module):
    fleet_buckets = torch.zeros(2, 4, 2, dtype=torch.float32)
    fleet_buckets[1, 0, 1] = 99.0
    score_in = torch.tensor([float("-inf")], dtype=torch.float32)
    score_out = main_module._apply_race_loss_penalty(
        score_in,
        cand_tgt_slot=torch.tensor([1], dtype=torch.long),
        cand_send=torch.tensor([[1.0]], dtype=torch.float32),
        cand_eta=torch.tensor([[2.0]], dtype=torch.float32),
        fleet_buckets=fleet_buckets,
        opp_owner_mask=torch.tensor([False, True]),
        multiplier=0.2,
    )
    assert score_out[0].item() == float("-inf")


# ---------------------------------------------------------------------------
# Env gate readers — make sure they default OFF and read truthy strings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("0", False), ("", False), ("false", False),
     ("1", True), ("true", True), ("YES", True), ("on", True)],
)
def test_source_exposure_env_gate(monkeypatch, main_module, value, expected):
    if value is None:
        monkeypatch.delenv("PRODUCER_PLUS_SOURCE_EXPOSURE", raising=False)
    else:
        monkeypatch.setenv("PRODUCER_PLUS_SOURCE_EXPOSURE", value)
    assert main_module._source_exposure_enabled() is expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("0", False), ("1", True), ("on", True)],
)
def test_race_loss_env_gate(monkeypatch, main_module, value, expected):
    if value is None:
        monkeypatch.delenv("PRODUCER_PLUS_RACE_LOSS", raising=False)
    else:
        monkeypatch.setenv("PRODUCER_PLUS_RACE_LOSS", value)
    assert main_module._race_loss_enabled() is expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("0", False), ("1", True), ("on", True)],
)
def test_counter_capture_env_gate(monkeypatch, main_module, value, expected):
    if value is None:
        monkeypatch.delenv("PRODUCER_PLUS_COUNTER_CAPTURE", raising=False)
    else:
        monkeypatch.setenv("PRODUCER_PLUS_COUNTER_CAPTURE", value)
    assert main_module._counter_capture_enabled() is expected


# ---------------------------------------------------------------------------
# Integration: env-off bit-identical default vs vanilla producer.
# Mirrors the wrap-and-restore test pattern but checks the simpler
# "with ALL new env vars unset, action emitted matches vanilla producer."
# ---------------------------------------------------------------------------


def test_all_gates_off_action_matches_vanilla_producer(monkeypatch):
    """With all opp-foresight env vars unset, producer_plus must emit the
    same action row as vanilla producer at step 0 on a fresh game."""
    # Clear every opp-foresight related env var.
    for var in (
        "PRODUCER_PLUS_OPP_PROJECTOR",
        "PRODUCER_PLUS_SOURCE_EXPOSURE",
        "PRODUCER_PLUS_RACE_LOSS",
        "PRODUCER_PLUS_COUNTER_CAPTURE",
        "PRODUCER_PLUS_ADAPTIVE_K",
        "PRODUCER_PLUS_SOURCE_EXPOSURE_MARGIN",
        "PRODUCER_PLUS_RACE_LOSS_MULT",
    ):
        monkeypatch.delenv(var, raising=False)

    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    prod = _load("prod_main_foresight_test",
                 os.path.join(PRODUCER_DIR, "main.py"))
    pp = _load("pp_main_foresight_offcheck",
               os.path.join(PRODUCER_PLUS_DIR, "main.py"))

    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 7}, debug=False)
    env.reset()
    raw_obs = dict(env.state[0].observation)

    prod._RUNTIME.reset()
    pp._RUNTIME.reset()

    a_prod = prod.agent(raw_obs)
    a_pp = pp.agent(raw_obs)
    assert a_prod == a_pp, (
        f"env-off bit-identical broken: producer={a_prod} producer_plus={a_pp}"
    )
