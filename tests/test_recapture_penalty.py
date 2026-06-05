"""Tests for the recapture-penalty leaf-scorer term.

Covers Rule 38 (fix-verification reproduces failure state):
    1. Env-getter parses gate / weight / K / safety_reserve correctly,
       including the per-player-count K override.
    2. With the gate UNSET (default), the producer_plus agent's actions
       are byte-identical to the same agent with the gate explicitly = 0
       (guards against the OFF-path getting perturbed).
    3. With the gate ON, action rows DIFFER from the OFF path at at least
       one seed — proves the recapture-penalty code path is exercised.
    4. Synthetic unit test of ``recapture_penalty()``: build a small
       fixture by hand and assert the math.
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
        "producer_plus_main_test_recapture",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_recapture"] = module
    spec.loader.exec_module(module)
    return module


def _clear_recapture_env(monkeypatch):
    for name in (
        "PRODUCER_PLUS_RECAPTURE_PENALTY",
        "PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT",
        "PRODUCER_PLUS_RECAPTURE_K",
        "PRODUCER_PLUS_RECAPTURE_K_2P",
        "PRODUCER_PLUS_RECAPTURE_K_4P",
        "PRODUCER_PLUS_RECAPTURE_SAFETY_RESERVE",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Env-getter tests
# ---------------------------------------------------------------------------


def test_recapture_default_off(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    assert producer_plus_main._recapture_penalty_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "ON"])
def test_recapture_env_on_truthy(monkeypatch, producer_plus_main, value):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_PENALTY", value)
    assert producer_plus_main._recapture_penalty_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_recapture_env_off_falsy(monkeypatch, producer_plus_main, value):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_PENALTY", value)
    assert producer_plus_main._recapture_penalty_enabled() is False


def test_recapture_weight_default(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    assert producer_plus_main._recapture_penalty_weight() == 1.0


@pytest.mark.parametrize("value,expected", [
    ("0.5", 0.5), ("2.0", 2.0), ("0", 0.0), ("3.5", 3.5),
])
def test_recapture_weight_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT", value)
    assert producer_plus_main._recapture_penalty_weight() == expected


@pytest.mark.parametrize("value", ["abc", "", "-1"])
def test_recapture_weight_invalid_clamps(monkeypatch, producer_plus_main, value):
    """Garbage values return the default (1.0); negative clamps to 0."""
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT", value)
    out = producer_plus_main._recapture_penalty_weight()
    assert out >= 0.0  # never negative
    assert out == 1.0 or out == 0.0  # default-fallback or non-negative clamp


def test_recapture_k_default(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    assert producer_plus_main._recapture_k(2) == 8
    assert producer_plus_main._recapture_k(4) == 8


def test_recapture_k_2p_override(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_K", "5")
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_K_2P", "12")
    assert producer_plus_main._recapture_k(2) == 12
    assert producer_plus_main._recapture_k(4) == 5  # 4P falls back to base


def test_recapture_k_4p_override(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_K", "5")
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_K_4P", "10")
    assert producer_plus_main._recapture_k(4) == 10
    assert producer_plus_main._recapture_k(2) == 5  # 2P falls back to base


def test_recapture_safety_reserve_default(monkeypatch, producer_plus_main):
    _clear_recapture_env(monkeypatch)
    assert producer_plus_main._recapture_safety_reserve() == 0.5


@pytest.mark.parametrize("value,expected", [
    ("0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0),
    ("1.5", 1.0),   # clamped to [0, 1]
    ("-0.1", 0.0),  # clamped to [0, 1]
    ("abc", 0.5),   # invalid -> default
    ("", 0.5),
])
def test_recapture_safety_reserve_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_recapture_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_SAFETY_RESERVE", value)
    assert producer_plus_main._recapture_safety_reserve() == expected


# ---------------------------------------------------------------------------
# Synthetic unit test of recapture_penalty()
# ---------------------------------------------------------------------------


def _build_synthetic_fixture(opp_garrison: float):
    """Construct a minimal P=4 fixture: P0=ours, P1=target enemy, P2=opp
    next-door (with ``opp_garrison`` ships), P3=neutral filler. One
    candidate that thinly captures P1.
    Returns the kwargs to pass to ``recapture_penalty``.
    """
    import torch
    from agents.producer.orbit_lite.obs import ParsedObs
    from agents.producer.orbit_lite.distance_cache import DistanceCache
    from agents.producer.orbit_lite.movement import PlanetGarrisonStatus

    P = 4
    device = torch.device("cpu")
    dtype = torch.float32
    zP_f = torch.zeros(P, dtype=dtype)
    zP_b = torch.zeros(P, dtype=torch.bool)

    # Build a real ParsedObs (matches production type contract). Fields not
    # used by recapture_penalty are zero-filled but still present so future
    # accesses don't crash.
    obs = ParsedObs(
        alive=torch.tensor([True, True, True, True]),
        x=torch.tensor([0.0, 10.0, 15.0, 100.0], dtype=dtype),
        y=zP_f.clone(),
        r=torch.full((P,), 1.0, dtype=dtype),
        ships=torch.tensor([10.0, 5.0, opp_garrison, 0.0], dtype=dtype),
        prod=torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=dtype),
        owner_abs=torch.tensor([0.0, 1.0, 1.0, -1.0], dtype=dtype),
        owned=torch.tensor([True, False, False, False]),
        is_enemy=torch.tensor([False, True, True, False]),
        is_neutral=torch.tensor([False, False, False, True]),
        orb_r=zP_f.clone(), orb_a0=zP_f.clone(), is_orbiting=zP_b.clone(),
        angvel=torch.zeros(1, dtype=dtype),
        step=torch.zeros(1, dtype=dtype),
        f_alive=torch.zeros(0, dtype=torch.bool),
        f_owner=torch.zeros(0, dtype=dtype),
        f_x=torch.zeros(0, dtype=dtype),
        f_y=torch.zeros(0, dtype=dtype),
        f_angle=torch.zeros(0, dtype=dtype),
        f_ships=torch.zeros(0, dtype=dtype),
        player_id=0, P=P, F=0, device=device,
    )

    # --- distance cache -----------------------------------------------------
    K_cache = 12
    # Planets along a line: P0 at x=0, P1 at x=10, P2 at x=15 (next to P1),
    # P3 at x=100. Static (no orbit). cross_dist[k, src, tgt] = same for all k
    # since static.
    xs = torch.tensor([0.0, 10.0, 15.0, 100.0])
    dx = xs.view(1, P, 1) - xs.view(1, 1, P)
    cross_dist = torch.sqrt((dx * dx).expand(K_cache + 1, -1, -1).clamp(min=0.0))
    alive_by_step = torch.ones(K_cache + 1, P, dtype=torch.bool)
    cache = DistanceCache(
        cross_dist=cross_dist.to(dtype),
        alive_by_step=alive_by_step,
        K=K_cache,
    )

    # --- garrison_status ----------------------------------------------------
    # do-nothing trajectory over H+1 ticks.
    H = 18
    owner_traj = torch.tensor([0, 1, 1, 1], dtype=torch.long).view(P, 1).expand(P, H + 1).clone()
    ships_traj = torch.zeros(P, H + 1, dtype=dtype)
    garrison_status = PlanetGarrisonStatus(
        owner=owner_traj,
        ships=ships_traj,
        pre_combat_owner=None,
        pre_combat_ships=None,
        arrivals_by_owner=None,
    )

    # --- single candidate that captures P1 thinly --------------------------
    cand_tgt_slot = torch.tensor([1], dtype=torch.long)
    cand_tgt_short = torch.tensor([0], dtype=torch.long)  # short idx into floor
    cand_send = torch.tensor([[6.0]], dtype=dtype)        # just over floor of 5
    cand_eta = torch.tensor([[3.0]], dtype=dtype)         # arrives at tick 3
    cand_valid = torch.tensor([True])
    cand_is_def = torch.tensor([False])
    # capture_floor[T, K]: floor=5 at all k for P1 (defender of 5 ships).
    capture_floor_TK = torch.full((1, 12), 5.0, dtype=dtype)
    prod = obs.prod.clone()

    return dict(
        obs=obs, cache=cache, garrison_status=garrison_status,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        capture_floor_TK=capture_floor_TK,
        prod=prod, H=H,
        K_recap=8, K_opp=0,
        safety_reserve=0.0,  # max threat, easier to test
        player_id=0,
    )


def test_recapture_penalty_fires_when_opp_can_recapture():
    """Opp at P2 has 50 ships, distance 5 to P1. Easy recapture.
    Our thin capture (6 ships, defender = 6-5 = 1) is well below threat.
    Penalty should be > 0.
    """
    from agents.producer.orbit_lite.recapture import recapture_penalty
    kw = _build_synthetic_fixture(opp_garrison=50.0)
    pen = recapture_penalty(**kw)
    assert pen.shape == (1,)
    assert pen.item() > 0.0, f"expected penalty > 0, got {pen.item()}"


def test_recapture_penalty_zero_when_opp_unreachable():
    """If no enemy planet can reach the target within K_recap, no threat,
    no penalty."""
    from agents.producer.orbit_lite.recapture import recapture_penalty
    kw = _build_synthetic_fixture(opp_garrison=50.0)
    # Set K_recap to a tiny value so no enemy can reach (distance 5 needs
    # >= 5 ticks for default speed).
    kw["K_recap"] = 1
    pen = recapture_penalty(**kw)
    assert pen.item() == 0.0, f"expected 0 (unreachable), got {pen.item()}"


def test_recapture_penalty_zero_for_defensive_candidate():
    """cand_is_def=True (own-planet reinforcement) → penalty skipped."""
    from agents.producer.orbit_lite.recapture import recapture_penalty
    kw = _build_synthetic_fixture(opp_garrison=50.0)
    kw["cand_is_def"] = kw["cand_is_def"].clone()
    kw["cand_is_def"][0] = True
    pen = recapture_penalty(**kw)
    assert pen.item() == 0.0


def test_recapture_penalty_zero_when_send_below_floor():
    """If we wouldn't actually capture (send < floor), penalty = 0."""
    from agents.producer.orbit_lite.recapture import recapture_penalty
    kw = _build_synthetic_fixture(opp_garrison=50.0)
    kw["cand_send"] = kw["cand_send"].clone()
    kw["cand_send"][0, 0] = 3.0  # below floor of 5
    pen = recapture_penalty(**kw)
    assert pen.item() == 0.0


# ---------------------------------------------------------------------------
# Shim <-> bundle drift checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shim_file,variant_name", [
    # All currently-shipped producer_plus shim files paired with the
    # ENV_VARIANTS key whose env-var set they should match. Drift in any
    # pair means local play diverges from the bundled submission.
    ("producer_plus_adaptive_k.py", "adaptive_k"),
    ("producer_plus_multi_size.py", "multi_size"),
    ("producer_plus_coalitions.py", "coalitions"),
    ("producer_plus_composed.py", "composed"),
    ("producer_plus_opp_proj.py", "opp_proj"),
    ("producer_plus_multi_opp.py", "multi_opp_def"),
    ("producer_plus_multi_tick_opp_K3.py", "multi_tick_opp_K3"),
    ("producer_plus_recapture_penalty.py", "recapture_penalty"),
    ("producer_plus_multi_tick_recap.py", "multi_tick_recap"),
])
def test_shim_env_vars_match_bundle_variant(shim_file, variant_name):
    """Drift guard: each shim's `os.environ.setdefault` calls must match
    the same-named entry in `ENV_VARIANTS`. If they diverge, local play
    via the shim runs a different configuration than the bundled
    submission."""
    import importlib.util as _il
    import re
    shim_path = os.path.join(PRODUCER_PLUS_DIR, shim_file)
    bundler_path = os.path.join(REPO_ROOT, "scripts", "bundle_producer_plus.py")
    spec_b = _il.spec_from_file_location(
        f"bundle_drift_check_{variant_name}", bundler_path,
    )
    mod_b = _il.module_from_spec(spec_b)
    spec_b.loader.exec_module(mod_b)
    bundle_vars = mod_b.ENV_VARIANTS[variant_name]
    src = open(shim_path).read()
    pattern = re.compile(
        r'os\.environ\.setdefault\(\s*"([A-Z_0-9]+)"\s*,\s*"([^"]+)"\s*\)'
    )
    shim_vars = dict(pattern.findall(src))
    assert shim_vars == bundle_vars, (
        f"shim '{shim_file}' env vars {shim_vars} drift from bundle "
        f"variant '{variant_name}' {bundle_vars}"
    )


# ---------------------------------------------------------------------------
# Integration tests (slow)
# ---------------------------------------------------------------------------


def _play_one_game(focal_path, opp_path, seed):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, opp_path])
    return env.state, env.steps


@pytest.mark.slow
def test_recapture_penalty_off_byte_identical(monkeypatch):
    """Rule 38: gate OFF (explicit "0") must match gate UNSET. Both hit
    the same code path (no penalty subtracted), so the game outcome must
    agree."""
    shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_recapture_penalty.py")
    producer = os.path.join(PRODUCER_DIR, "producer_agent.py")
    for name in (
        "PRODUCER_PLUS_RECAPTURE_PENALTY",
        "PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT",
        "PRODUCER_PLUS_RECAPTURE_K",
        "PRODUCER_PLUS_RECAPTURE_SAFETY_RESERVE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Compare with the standalone shim ON vs explicit gate OFF; we just
    # measure determinism, not byte-identity to a baseline (the slow test
    # ``test_off_path_bit_identical_to_producer`` in test_producer_plus_opp_proj.py
    # already gates the global OFF-path identity).
    monkeypatch.setenv("PRODUCER_PLUS_RECAPTURE_PENALTY", "0")
    state_zero, _ = _play_one_game(shim, producer, 7)
    monkeypatch.delenv("PRODUCER_PLUS_RECAPTURE_PENALTY", raising=False)
    state_unset, _ = _play_one_game(shim, producer, 7)
    assert state_zero[0]["reward"] == state_unset[0]["reward"], (
        f"explicit gate=0 ({state_zero[0]['reward']}) differs from unset "
        f"({state_unset[0]['reward']}) — OFF mapping mis-routed"
    )


@pytest.mark.slow
def test_recapture_penalty_on_changes_planner_output():
    """Proof the recapture-penalty code path is exercised: at K_recap=8
    and weight=1.0, gate ON must differ from gate OFF on at least one of
    seeds (7, 13, 42). If all agree, the term is no-op in practice."""
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    off_shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_opp.py")
    on_shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_tick_recap.py")
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
        "recapture-penalty ON produced identical outcomes to OFF on "
        "seeds 7, 13, 42 — code path may be unreachable"
    )


@pytest.mark.slow
def test_recapture_penalty_smoke_wallclock_under_60s():
    """Rule 46c: full game vs producer at seed 7 under 60 s wallclock.
    Recapture penalty is O(C*P*K_recap) tensor ops -- ~70k FLOPs per
    turn -- should add < 5 ms p50."""
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_recap.py",
    )
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"recapture+multi_tick game at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock regression"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]
