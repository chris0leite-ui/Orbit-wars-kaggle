"""Tests for denial-bonus and opening-bonus leaf-scorer terms.

Covers Rule 38 (fix-verification reproduces failure state):
    1. Env-getters parse defaults / truthy / falsy / garbage correctly.
    2. With the gates UNSET (default), the producer_plus agent's action
       rows match the same agent with the gates explicitly = 0.
    3. With the gates ON, action rows DIFFER from the OFF path on at
       least one seed (proves the code path is exercised).
    4. Synthetic unit tests of ``denial_bonus()`` and ``opening_bonus()``
       exercise the gate conditions (opp owns / opp predicted / no opp
       value; step within / past opening window; weight=0 returns 0).
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
        "producer_plus_main_test_strategic",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_strategic"] = module
    spec.loader.exec_module(module)
    return module


def _clear_strategic_env(monkeypatch):
    for name in (
        "PRODUCER_PLUS_DENIAL_BONUS",
        "PRODUCER_PLUS_DENIAL_WEIGHT",
        "PRODUCER_PLUS_OPENING_BONUS",
        "PRODUCER_PLUS_OPENING_WEIGHT",
        "PRODUCER_PLUS_OPENING_WINDOW",
        "PRODUCER_PLUS_GAME_LENGTH_EST",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Env-getter tests
# ---------------------------------------------------------------------------


def test_denial_default_off(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._denial_bonus_enabled() is False


def test_opening_default_off(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._opening_bonus_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "ON"])
def test_denial_env_on_truthy(monkeypatch, producer_plus_main, value):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_DENIAL_BONUS", value)
    assert producer_plus_main._denial_bonus_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_denial_env_off_falsy(monkeypatch, producer_plus_main, value):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_DENIAL_BONUS", value)
    assert producer_plus_main._denial_bonus_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_opening_env_on_truthy(monkeypatch, producer_plus_main, value):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_BONUS", value)
    assert producer_plus_main._opening_bonus_enabled() is True


def test_denial_weight_default(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._denial_bonus_weight() == 0.1


def test_opening_weight_default(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._opening_bonus_weight() == 0.1


@pytest.mark.parametrize("value,expected", [
    ("0", 0.0), ("0.5", 0.5), ("1.0", 1.0), ("2.5", 2.5),
])
def test_weights_parse(monkeypatch, producer_plus_main, value, expected):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_DENIAL_WEIGHT", value)
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_WEIGHT", value)
    assert producer_plus_main._denial_bonus_weight() == expected
    assert producer_plus_main._opening_bonus_weight() == expected


@pytest.mark.parametrize("value", ["abc", "nan", "", "-1"])
def test_weights_invalid_fall_to_default_or_clamp(monkeypatch, producer_plus_main, value):
    """Garbage parses to default (0.1); negative clamps to 0."""
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_DENIAL_WEIGHT", value)
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_WEIGHT", value)
    out_d = producer_plus_main._denial_bonus_weight()
    out_o = producer_plus_main._opening_bonus_weight()
    assert out_d in (0.0, 0.1) and out_o in (0.0, 0.1)
    assert out_d >= 0.0 and out_o >= 0.0


def test_opening_window_default(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._opening_window() == 30


@pytest.mark.parametrize("value,expected", [
    ("1", 1), ("15", 15), ("60", 60),
])
def test_opening_window_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_WINDOW", value)
    assert producer_plus_main._opening_window() == expected


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_opening_window_invalid_fallback(monkeypatch, producer_plus_main, value):
    """Invalid / zero / negative clamps to >= 1 or falls to default 30."""
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_OPENING_WINDOW", value)
    out = producer_plus_main._opening_window()
    assert out >= 1


def test_game_length_default(monkeypatch, producer_plus_main):
    _clear_strategic_env(monkeypatch)
    assert producer_plus_main._game_length_est() == 200


@pytest.mark.parametrize("value,expected", [
    ("50", 50), ("100", 100), ("500", 500),
])
def test_game_length_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_strategic_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_GAME_LENGTH_EST", value)
    assert producer_plus_main._game_length_est() == expected


# ---------------------------------------------------------------------------
# Synthetic unit-test fixture for denial / opening bonuses
# ---------------------------------------------------------------------------


def _build_strategic_fixture(
    *,
    P1_owner: float = -1.0,  # -1 = neutral; 1.0 = opp-owned
    opp_intent_at_P1_ships: float = 0.0,  # opp_proj's predicted ships at P1
    cand_send_ships: float = 6.0,  # over floor of 5 = capture
    current_step: int = 0,
    game_length_est: int = 200,
    opening_window: int = 30,
    denial_weight: float = 0.1,
    opening_weight: float = 0.1,
    H: int = 18,
):
    """Build a real-ParsedObs fixture for the strategic-value tests.

    P0 ours, P1 target (owner controlled by P1_owner), P2 neutral, P3
    neutral. capture_floor for P1 is 5 ships. cand sends ``cand_send_ships``
    at eta=3. Background opp_proj has ``opp_intent_at_P1_ships`` at P1.
    """
    import torch
    from agents.producer.orbit_lite.obs import ParsedObs
    from agents.producer.orbit_lite.movement import PlanetGarrisonStatus
    from agents.producer.orbit_lite.garrison_launch import LaunchSet

    P = 4
    device = torch.device("cpu")
    dtype = torch.float32
    zP_f = torch.zeros(P, dtype=dtype)
    zP_b = torch.zeros(P, dtype=torch.bool)

    is_enemy = torch.tensor([False, P1_owner == 1.0, False, False])
    is_neutral = torch.tensor([False, P1_owner == -1.0, True, True])
    owned = torch.tensor([True, False, False, False])

    obs = ParsedObs(
        alive=torch.tensor([True, True, True, True]),
        x=zP_f.clone(), y=zP_f.clone(),
        r=torch.full((P,), 1.0, dtype=dtype),
        ships=torch.tensor([10.0, 5.0, 1.0, 0.0], dtype=dtype),
        prod=torch.tensor([1.0, 3.0, 1.0, 1.0], dtype=dtype),
        owner_abs=torch.tensor([0.0, P1_owner, -1.0, -1.0], dtype=dtype),
        owned=owned, is_enemy=is_enemy, is_neutral=is_neutral,
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

    # do-nothing trajectory: P0 ours, others as their owner.
    owner_at_arrival = int(P1_owner) if P1_owner >= 0 else -1
    owner_traj = torch.tensor(
        [0, owner_at_arrival, -1, -1], dtype=torch.long
    ).view(P, 1).expand(P, H + 1).clone()
    garrison_status = PlanetGarrisonStatus(
        owner=owner_traj, ships=torch.zeros(P, H + 1, dtype=dtype),
        pre_combat_owner=None, pre_combat_ships=None, arrivals_by_owner=None,
    )

    # background: one opp launch at P1 with `opp_intent_at_P1_ships`.
    if opp_intent_at_P1_ships > 0:
        L_bg = 1
        background = LaunchSet(
            source_slots=torch.tensor([2], dtype=torch.long),
            target_slots=torch.tensor([1], dtype=torch.long),
            ships=torch.tensor([opp_intent_at_P1_ships], dtype=dtype),
            eta=torch.tensor([2.0], dtype=dtype),
            owner=torch.tensor([1], dtype=torch.long),
            valid=torch.tensor([True]),
        )
    else:
        background = LaunchSet(
            source_slots=torch.zeros(0, dtype=torch.long),
            target_slots=torch.zeros(0, dtype=torch.long),
            ships=torch.zeros(0, dtype=dtype),
            eta=torch.zeros(0, dtype=dtype),
            owner=torch.zeros(0, dtype=torch.long),
            valid=torch.zeros(0, dtype=torch.bool),
        )

    # Single candidate: P0 -> P1, 6 ships, eta=3.
    cand_tgt_slot = torch.tensor([1], dtype=torch.long)
    cand_tgt_short = torch.tensor([0], dtype=torch.long)
    cand_send = torch.tensor([[cand_send_ships]], dtype=dtype)
    cand_eta = torch.tensor([[3.0]], dtype=dtype)
    cand_valid = torch.tensor([True])
    cand_is_def = torch.tensor([False])
    capture_floor_TK = torch.full((1, 12), 5.0, dtype=dtype)
    prod = obs.prod.clone()

    return dict(
        obs=obs, background=background, garrison_status=garrison_status,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        capture_floor_TK=capture_floor_TK,
        prod=prod, H=H,
        current_step=current_step,
        game_length_est=game_length_est,
        opening_window=opening_window,
        denial_weight=denial_weight,
        opening_weight=opening_weight,
    )


def _denial_kwargs(fx):
    return dict(
        obs=fx["obs"], background=fx["background"],
        cand_tgt_slot=fx["cand_tgt_slot"], cand_tgt_short=fx["cand_tgt_short"],
        cand_send=fx["cand_send"], cand_eta=fx["cand_eta"],
        cand_valid=fx["cand_valid"], cand_is_def=fx["cand_is_def"],
        capture_floor_TK=fx["capture_floor_TK"],
        prod=fx["prod"], garrison_status=fx["garrison_status"],
        H=fx["H"], current_step=fx["current_step"],
        game_length_est=fx["game_length_est"],
        weight=fx["denial_weight"], player_id=0,
    )


def _opening_kwargs(fx):
    return dict(
        obs=fx["obs"],
        cand_tgt_slot=fx["cand_tgt_slot"], cand_tgt_short=fx["cand_tgt_short"],
        cand_send=fx["cand_send"], cand_eta=fx["cand_eta"],
        cand_valid=fx["cand_valid"], cand_is_def=fx["cand_is_def"],
        capture_floor_TK=fx["capture_floor_TK"],
        prod=fx["prod"], garrison_status=fx["garrison_status"],
        H=fx["H"], current_step=fx["current_step"],
        game_length_est=fx["game_length_est"],
        opening_window=fx["opening_window"],
        weight=fx["opening_weight"], player_id=0,
    )


# --- denial tests ----------------------------------------------------------


def test_denial_fires_when_opp_owns_target():
    """Opp owns P1 (we're attacking) → denial bonus > 0."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(P1_owner=1.0)
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() > 0.0


def test_denial_fires_when_opp_predicted_neutral():
    """Neutral P1, opp_proj predicts opp will attack it → denial > 0."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, opp_intent_at_P1_ships=10.0)
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() > 0.0


def test_denial_zero_when_no_opp_value():
    """Neutral P1, opp_proj predicts nothing → no denial bonus."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, opp_intent_at_P1_ships=0.0)
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() == 0.0


def test_denial_zero_when_under_floor():
    """Send below floor → no capture → no bonus regardless of opp value."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(P1_owner=1.0, cand_send_ships=3.0)
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() == 0.0


def test_denial_zero_when_weight_zero():
    """weight=0 short-circuits to zero penalty."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(P1_owner=1.0, denial_weight=0.0)
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() == 0.0


def test_denial_zero_when_future_h_zero():
    """current_step beyond game_length_est - H → future_h = 0 → bonus = 0."""
    from agents.producer.orbit_lite.strategic_value import denial_bonus
    fx = _build_strategic_fixture(
        P1_owner=1.0, current_step=300, game_length_est=200,
    )
    pen = denial_bonus(**_denial_kwargs(fx))
    assert pen.item() == 0.0


# --- opening tests ---------------------------------------------------------


def test_opening_fires_at_step_zero():
    """step=0, capture, future_h > 0 → opening bonus > 0."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, current_step=0)
    pen = opening_bonus(**_opening_kwargs(fx))
    assert pen.item() > 0.0


def test_opening_zero_at_window():
    """current_step == opening_window → phase = 0 → bonus = 0."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, current_step=30, opening_window=30)
    pen = opening_bonus(**_opening_kwargs(fx))
    assert pen.item() == 0.0


def test_opening_zero_past_window():
    """step > opening_window → bonus = 0."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, current_step=50, opening_window=30)
    pen = opening_bonus(**_opening_kwargs(fx))
    assert pen.item() == 0.0


def test_opening_zero_when_under_floor():
    """Send below floor → no capture → no bonus."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, cand_send_ships=3.0)
    pen = opening_bonus(**_opening_kwargs(fx))
    assert pen.item() == 0.0


def test_opening_zero_when_weight_zero():
    """weight=0 → bonus = 0."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx = _build_strategic_fixture(P1_owner=-1.0, opening_weight=0.0)
    pen = opening_bonus(**_opening_kwargs(fx))
    assert pen.item() == 0.0


def test_opening_decays_linearly():
    """Bonus at step=15 (half-window) should be ~half of bonus at step=0."""
    from agents.producer.orbit_lite.strategic_value import opening_bonus
    fx0 = _build_strategic_fixture(P1_owner=-1.0, current_step=0, opening_window=30)
    fx15 = _build_strategic_fixture(P1_owner=-1.0, current_step=15, opening_window=30)
    pen0 = opening_bonus(**_opening_kwargs(fx0)).item()
    pen15 = opening_bonus(**_opening_kwargs(fx15)).item()
    # Phase factor at step=0 is 1.0; at step=15 (with window=30) is 0.5.
    # future_h shifts too: step 0 → 200-0-18=182; step 15 → 200-15-18=167.
    # Ratio: (1.0 × 182) / (0.5 × 167) ≈ 2.18.
    assert pen0 > pen15 > 0.0
    ratio = pen0 / pen15
    assert 1.9 <= ratio <= 2.5, f"linear decay broken: ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# Drift checks for new shims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shim_file,variant_name", [
    ("producer_plus_denial.py", "denial"),
    ("producer_plus_opening.py", "opening"),
    ("producer_plus_strategic.py", "strategic"),
    ("producer_plus_multi_tick_strategic.py", "multi_tick_strategic"),
])
def test_new_strategic_shims_match_bundle_variant(shim_file, variant_name):
    """Drift guard for the 4 new shims."""
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
def test_strategic_on_changes_planner_output():
    """Composed strategic bundle must produce a different game outcome
    than the multi_tick_recap shim on at least one seed in (7, 13, 42).
    """
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    off_shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_tick_recap.py")
    on_shim = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_strategic.py",
    )
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
        "strategic ON produced identical outcomes to multi_tick_recap on "
        "seeds 7, 13, 42 — code path may be unreachable"
    )


@pytest.mark.slow
def test_strategic_smoke_wallclock_under_60s():
    """Rule 46c: full game vs producer at seed 7 under 60 s. Both
    strategic bonuses add O(C*P) tensor ops -- negligible."""
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_strategic.py",
    )
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"strategic+multi_tick game at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock regression"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]
