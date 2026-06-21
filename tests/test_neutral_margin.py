"""Guards for the 2026-06-21 three-lens-review changes to least_resistance:

- C1: importing main.py must NOT mutate os.environ (the old os.environ.setdefault
  bake leaked the shipped config process-wide into tests / AB harnesses).
- The shipped config is resolved from _SHIP_DEFAULTS at call time (env overrides).
- Lead-gated win-equity (point 3): agent() sets a smoothed _LEAD_D in [0,1] when ON,
  None OFF; the EMA resets at a new game (step 0).
- Neutral mass margin (points 2/4): default 0.25 in 2P, OFF (LR_NEUTRAL_MARGIN_4P=0)
  in 4P; the agent runs legally with the full new stack.
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_LR_KEYS = ("LR_NATIVE_LEAF", "LR_NATIVE_REINFORCE", "LR_CONCENTRATE",
            "LR_NATIVE_OFFENSE", "LR_NEUTRAL_MARGIN", "LR_NEUTRAL_MARGIN_4P",
            "LR_LEAD_GATE", "LR_LEAD_OFFENSE_BOOST", "LR_LEAD_STEEPNESS", "LR_LEAD_EMA")


def _load_clean():
    """Load a FRESH main.py with no LR_* env vars set (so we test the code defaults)."""
    for k in list(os.environ):
        if k.startswith("LR_"):
            os.environ.pop(k, None)
    spec = importlib.util.spec_from_file_location(
        "lr_main_nm", str(REPO / "agents" / "least_resistance" / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lr_main_nm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _initial_obs(seed=1393478882, n=2):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(n)
    return env.state[0]["observation"]


def test_import_does_not_mutate_environ():
    # C1 regression guard: loading the module must add no LR_* keys to os.environ.
    for k in list(os.environ):
        if k.startswith("LR_"):
            os.environ.pop(k, None)
    _load_clean()
    leaked = [k for k in os.environ if k.startswith("LR_")]
    assert leaked == [], "main.py leaked config into os.environ: %r" % leaked


def test_cfg_resolves_ship_defaults_and_override():
    lr = _load_clean()
    # Shipped gates default ON via _SHIP_DEFAULTS (no env set).
    for k in ("LR_NATIVE_LEAF", "LR_NATIVE_REINFORCE", "LR_CONCENTRATE", "LR_NATIVE_OFFENSE"):
        assert lr._cfg(k) == "1", k
    assert lr._native_leaf() and lr._native_reinforce() and lr._concentrate() and lr._native_offense()
    # Neutral margin default 0.25 (2P read path).
    assert abs(lr._f("LR_NEUTRAL_MARGIN", 0.0) - 0.25) < 1e-9
    # A non-shipped gate stays default-OFF.
    assert lr._cfg("LR_NATIVE_BUILDER") == "0"
    assert lr._native_builder() is False
    # Explicit env overrides the ship default.
    os.environ["LR_NATIVE_LEAF"] = "0"
    try:
        assert lr._cfg("LR_NATIVE_LEAF") == "0"
        assert lr._native_leaf() is False
    finally:
        os.environ.pop("LR_NATIVE_LEAF", None)


def test_lead_gate_default_off():
    lr = _load_clean()
    assert lr._lead_gate() is False


def test_lead_gate_sets_smoothed_d_on_and_clears_off():
    lr = _load_clean()
    obs = _initial_obs()
    # OFF: defensiveness stays None.
    os.environ.pop("LR_LEAD_GATE", None)
    out = lr.agent(obs, None)
    assert isinstance(out, list)
    assert lr._lead_d() is None
    # ON: defensiveness is a smooth float in [0,1].
    os.environ["LR_LEAD_GATE"] = "1"
    try:
        out = lr.agent(obs, None)
        assert isinstance(out, list)
        d = lr._lead_d()
        assert d is not None and 0.0 <= d <= 1.0
    finally:
        os.environ.pop("LR_LEAD_GATE", None)


def test_lead_ema_resets_on_new_game():
    # A step-0 (new game) obs must reset the EMA to the raw sigmoid (no carryover),
    # so multiple games in one process don't bleed defensiveness across each other.
    lr = _load_clean()
    os.environ["LR_LEAD_GATE"] = "1"
    try:
        obs = _initial_obs()  # step 0
        lr._LEAD_D = 0.999     # stale value from a prior game
        lr._LEAD_LAST_STEP = 500
        lr.agent(obs, None)
        # After a step-0 turn the value is the raw sigmoid of the start gap, not the
        # stale 0.999 carried through the EMA.
        assert lr._LEAD_D is not None and abs(lr._LEAD_D - 0.999) > 1e-6
        assert 0.0 <= lr._LEAD_D <= 1.0
    finally:
        os.environ.pop("LR_LEAD_GATE", None)


def test_neutral_margin_4p_off_knob_default():
    lr = _load_clean()
    # 4P branch reads LR_NEUTRAL_MARGIN_4P, default 0 (margin disabled in 4P).
    assert abs(lr._f("LR_NEUTRAL_MARGIN_4P", 0.0)) < 1e-9


def test_agent_runs_legally_with_full_new_stack():
    lr = _load_clean()
    os.environ["LR_LEAD_GATE"] = "1"
    try:
        for n in (2, 4):
            obs = _initial_obs(n=n)
            out = lr.agent(obs, None)
            assert isinstance(out, list)
            for e in out:
                assert len(e) == 3  # [src_id, angle, ships]
    finally:
        os.environ.pop("LR_LEAD_GATE", None)
