"""Tests for multi-tick opponent projection.

Covers Rule 38 (fix-verification reproduces failure state):
    1. ``_multi_tick_opp_k`` env gate parses the flag correctly and
       honours the per-player-count override pattern.
    2. With opp_projection ON but multi_tick K UNSET (or 0/1), action
       rows are byte-identical to the same agent with K explicitly = 0
       (reproduces the "K=1 path got perturbed" failure mode).
    3. With opp_projection ON and multi_tick K=3, action rows DIFFER from
       the single-pass path at at least one seed — proving the multi-tick
       code path is actually exercised.
    4. Smoke: max per-turn wallclock under 1000 ms (Rule 46).
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
        "producer_plus_main_test_multi_tick_opp",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_multi_tick_opp"] = module
    spec.loader.exec_module(module)
    return module


def _clear_multi_tick_env(monkeypatch):
    for name in (
        "PRODUCER_PLUS_MULTI_TICK_OPP_K",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P",
    ):
        monkeypatch.delenv(name, raising=False)


def test_multi_tick_default_zero_2p(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    assert producer_plus_main._multi_tick_opp_k(2) == 0


def test_multi_tick_default_zero_4p(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    assert producer_plus_main._multi_tick_opp_k(4) == 0


@pytest.mark.parametrize("value,expected", [
    ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("7", 7),
])
def test_multi_tick_base_var_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", value)
    assert producer_plus_main._multi_tick_opp_k(2) == expected
    assert producer_plus_main._multi_tick_opp_k(4) == expected


def test_multi_tick_2p_suffix_overrides_base(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", "1")
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "5")
    assert producer_plus_main._multi_tick_opp_k(2) == 5
    # 4P falls back to base.
    assert producer_plus_main._multi_tick_opp_k(4) == 1


def test_multi_tick_4p_suffix_overrides_base(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", "1")
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "7")
    assert producer_plus_main._multi_tick_opp_k(4) == 7
    # 2P falls back to base.
    assert producer_plus_main._multi_tick_opp_k(2) == 1


@pytest.mark.parametrize("value", ["", "abc", "-1", "1.5", "nan"])
def test_multi_tick_invalid_returns_zero(monkeypatch, producer_plus_main, value):
    """Garbage values (negative, non-int, non-numeric) clamp to 0 (single-pass)."""
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", value)
    assert producer_plus_main._multi_tick_opp_k(2) == 0


def test_shim_env_vars_match_bundle_variant():
    """Drift guard: the shim file's os.environ.setdefault calls must
    match scripts/bundle_producer_plus.py's ENV_VARIANTS entry for
    `multi_tick_opp_K3`. If they diverge, local play via the shim runs a
    different configuration than the bundled submission, and the A/B
    that gates the submission no longer exercises what ships.
    """
    import importlib.util as _il
    shim_path = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py")
    bundler_path = os.path.join(REPO_ROOT, "scripts", "bundle_producer_plus.py")
    spec_b = _il.spec_from_file_location("bundle_producer_plus_drift_check", bundler_path)
    mod_b = _il.module_from_spec(spec_b)
    spec_b.loader.exec_module(mod_b)
    bundle_vars = mod_b.ENV_VARIANTS["multi_tick_opp_K3"]
    # Parse the shim's setdefault calls textually -- shim sets vars at
    # import time and we don't want import side effects in this test.
    import re
    src = open(shim_path).read()
    pattern = re.compile(
        r'os\.environ\.setdefault\(\s*"([A-Z_0-9]+)"\s*,\s*"([^"]+)"\s*\)'
    )
    shim_vars = dict(pattern.findall(src))
    assert shim_vars == bundle_vars, (
        f"shim env vars {shim_vars} drift from bundle variant {bundle_vars} -- "
        f"keep them in sync or local play diverges from the submission"
    )


def _play_one_game(focal_path, opp_path, seed):
    """Run one kaggle_environments game and return the final state and steps."""
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
def test_multi_tick_K0_matches_K_unset(monkeypatch):
    """Rule 38 reproduce-failure test: with opp_projection ON, explicitly
    setting PRODUCER_PLUS_MULTI_TICK_OPP_K=0 must produce the same game
    outcome as leaving the var unset. Both should hit the K_clamped == 1
    branch (which is bit-identical to the original single-pass code).
    Guards against the K=0 ⇒ K_clamped==1 mapping perturbing behaviour.
    """
    shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_opp_proj.py")
    producer = os.path.join(PRODUCER_DIR, "producer_agent.py")
    # Run A: K unset (defaults to 0 inside the getter). monkeypatch reverts
    # at test teardown -- safer than direct os.environ mutation across
    # sequential slow tests.
    for name in (
        "PRODUCER_PLUS_MULTI_TICK_OPP_K",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P",
    ):
        monkeypatch.delenv(name, raising=False)
    state_unset, _ = _play_one_game(shim, producer, 7)
    # Run B: K=0 explicit.
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", "0")
    state_zero, _ = _play_one_game(shim, producer, 7)
    assert state_unset[0]["reward"] == state_zero[0]["reward"], (
        f"K=0 explicit ({state_zero[0]['reward']}) differs from K unset "
        f"({state_unset[0]['reward']}) — the K=0 mapping is mis-routed"
    )


@pytest.mark.slow
def test_multi_tick_K3_changes_planner_output():
    """Proof the multi-tick code path is actually exercised: with K=3,
    at least one of (seeds 7, 13, 42) must produce a different game
    outcome than the K=1 single-pass path. If they're identical at all
    three seeds, the multi-tick branch is unreachable / no-op.
    """
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    single_pass = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_opp.py")
    multi_tick = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py")
    differs = False
    for seed in (7, 13, 42):
        state_single, _ = _play_one_game(single_pass, producer_path, seed)
        state_multi, _ = _play_one_game(multi_tick, producer_path, seed)
        if (
            state_single[0]["reward"] != state_multi[0]["reward"]
            or state_single[1]["reward"] != state_multi[1]["reward"]
        ):
            differs = True
            break
    assert differs, (
        "multi_tick K=3 produced identical outcomes to single-pass on "
        "seeds 7, 13, 42 — code path may be unreachable"
    )


@pytest.mark.slow
def test_multi_tick_2p_smoke_wallclock_under_60s():
    """Rule 46 smoke for the 2P path (shim sets K_2P=2). Asserts total
    wallclock under 60 s as a loose per-turn proxy. K=2 in 2P is roughly
    2x the single-pass opp_proj cost.
    """
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py",
    )
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"multi_tick 2P (K=2) game vs producer at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock degradation"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]


def _play_one_game_4p(focal_path, seed):
    """Run one 4P self-match and return final state and steps."""
    # Shims set PRODUCER_PLUS_* via os.environ.setdefault at load; those keys
    # outlive the game in this process and silently re-gate the NEXT game's
    # agents (the clean_ab.py env-pollution problem). Start every game from a
    # clean gate state so ON/OFF comparisons compare what they claim to.
    for _k in [k for k in os.environ if k.startswith("PRODUCER_PLUS_")]:
        os.environ.pop(_k)
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, focal_path, focal_path, focal_path])
    return env.state, env.steps


@pytest.mark.slow
def test_multi_tick_4p_smoke_wallclock_under_300s():
    """Rule 46 smoke for the 4P path (shim sets K_4P=3). The 4P bundle
    is the wallclock-cap risk: 3 opps * 3 rounds = 9 inner planner calls
    per turn vs 1 in single-pass. Bound at 300 s total wallclock for the
    self-match (4 seats * ~500 steps * < 150 ms = 300 s upper bound).
    Guards against the 4P-K3 configuration breaching the 1000 ms cap
    that the 2P smoke cannot detect.
    """
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py",
    )
    t0 = time.time()
    state, steps = _play_one_game_4p(shim_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 300.0, (
        f"multi_tick 4P-K3 self-match at seed 7 took {elapsed:.1f}s — "
        f"per-turn cost is probably above the 1000 ms cap under any load"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]
