"""Rule 38 fix-verification anchor for the kinematic-table baseline wiring.

The unit tests in `test_kinematic_table_parity.py` pin that table lookups
match scalar `predict_relative` — they prove the TABLE is correct. They
do NOT prove that the baseline agent actually CALLS `begin_turn` or that
`predict_fleet_fate` takes the table fast-path.

This file closes the silent-fail mode where the wiring exists but doesn't
engage: env var unset in some path, `begin_turn` skipped, fall-through guard
mis-predicate, etc. The friction archive
(`bundler-default-lib-order-stale-kinematic-table`,
`agent-introspection-skipped-bootstrap`) shows this exact pattern recurring
when a contract is only unit-tested.
"""

from __future__ import annotations

import os

import pytest
from kaggle_environments import make

import lib.kinematic_table as kt
from lib.kinematic_table import get_default


def _first_obs():
    """Return (obs_dict, configuration_dict) for the seed-0 initial state.

    Uses the env's initial step so we get a realistic planet/fleet layout
    instead of synthesising a hand-crafted one.
    """
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset()
    step0 = env.steps[0][0]
    obs = step0["observation"]
    return obs, env.configuration


def test_baseline_primes_table_on_first_call(monkeypatch):
    """After one call to `agent(obs)`, the singleton table must report
    n_planets > 0 — proof that `begin_turn(world)` ran inside the agent."""
    monkeypatch.setenv("KINEMATIC_TABLE_ENABLED", "1")
    kt.clear()
    assert get_default().stats()["n_planets"] == 0

    from agents.baseline.main import agent
    obs, cfg = _first_obs()
    agent(obs, cfg)

    stats = get_default().stats()
    assert stats["n_planets"] > 0, (
        "kinematic table singleton stayed empty after agent(obs); "
        "begin_turn(world) didn't run — wiring is broken."
    )


def test_baseline_engages_table_via_predict_fleet_fate(monkeypatch):
    """The agent's chooser uses `predict_fleet_fate`, which goes through
    `_table_window_or_none` — that path MUST be hit at least once per turn
    when the table is primed and the env var is on. We spy on
    `KinematicTable.window` to verify."""
    monkeypatch.setenv("KINEMATIC_TABLE_ENABLED", "1")
    kt.clear()

    calls = {"n": 0}
    orig_window = kt.KinematicTable.window

    def counting_window(self, pids, *, start_offset, length):
        calls["n"] += 1
        return orig_window(self, pids, start_offset=start_offset, length=length)

    monkeypatch.setattr(kt.KinematicTable, "window", counting_window)

    from agents.baseline.main import agent
    obs, cfg = _first_obs()
    agent(obs, cfg)

    assert calls["n"] > 0, (
        "KinematicTable.window was never called inside agent(obs); "
        "predict_fleet_fate is taking the inline-list-comp fallback even "
        "though the table is primed. Check KINEMATIC_TABLE_ENABLED handling "
        "in lib/trajectory._table_window_or_none."
    )


def test_baseline_falls_through_when_env_var_off(monkeypatch):
    """Rule 38 negative: with KINEMATIC_TABLE_ENABLED disabled, the agent
    must still run (the inline list-comp fallback is the safety net), and
    the spy on `KinematicTable.window` must record ZERO calls. If this test
    passes when env-var-off but the positive test ALSO passes when env-var-on,
    we have a real contract: table-on engages, table-off falls through."""
    monkeypatch.setenv("KINEMATIC_TABLE_ENABLED", "0")
    kt.clear()

    calls = {"n": 0}
    orig_window = kt.KinematicTable.window

    def counting_window(self, pids, *, start_offset, length):
        calls["n"] += 1
        return orig_window(self, pids, start_offset=start_offset, length=length)

    monkeypatch.setattr(kt.KinematicTable, "window", counting_window)

    from agents.baseline.main import agent
    obs, cfg = _first_obs()
    actions = agent(obs, cfg)

    assert isinstance(actions, list), "agent didn't return a list under KT-off"
    assert calls["n"] == 0, (
        f"KinematicTable.window was called {calls['n']} times even though "
        "KINEMATIC_TABLE_ENABLED=0. The fall-through guard is leaking."
    )
