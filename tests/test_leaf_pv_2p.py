"""Unit tests for BASELINE_LEAF_PV_2P — 2P composite-leaf production-PV gate.

The peak baseline's 2P leaf (`composite_capture_value` in `lib/value_heads.py`)
hard-disables its per-planet production-PV term by default (the
`_COMPOSITE_PV_ENABLED` gate, off since 2026-05-18). Mid-game silent turns on
seed=2 vs v4_planner (handover 2026-05-28 PM) were traced to this — the leaf
is dimensionally myopic without it, since captures earn zero credit for the
future production stream beyond the rollout horizon.

The new alias env var `BASELINE_LEAF_PV_2P=1` re-enables the same code path
that `COMPOSITE_PRODUCTION_PV=1` already controlled. Either var flips the
gate; the namespaced `BASELINE_*` form is the new public knob.

Rule 38 fix-verification: the "captures-register-at-leaf" test below is the
contrived state where current (OFF) code returns ≈ ship-delta only and the
fix (ON) returns ship-delta + production × pv_horizon. It reproduces the
failure mode the patch addresses.
"""

from __future__ import annotations

import importlib
import os

import pytest

from lib.scoring import pv_horizon


def _obs(planets, fleets=None, step=0, player=0):
    return {
        "player": player,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
    }


def _reload_vh():
    import lib.value_heads as vh
    importlib.reload(vh)
    return vh


def _clear_env():
    os.environ.pop("BASELINE_LEAF_PV_2P", None)
    os.environ.pop("COMPOSITE_PRODUCTION_PV", None)


# Planet tuple: (id, owner, x, y, radius, ships, production)


def test_default_off_byte_identical():
    """Both gate vars unset → byte-identical to peak's pre-2026-05-28 leaf.

    With no fleets, no PV term, no per-fleet credit, `composite_capture_value`
    collapses to the raw ship-delta: 30 - 10 = 20.0. Pins the peak-preservation
    contract — any code change that breaks this assertion has changed the
    default-OFF leaf and therefore changed peak behavior.
    """
    _clear_env()
    try:
        vh = _reload_vh()
        assert vh._COMPOSITE_PV_ENABLED is False, "default must be OFF"
        obs = _obs([
            (0, 0, 10, 50, 1.0, 30, 2),  # me: ships=30, prod=2
            (1, 1, 90, 50, 1.0, 10, 2),  # opp: ships=10, prod=2
        ])
        v = vh.composite_capture_value(obs, my_id=0)
        assert v == pytest.approx(20.0, abs=1e-9)
    finally:
        _clear_env()


def test_on_adds_pv_term_asymmetric():
    """BASELINE_LEAF_PV_2P=1 adds `(my_prod - opp_prod) × pv_horizon`.

    Asymmetric production (me=3, opp=1) at step=0 → ΔPV-term = 2 × ~99.34.
    Total leaf = 20 (ship-delta) + 198.68 (PV) ≈ 218.68.
    """
    _clear_env()
    os.environ["BASELINE_LEAF_PV_2P"] = "1"
    try:
        vh = _reload_vh()
        assert vh._COMPOSITE_PV_ENABLED is True
        obs = _obs([
            (0, 0, 10, 50, 1.0, 30, 3),  # me: ships=30, prod=3
            (1, 1, 90, 50, 1.0, 10, 1),  # opp: ships=10, prod=1
        ], step=0)
        expected_pv = pv_horizon(0, 0, gamma=0.99, t_total=500)
        v = vh.composite_capture_value(obs, my_id=0)
        # ship-delta = 30 - 10 = 20; pv-delta = (3 - 1) * pv ≈ 198.68
        assert v == pytest.approx(20.0 + 2.0 * expected_pv, abs=1e-6)
    finally:
        _clear_env()


def test_alias_compat_with_legacy_var():
    """COMPOSITE_PRODUCTION_PV=1 still works; BASELINE_LEAF_PV_2P=1 gives the
    same numerical leaf. Repro-friendly: bisecting on old branches with the
    legacy var keeps working.
    """
    obs = _obs([
        (0, 0, 10, 50, 1.0, 30, 3),
        (1, 1, 90, 50, 1.0, 10, 1),
    ], step=0)

    _clear_env()
    os.environ["COMPOSITE_PRODUCTION_PV"] = "1"
    try:
        vh = _reload_vh()
        v_legacy = vh.composite_capture_value(obs, my_id=0)
    finally:
        _clear_env()

    _clear_env()
    os.environ["BASELINE_LEAF_PV_2P"] = "1"
    try:
        vh = _reload_vh()
        v_alias = vh.composite_capture_value(obs, my_id=0)
    finally:
        _clear_env()

    assert v_legacy == pytest.approx(v_alias, abs=1e-12)


def test_captures_register_at_leaf_rule_38_fix_verification():
    """Rule 38 fix-verification: reproduce the silent-turn failure state.

    Contrived leaf where me has just captured opp's last planet (2 of my
    planets prod=2 each, opp has 0 planets/ships, step=100). The chooser
    rollout reaches this leaf. Under OFF (current peak), the leaf scores
    just the ship-delta. Under ON, the leaf adds `(4 - 0) × pv_horizon` —
    the future-production stream credit that the OFF leaf is blind to.

    This blindness is the mechanism behind the silent-turn pathology
    diagnosed in 2026-05-28-silent-turns-pre-existing-weakness.md.
    """
    obs = _obs([
        (0, 0, 10, 50, 1.0, 20, 2),  # me planet 1
        (1, 0, 30, 50, 1.0, 20, 2),  # me planet 2 (just-captured from opp)
    ], step=100)

    _clear_env()
    try:
        vh = _reload_vh()
        v_off = vh.composite_capture_value(obs, my_id=0)
    finally:
        _clear_env()

    _clear_env()
    os.environ["BASELINE_LEAF_PV_2P"] = "1"
    try:
        vh = _reload_vh()
        v_on = vh.composite_capture_value(obs, my_id=0)
    finally:
        _clear_env()

    # OFF: ship-delta = 40 - 0 = 40, no PV. The capture's strategic value
    # (4 prod for 400 turns) is invisible to the leaf.
    assert v_off == pytest.approx(40.0, abs=1e-9)
    # ON: + (4 - 0) × pv_horizon(step=100, eta=0). pv at step=100 ≈ 98.2.
    expected_pv = pv_horizon(100, 0, gamma=0.99, t_total=500)
    assert v_on == pytest.approx(40.0 + 4.0 * expected_pv, abs=1e-6)
    # The lift the chooser sees on this capture-leaf is large:
    assert (v_on - v_off) > 350.0  # ≈ 4 × 98.2 ≈ 392.8
