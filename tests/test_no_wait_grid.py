"""Positive lock: no wait-grid mechanism in agents/baseline/.

The wait-grid (`wait_then_fire_variants`, `min_wait_affordable`,
`wait_band`, `WAIT_GRID_MODE`, `WAIT_EXTRA_SURPLUS`) and the
`_PENDING_LAUNCHES` / `_tick_ledger` ledger machinery were stripped
on 2026-05-29 because they generated and scored candidates that were
silently discarded (ledger off at peak). See state/PEAK_BASELINE.md.

These tests fire if anyone tries to re-introduce the mechanism without
designing the commit semantics deliberately. The user's strategic
framing: every turn is a fresh decision; committing to a future launch
is wrong semantics. Re-introducing should require a separate plan,
not a stealth revert.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_proposer_has_no_wait_grid_symbols():
    """No producers, helpers, or env-var hooks for wait-N candidates."""
    import agents.baseline.proposer as p
    forbidden = (
        "wait_then_fire_variants",
        "wait_then_fire_variants_forward",
        "min_wait_affordable",
        "wait_band",
        "WAIT_GRID_MODE",
        "WAIT_EXTRA_SURPLUS",
        "WAIT_BUFFER_OFFSET",
    )
    present = [name for name in forbidden if hasattr(p, name)]
    assert not present, (
        f"Wait-grid symbols re-introduced into agents.baseline.proposer: "
        f"{present}. See state/PEAK_BASELINE.md (the 2026-05-29 strip)."
    )


def test_main_has_no_ledger_symbols():
    """No ledger flag, mode, pending-launches dict, or tick function."""
    import agents.baseline.main as m
    forbidden = (
        "LEDGER_ENABLED",
        "LEDGER_MODE",
        "_PENDING_LAUNCHES",
        "_tick_ledger",
    )
    present = [name for name in forbidden if hasattr(m, name)]
    assert not present, (
        f"Ledger symbols re-introduced into agents.baseline.main: "
        f"{present}. See state/PEAK_BASELINE.md (the 2026-05-29 strip)."
    )


def test_choose_trajectory_returns_moves_only():
    """`choose_trajectory()` returns `moves` (a list), not `(moves, commits)`."""
    from agents.baseline.chooser_trajectory import choose_trajectory
    snap_base = SimpleNamespace()
    out = choose_trajectory(
        snap_base=snap_base, prerank=[], baseline_favors=[],
        me=0, num_seats=2, wallclock_ms=10.0,
        min_horizon=20, max_horizon=40, gamma=0.99,
        world=None, model=None,
    )
    assert isinstance(out, list), (
        f"choose_trajectory must return a list of moves, got {type(out)}. "
        "Re-introducing a (moves, commits) tuple revives the dead ledger path."
    )


def test_composite_chooser_returns_moves_only():
    """`agents.baseline.chooser.choose()` returns `moves` (a list)."""
    from agents.baseline.chooser import choose
    snap_base = SimpleNamespace()
    out = choose(
        snap_base=snap_base, prerank=[], baseline_favors=[],
        me=0, num_seats=2, wallclock_ms=10.0,
        min_horizon=20, max_horizon=40, gamma=0.99,
        world=None,
    )
    assert isinstance(out, list), (
        f"chooser.choose must return a list of moves, got {type(out)}."
    )


def test_choose_trajectory_signature_has_no_commit_kwargs():
    """No `reserved_for_new_commits` parameter — that was the ledger hook."""
    import inspect
    from agents.baseline.chooser_trajectory import choose_trajectory
    sig = inspect.signature(choose_trajectory)
    forbidden = {"reserved_for_new_commits"}
    overlap = set(sig.parameters) & forbidden
    assert not overlap, (
        f"choose_trajectory has commit-related kwargs back: {overlap}. "
        "See state/PEAK_BASELINE.md (the 2026-05-29 strip)."
    )


def test_dormant_state_test_no_longer_pins_ledger():
    """The original 'ledger defaults off' test was removed by the strip.
    Guard against accidental revert.
    """
    import tests.test_peak_dormant_state as tpds
    assert not hasattr(tpds, "test_ledger_disabled_at_default"), (
        "test_ledger_disabled_at_default came back; the env var it pins "
        "no longer exists post-strip."
    )
