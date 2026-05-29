"""Lock: ledger gone, patience preserved.

The 2026-05-29 wait-grid strip was over-aggressive: it deleted both
the broken `_PENDING_LAUNCHES` ledger machinery (correct) AND the
wait-N candidate generation that implements the agent's per-turn
patience signal (wrong). The 2026-05-29 surgical revert restored the
patience side-effect (wait-N candidates win the chooser's score loop,
reserve `used_srcs`/`used_tgts`, emit nothing — next turn re-decides
with fresh state) while keeping the ledger and commit-dict gone.

These tests pin both halves of that design:

  GONE  — the cross-turn persistence (LEDGER_ENABLED, _PENDING_LAUNCHES,
          _tick_ledger, the (moves, commits) return, the
          reserved_for_new_commits chooser parameter).
  KEPT  — the wait-N producers (wait_then_fire_variants,
          min_wait_affordable, wait_band, WAIT_GRID_MODE) and the
          chooser emit-loop's pre-check `used_srcs.add(sid)` /
          `used_tgts.add(tid)` reservation, which BLOCKS subsequent
          fire-now from the same source when wait-N wins.

Flipping any of these is a design change, not a refactor.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def test_no_ledger_symbols_in_main():
    """The cross-turn persistence machinery stays deleted."""
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
        f"{present}. Cross-turn persistence was deleted 2026-05-29; "
        "see state/PEAK_BASELINE.md."
    )


def test_wait_grid_producers_preserved_in_proposer():
    """Patience: wait-N producers must still exist so wait-N candidates
    enter prerank and can win the chooser's score loop."""
    import agents.baseline.proposer as p
    required = (
        "wait_then_fire_variants",
        "min_wait_affordable",
        "wait_band",
        "WAIT_GRID_MODE",
    )
    missing = [name for name in required if not hasattr(p, name)]
    assert not missing, (
        f"Wait-grid producers missing from agents.baseline.proposer: "
        f"{missing}. These implement the patience signal — see "
        "state/PEAK_BASELINE.md (2026-05-29 surgical revert)."
    )


def test_choose_trajectory_returns_moves_only():
    """`choose_trajectory()` returns `moves` (a list), not `(moves, commits)`."""
    from agents.baseline.chooser_trajectory import choose_trajectory
    out = choose_trajectory(
        snap_base=SimpleNamespace(), prerank=[], baseline_favors=[],
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
    out = choose(
        snap_base=SimpleNamespace(), prerank=[], baseline_favors=[],
        me=0, num_seats=2, wallclock_ms=10.0,
        min_horizon=20, max_horizon=40, gamma=0.99,
        world=None,
    )
    assert isinstance(out, list), (
        f"chooser.choose must return a list of moves, got {type(out)}."
    )


def test_no_reserved_for_new_commits_kwarg():
    """The ledger-only kwarg is gone from both choosers."""
    from agents.baseline.chooser_trajectory import choose_trajectory
    from agents.baseline.chooser import choose
    for fn in (choose_trajectory, choose):
        sig = inspect.signature(fn)
        assert "reserved_for_new_commits" not in sig.parameters, (
            f"{fn.__name__} has reserved_for_new_commits kwarg back; "
            "that parameter only made sense with the deleted ledger."
        )
