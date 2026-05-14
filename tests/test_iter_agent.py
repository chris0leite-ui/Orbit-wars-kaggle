"""Tests for the iter fast-iteration scaffold.

The iter agent (`agents/iter/main.py`) is a thin fork of v7_pv. Two
invariants we guard:

1. The module exports a callable `agent` and completes a 1-game smoke
   against a stock kaggle_environments opponent without crashing.
2. Importing `agents.iter.main` enforces `lib.scoring.PV_GAMMA == 0.99`,
   which is what makes day-zero iter functionally equivalent to v7_pv.
   v7_pv was never a committed source file (it was a bundled artifact
   produced with `scripts/ab_variants.py --variant pv PV_GAMMA=0.99`),
   so the parity contract lives in this PV_GAMMA invariant plus
   `tests/test_jax_pv_horizon_parity.py` which verifies the PV math.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture()
def iter_agent_module():
    # Force a fresh import: PV_GAMMA override only fires on first load.
    for name in list(sys.modules):
        if name.startswith("agents.iter") or name == "lib.scoring":
            del sys.modules[name]
    mod = importlib.import_module("agents.iter.main")
    return mod


def test_iter_agent_callable(iter_agent_module):
    assert callable(iter_agent_module.agent)


def test_iter_pv_gamma_is_enforced(iter_agent_module):
    import lib.scoring as scoring
    assert scoring.PV_GAMMA == 0.99, (
        f"iter must enforce PV_GAMMA=0.99 (v7_pv parity); got {scoring.PV_GAMMA}"
    )
    # Also assert the agent module's own constant matches.
    assert iter_agent_module.PV_GAMMA == 0.99


def test_iter_knob_constants_present(iter_agent_module):
    # If any knob name disappears, downstream sweep scripts will need
    # updating — surface it here instead of via silent agent regression.
    for knob in ("K", "WALLCLOCK_MS", "ENUMERATOR_MODE", "OPP_TIERS",
                 "PV_GAMMA", "VALUE_FN"):
        assert hasattr(iter_agent_module, knob), f"missing knob: {knob}"


def test_iter_smoke_one_game_completes(iter_agent_module):
    # 1-game self-play smoke (iter vs the kaggle_environments `random`
    # builtin). Cheap — should run in <30 s on local CPU.
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.run([iter_agent_module.agent, "random"])
    assert env.done, "env did not reach DONE"
    statuses = [s["status"] for s in env.steps[-1]]
    assert all(s in {"DONE", "INACTIVE"} for s in statuses), (
        f"unexpected terminal statuses: {statuses}"
    )
