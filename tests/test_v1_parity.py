"""Parity gate for the Step 3.5.A refactor.

The refactored v1 (agents/v1_orbitfix/main.py — uses lib/intent + lib/mechanism)
must produce identical rewards to the pre-refactor snapshot
(tests/fixtures/v1_pre_refactor.py — the v1 from commit 17fb9aa).

If any seed shows reward drift, the refactor leaked. Bisect by mechanism set:
- mechanisms=[] vs mechanisms=[validate] vs mechanisms=[validate, lead_aim].
- Suspect `random.Random` seed/order divergence first; second suspect is
  intent emission-order vs original launch-order.

Runs 10 fixed seeds × P0-side. ~60-90s runtime; kept short so this gate
runs on every pre-commit hook.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pre_refactor_v1():
    return _load("v1_pre_refactor", REPO / "tests" / "fixtures" / "v1_pre_refactor.py")


@pytest.fixture(scope="module")
def current_v1():
    return _load("v1_current", REPO / "agents" / "v1_orbitfix" / "main.py")


# First 10 seeds from scripts/eval_v1.py SEEDS_20 — the seed bag used for the
# Step 3 winrate gate.
SEEDS_PARITY = [42, 1, 7, 13, 31, 100, 17, 23, 53, 71]


@pytest.mark.parametrize("seed", SEEDS_PARITY)
def test_v1_rewards_match_pre_refactor_vs_baseline(pre_refactor_v1, current_v1, seed):
    baseline = str(REPO / "data" / "main.py")
    env_old = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env_old.run([pre_refactor_v1.agent, baseline])
    env_new = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env_new.run([current_v1.agent, baseline])
    r_old = [s.reward for s in env_old.steps[-1]]
    r_new = [s.reward for s in env_new.steps[-1]]
    assert r_old == r_new, (
        f"seed={seed}: refactored v1 diverged from pre-refactor — "
        f"old={r_old}, new={r_new}. Bisect via mechanism set."
    )
