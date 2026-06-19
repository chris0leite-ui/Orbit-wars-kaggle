"""Anytime-safety guard for deep-search (LR_ROLLOUT_DEPTH>=2).

The depth-3 deep-search submit (53836276) ERRORED on the ladder: a single
per-turn overrun past the 1000ms wall, because (a) the producer rollout opponent
is too expensive to call O(depth*seats*candidates)/turn and (b) the fixed-depth
rollout had no intra-rollout deadline. The shipped fix is the CHEAP lite_greedy
rollout opponent (LR_DEEP_OPP=1) plus anytime deadline guards.

This test reproduces the overrun *mechanism* on local hardware by setting a tight
per-turn budget and asserting depth-3 search tracks the budget (+ one-step slack)
instead of running unbounded. Local CPU is faster than Kaggle's, so passing here
is necessary-not-sufficient for the 1000ms wall -- but a regression that removes
the guard will blow this bound loudly.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LR = str(REPO / "agents" / "least_resistance" / "main.py")
PRODUCER = str(REPO / "agents" / "producer" / "main.py")
BUDGET_MS = 150.0
# Generous slack: the guard may overshoot by up to one cheap rollout step plus
# the leaf projection. Well under the 1000ms wall, and far below the unbounded
# multi-second overrun the producer opponent produces without the cheap model.
SLACK_MS = 550.0


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "_t_%s_%d" % (Path(path).stem, id(object())), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


@pytest.mark.parametrize("iterdeepen", ["0", "1"])
def test_depth3_respects_tight_budget(iterdeepen):
    """depth-3 + cheap opponent must not overrun the tight per-turn budget,
    on both the fixed-depth and iterative-deepening paths."""
    os.environ["LR_ROLLOUT_DEPTH"] = "3"
    os.environ["LR_DEEP_OPP"] = "1"            # cheap lite_greedy rollout opponent
    os.environ["LR_ITERDEEPEN"] = iterdeepen
    os.environ["LR_DEEP_EXTRA_CAP_MS"] = "0"   # overage bank OFF -> budget = base
    os.environ["LR_DEEP_BANK_FRAC"] = "0"
    os.environ["LR_WALLCLOCK_MS"] = str(BUDGET_MS)
    os.environ.pop("ORBIT_WARS_PARITY_WALLCLOCK_MS", None)

    from kaggle_environments import make

    focal = _load(LR)
    opp = _load(PRODUCER)
    ts = []

    def timed(o, c=None):
        t = time.perf_counter()
        try:
            return focal(o, c)
        finally:
            ts.append((time.perf_counter() - t) * 1000.0)

    env = make("orbit_wars",
               configuration={"seed": 76670184, "episodeSteps": 60}, debug=False)
    env.run([timed, opp, opp, opp])

    assert ts, "focal agent never moved"
    mx = max(ts)
    assert mx <= BUDGET_MS + SLACK_MS, (
        "depth-3 search overran the per-turn budget: max=%.0fms > %.0f+%.0f. "
        "The anytime deadline guard or the cheap rollout opponent regressed."
        % (mx, BUDGET_MS, SLACK_MS))
