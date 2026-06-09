"""Opponent-projection-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_OPP_PROJECTION=1`` BEFORE loading the producer_plus
agent so the env gate is picked up at import time. See
``state/MIGRATION_PLAN.md`` Step 3 and ``orbit_lite/opp_projection.py``.

Multi-size (Step 4) and coalitions (Step 5 Fix-A) are deliberately NOT
enabled here: this is the Step 3a standalone variant. The point is to
test the modeling-correctness fix (opp's projected actions injected
into the per-candidate scorer) in isolation, so we can attribute any
lift to the opp-projection mechanism specifically. Composing with
Step 4 / Step 5 Fix-A is a follow-on (Step 3b) gated on this
standalone variant clearing Rule 45.

Uses ``importlib`` (rather than ``import``) because fast.py /
kaggle_environments load agents by file path and do not put their
directory on ``sys.path`` — a bare ``import producer_agent`` would
fail. Same pattern as ``producer_plus_coalitions.py``.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")

# kaggle_environments loads agent files via exec without setting __file__,
# so fall back to cwd. Matches the guard in main.py:12-17.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_opp_proj_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
