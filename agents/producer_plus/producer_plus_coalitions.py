"""Coalitions-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_COALITIONS=1`` BEFORE loading the producer_plus
agent so the env gate is picked up at import time. See
``state/MIGRATION_PLAN.md`` Step 5.

Multi-size (Step 4) is deliberately NOT enabled here: Step 5
standalone keeps the candidate tensor at `C_total = S*T + T*C(K,2)`
instead of the `S*T*N + T*C(K,2)*N^2` explosion that 3-size variants
per contributor would create. Compose later as Step 5b if both
mechanisms lift independently.

Uses ``importlib`` (rather than ``import``) because fast.py /
kaggle_environments load agents by file path and do not put their
directory on ``sys.path`` — a bare ``import producer_agent`` would
fail. Same pattern as ``producer_plus_multi_size.py``.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_COALITIONS", "1")

# kaggle_environments loads agent files via exec without setting __file__,
# so fall back to cwd. Matches the guard in main.py:12-17.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_coalitions_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
