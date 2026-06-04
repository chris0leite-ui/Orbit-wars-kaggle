"""Opp-projection-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_OPP_PROJECTOR=lite_greedy`` (and ``PRODUCER_PLUS_ADAPTIVE_K=1``
to carry Step 2) BEFORE loading the producer_plus agent so both env
gates are picked up at import time. See ``state/MIGRATION_PLAN.md``
Step 3.

Uses ``importlib`` (rather than ``import``) because fast.py /
kaggle_environments load agents by file path and do not put their
directory on ``sys.path`` — a bare ``import producer_agent`` would
fail. Same pattern as ``producer_plus_adaptive_k.py``.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_ADAPTIVE_K", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTOR", "lite_greedy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_opp_proj_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
