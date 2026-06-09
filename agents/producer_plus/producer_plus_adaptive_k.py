"""Adaptive-K-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_ADAPTIVE_K=1`` BEFORE loading the producer_plus
agent module, so the planner picks up the adaptive-K env-gate at import
time. Mirrors the pattern in ``submissions/champ_adaptiveK_on.py``. See
``state/MIGRATION_PLAN.md`` Step 2.

We load ``producer_agent.py`` via ``importlib`` (rather than ``import``)
because the fast.py / kaggle_environments loader resolves agents by file
path and does not put their directory on ``sys.path`` — a bare
``import producer_agent`` from this file would fail.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_ADAPTIVE_K", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_adaptive_k_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
