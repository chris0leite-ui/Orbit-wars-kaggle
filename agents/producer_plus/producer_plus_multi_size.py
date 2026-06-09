"""Multi-size-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_MULTI_SIZE=1`` BEFORE loading the producer_plus
agent so the env gate is picked up at import time. See
``state/MIGRATION_PLAN.md`` Step 4.

Adaptive-K (Step 2) is deliberately NOT enabled here: 16-game
seat-alternated A/B 2026-06-04 showed adaptive_k at exactly 8/16 vs
vanilla producer (parity, no lift). Keep the adaptive_k code path in
``main.py`` for future tuning, but stop carrying it by default in the
multi-size shim until we have evidence it composes positively with
producer's calibration.

Uses ``importlib`` (rather than ``import``) because fast.py /
kaggle_environments load agents by file path and do not put their
directory on ``sys.path`` — a bare ``import producer_agent`` would
fail. Same pattern as ``producer_plus_adaptive_k.py``.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_multi_size_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
