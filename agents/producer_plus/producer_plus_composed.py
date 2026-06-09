"""Step 4 + Step 5 composed variant of ``producer_plus``.

Sets BOTH ``PRODUCER_PLUS_MULTI_SIZE=1`` AND ``PRODUCER_PLUS_COALITIONS=1``
BEFORE loading the producer_plus agent so both env gates are picked up
at import time. See ``state/MIGRATION_PLAN.md`` Steps 4 + 5.

This variant emits N=3 size variants per (source, target) AND L=2
multi-source coalitions. Single-source rows pad slot 1 with
active=False; coalition rows use safe_drain per contributor with both
slots active. Greedy's target mutex picks the highest-scoring
candidate per target across all variants and coalitions.

Motivation: Step 5 standalone (no multi-size) regressed to 13/32 =
40.6% vs producer at n=32. Hypothesis: coalitions only help when the
multi-size baseline gives the planner finer-grained alternatives to
choose against. Composed variant tests that hypothesis.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_COALITIONS", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_composed_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
