"""Source-exposure-ON variant of ``producer_plus``.

Sets ``PRODUCER_PLUS_OPP_PROJECTOR=lite_greedy`` AND
``PRODUCER_PLUS_SOURCE_EXPOSURE=1`` so the planner rejects candidates
whose launch would leave the source planet defenseless against
projected opp arrivals. See ``state/MIGRATION_PLAN.md`` — opp-foresight
Mechanism 1.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTOR", "lite_greedy")
os.environ.setdefault("PRODUCER_PLUS_SOURCE_EXPOSURE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_source_exposure_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
