"""Counter-capture-ON variant of ``producer_plus``.

Sets opp projector + counter-capture target seeding (mechanism 3 alone,
without mechanisms 1 + 2). For ablation; isolates the new-candidate
behaviour from the filter behaviours. See ``state/MIGRATION_PLAN.md``
opp-foresight plan.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTOR", "lite_greedy")
os.environ.setdefault("PRODUCER_PLUS_COUNTER_CAPTURE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_counter_capture_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
