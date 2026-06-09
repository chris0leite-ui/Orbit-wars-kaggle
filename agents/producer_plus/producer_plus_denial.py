"""Standalone denial-bonus variant (with opp_proj on so denial fires).

Sets ``PRODUCER_PLUS_OPP_PROJECTION=1`` and
``PRODUCER_PLUS_DENIAL_BONUS=1`` BEFORE loading the producer_plus agent.
Tests the denial-scoring mechanism in isolation.

See agents/producer/orbit_lite/strategic_value.py for the math.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_DENIAL_BONUS", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_denial_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
