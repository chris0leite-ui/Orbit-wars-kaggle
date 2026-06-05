"""Both strategic-value bonuses (denial + opening) on, with opp_proj
so denial fires fully.

Tests the combined denial+opening signal in isolation from multi-tick
and recapture (so an A/B against vanilla producer attributes any lift
to the new bonuses themselves).
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_DENIAL_BONUS", "1")
os.environ.setdefault("PRODUCER_PLUS_OPENING_BONUS", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_strategic_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
