"""Multi-size + opp-projection variant (no coalitions).

Sets ``PRODUCER_PLUS_MULTI_SIZE=1`` AND ``PRODUCER_PLUS_OPP_PROJECTION=1``
BEFORE loading the producer_plus agent. Coalitions stay off.

Tests whether opp_proj contributes lift over multi_size alone. Diagnostic
trace on seed 7 showed coalitions barely firing in the kitchen sink, so
the kitchen sink's gain over standalone opp_proj likely comes from
multi_size, not coalitions. This variant isolates that.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_multi_opp_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
