"""Kitchen-sink variant of ``producer_plus``.

Sets ALL THREE env knobs BEFORE loading the producer_plus agent:
- ``PRODUCER_PLUS_MULTI_SIZE=1`` (Step 4: N=3 size variants per source/target)
- ``PRODUCER_PLUS_COALITIONS=1`` (Step 5 Fix-A: L=2 multi-source coalitions)
- ``PRODUCER_PLUS_OPP_PROJECTION=1`` (Step 3 redux: Producer-mirror opp model)

Step 3's original hypothesis was that opp-aware scoring UNLOCKS Steps 4
and 5 -- specifically, that the under-send LO/MID variants from multi-size
and the full-drain coalitions from Fix-A score correctly when the scorer
accounts for opp's counter-launches. Step 3 was tested standalone (no
multi-size, no coalitions) which only validates the projection mechanism
itself, not the unlock claim. This shim tests the unlock claim directly.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_COALITIONS", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_kitchen_sink_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
