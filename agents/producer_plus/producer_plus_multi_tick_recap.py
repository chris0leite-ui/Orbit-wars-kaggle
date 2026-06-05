"""Multi-tick + recapture-penalty composed variant.

Sets multi_size + opp_projection + multi-tick (K_4P=3, K_2P=2) +
recapture penalty BEFORE loading the producer_plus agent. This is the
composed path that ships if the standalone recapture A/B clears: the
multi-tick projection informs the scorer about opp's near-term actions,
and the recapture penalty discounts thin captures opp can punish past
the projection window (K_recap_eff = max(1, K_recap - K_opp)).
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "3")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "2")
os.environ.setdefault("PRODUCER_PLUS_RECAPTURE_PENALTY", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_multi_tick_recap_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
