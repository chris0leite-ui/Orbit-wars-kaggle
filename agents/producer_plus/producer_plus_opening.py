"""Standalone opening-bonus variant.

Sets ``PRODUCER_PLUS_OPENING_BONUS=1`` BEFORE loading the producer_plus
agent. Opp-agnostic: rewards captures during the early-game phase,
linearly decaying to zero at ``opening_window`` (default 30).
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPENING_BONUS", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_opening_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
