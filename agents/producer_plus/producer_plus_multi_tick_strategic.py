"""Full composed variant: multi-tick + recap + denial + opening.

Sets multi_size + opp_projection + multi-tick (K_4P=3, K_2P=2) +
recapture penalty + denial bonus + opening bonus BEFORE loading the
producer_plus agent.

The path that ships if the strategic A/B (denial+opening on top of
multi_tick_recap) confirms lift. Layered semantics:
- multi_tick: scorer sees opp's launches at game-ticks 0..K-1.
- recap: subtracts a discount for captures opp can recapture.
- denial: adds a bonus for captures that block opp's plans.
- opening: adds a bonus during early-game expansion phase.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "3")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "2")
os.environ.setdefault("PRODUCER_PLUS_RECAPTURE_PENALTY", "1")
os.environ.setdefault("PRODUCER_PLUS_DENIAL_BONUS", "1")
os.environ.setdefault("PRODUCER_PLUS_OPENING_BONUS", "1")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_multi_tick_strategic_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
