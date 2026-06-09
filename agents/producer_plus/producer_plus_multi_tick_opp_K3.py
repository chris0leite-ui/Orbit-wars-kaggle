"""Multi-tick opp projection variant on top of multi_opp_def.

Sets ``PRODUCER_PLUS_MULTI_SIZE=1``, ``PRODUCER_PLUS_OPP_PROJECTION=1``,
and per-player-count multi-tick depth (K=3 in 4P, K=2 in 2P) BEFORE
loading the producer_plus agent. Horizon bump stays off so this variant
isolates the multi-tick effect.

Tests whether projecting opp at game-ticks 0..K-1 (instead of tick 0
only) breaks the cycle stalemate observed in validation game 78807326.
See knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-
scaling.md for the structural-defect diagnosis.
"""
import os
import sys
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "3")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "2")

# kaggle_environments execs agent files without defining __file__, but it
# appends the agent file's directory to sys.path during the exec — recover
# our directory from there (else this shim dies and plays None every turn).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if sys.path and os.path.isfile(os.path.join(sys.path[-1], "producer_agent.py")):
        _HERE = sys.path[-1]
    else:
        _HERE = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "producer_plus_multi_tick_opp_K3_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
