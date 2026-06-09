"""Multi-tick + recap + force-concentration composed variant.

Sets multi_size + opp_projection + multi-tick (K_4P=3, K_2P=2) + recapture
penalty + force-concentration BEFORE loading the producer_plus agent.
This is the composed path that ships if the standalone force-concentration
A/B clears: scorer-stack mechanisms remain in place, with the chooser
relaxed to allow up to MAX_WAVES (default 2) waves per target so the
remaining ships can reinforce a high-value capture rather than scatter
to lower-value targets.
"""
import os
import sys
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "3")
os.environ.setdefault("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "2")
os.environ.setdefault("PRODUCER_PLUS_RECAPTURE_PENALTY", "1")
os.environ.setdefault("PRODUCER_PLUS_FORCE_CONCENTRATION", "1")

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
    "producer_plus_multi_tick_force_concentration_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
