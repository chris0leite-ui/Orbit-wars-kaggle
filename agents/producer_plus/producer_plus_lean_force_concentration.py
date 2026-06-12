"""Lean force-concentration composed variant (no opponent model).

Sets multi_size + recapture penalty + force-concentration BEFORE loading
the producer_plus agent. NO opp_projection, NO multi-tick. Tests whether
the producer-mirror opp model still pulls its weight once force-
concentration relaxes the chooser's source-scatter pathology — if this
variant lifts equally well as `multi_tick_force_concentration`, the opp
model is dead weight and this cheaper variant ships instead.
"""
import os
import sys
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
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
    "producer_plus_lean_force_concentration_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
