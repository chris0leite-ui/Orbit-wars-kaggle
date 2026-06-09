"""Standalone force-concentration variant.

Sets ``PRODUCER_PLUS_FORCE_CONCENTRATION=1`` BEFORE loading the
producer_plus agent. No other knobs — tests the chooser-architecture
mechanism in isolation against vanilla producer.

Force-concentration relaxes the one-wave-per-target mutex in
``_greedy_select`` so up to ``MAX_WAVES`` (default 2) waves can land on
the same target per turn. Between waves the candidates are re-scored
against the committed waves so wave 2 sees wave 1's reinforcement and
does not double-count the capture.
"""
import os
import sys
import importlib.util

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
    "producer_plus_force_concentration_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
