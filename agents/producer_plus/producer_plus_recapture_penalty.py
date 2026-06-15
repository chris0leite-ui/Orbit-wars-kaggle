"""Standalone recapture-penalty variant.

Sets ``PRODUCER_PLUS_RECAPTURE_PENALTY=1`` BEFORE loading the
producer_plus agent. No other knobs — tests the recapture mechanism in
isolation against vanilla producer.

See agents/producer/orbit_lite/recapture.py for the math; see
knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-scaling.md
for the structural-defect diagnosis the mechanism targets.
"""
import os
import sys
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_RECAPTURE_PENALTY", "1")

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
    "producer_plus_recapture_penalty_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
