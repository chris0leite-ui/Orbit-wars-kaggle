"""Panel entry-point for ``producer_plus`` — our hybrid migration host.

Step 1 of the Producer-engine migration (see ``state/MIGRATION_PLAN.md``):
this file is identical in behaviour to ``agents/producer/producer_agent.py``.
``producer_plus`` owns its own ``main.py`` (a verbatim copy at this step)
so future steps can modify the planner without touching the vendored
Producer agent. The engine package ``orbit_lite/`` stays vendored at
``agents/producer/orbit_lite/`` — we reach into it via ``sys.path`` rather
than duplicate it.

Loads ``main.py`` under module name ``producer_plus_main`` to avoid the
``sys.modules`` collision with ``agents/producer/`` (which registers
``producer_main``) when both agents are loaded in the same process.
"""
import os
import sys
import importlib.util

# kaggle_environments execs agent files without defining __file__, but it
# appends the agent file's directory to sys.path for the duration of the
# exec — recover our directory from there.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if sys.path and os.path.isfile(os.path.join(sys.path[-1], "main.py")):
        _HERE = sys.path[-1]
    else:
        _HERE = os.getcwd()
_PRODUCER = os.path.join(os.path.dirname(_HERE), "producer")

# producer/ goes on sys.path first so `from orbit_lite.X import ...` inside
# main.py resolves to the vendored engine. producer_plus/ is also on the
# path so future producer_plus-local helpers (added in later migration
# steps) can be imported the same way.
for _p in (_PRODUCER, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "producer_plus_main", os.path.join(_HERE, "main.py")
)
_m = importlib.util.module_from_spec(_spec)
sys.modules["producer_plus_main"] = _m
_spec.loader.exec_module(_m)


def agent(obs, configuration=None):  # harness expects a 2-arg signature
    return _m.agent(obs)
