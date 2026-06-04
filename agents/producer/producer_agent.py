"""Panel entry-point for the vendored third-party "Producer" agent.

This is the 2-arg adapter the harness loads (``fast.py`` short-name
``producer`` -> this file). It loads the agent's own ``main.py`` under a
*unique* module name so it cannot collide with the many other
``agents/<name>/main.py`` modules the panel loads in the same process —
the original upstream shim used ``from main import agent``, which
registers a generic ``main`` in ``sys.modules`` and would return the
wrong module depending on load order.

Provenance: see PROVENANCE.md in this directory.
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # so main.py can `import orbit_lite`

_spec = importlib.util.spec_from_file_location(
    "producer_main", os.path.join(_HERE, "main.py")
)
_m = importlib.util.module_from_spec(_spec)
sys.modules["producer_main"] = _m
_spec.loader.exec_module(_m)


def agent(obs, configuration=None):  # harness expects a 2-arg signature
    return _m.agent(obs)
