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

# `__file__` is undefined when kaggle_environments execs this file as raw
# source (its agent loader uses `exec(code, {})`). Without this guard the shim
# raised NameError on load -> the agent failed to load and idled every turn
# (silently losing every game in env.run / scripts/clean_ab). The loader does
# append this file's own directory to sys.path before exec, so recover _HERE
# from the sys.path entry that contains main.py + orbit_lite.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = None
    for _p in reversed(sys.path):
        if _p and os.path.isfile(os.path.join(_p, "main.py")) \
                and os.path.isdir(os.path.join(_p, "orbit_lite")):
            _HERE = _p
            break
    if _HERE is None:
        _HERE = os.getcwd()
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
