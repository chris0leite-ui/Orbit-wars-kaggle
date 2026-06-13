"""Panel entry-point for the vendored public "smarter" agent (eval-only).

A third-party public Kaggle Orbit Wars agent, vendored as a local A/B
panel opponent only (not submitted, not copied, not derived from). Like
all strong public agents it is a ProducerLite variant importing the
shared ``orbit_lite`` package, which we already vendor at
``agents/producer/orbit_lite`` — this adapter points its imports there
and loads ``main.py`` under a unique module name. See PROVENANCE.md.
"""
import os
import sys
import importlib.util

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_REPO = os.path.dirname(os.path.dirname(_HERE))
_PRODUCER_DIR = os.path.join(_REPO, "agents", "producer")
for _p in (_HERE, _PRODUCER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "panel_smarter_main", os.path.join(_HERE, "main.py")
)
_m = importlib.util.module_from_spec(_spec)
sys.modules["panel_smarter_main"] = _m
_spec.loader.exec_module(_m)


def agent(obs, configuration=None):
    return _m.agent(obs)
