"""Oracle agent — entry point.

Exact-ledger adversarial planner with a learned value function trained on
top-ladder replays. See planner.py for the per-turn pipeline.

Loader notes: kaggle_environments' agent loader exec()s this file WITHOUT
__file__ when given a path string (importlib loaders define it). A wrong
directory guess would make every import fail and — if swallowed — produce
a silent always-[] agent, the exact dead-agent failure mode documented in
audit/2026-06-12-dead-opponent-ab-correction.md. So: locate the package
defensively and raise LOUDLY if the planner cannot be imported.
"""

import os
import sys
import traceback

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = None
if _HERE is None or not os.path.isfile(os.path.join(_HERE, "planner.py")):
    for _cand in ("agents/oracle",
                  os.path.join(os.getcwd(), "agents", "oracle"),
                  "/home/user/Orbit-wars-kaggle/agents/oracle"):
        if os.path.isfile(os.path.join(_cand, "planner.py")):
            _HERE = os.path.abspath(_cand)
            break

_IMPORT_ERROR = None
Planner = None
if _HERE is not None:
    _PARENT = os.path.dirname(_HERE)
    for _p in (_PARENT, _HERE):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from oracle.planner import Planner       # package import (local run)
    except Exception:
        try:
            from planner import Planner          # flat layout (bundle root)
        except Exception as e:
            _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = FileNotFoundError("oracle package directory not found")

_PLANNER = Planner() if Planner is not None else None


def agent(obs, configuration=None):
    if _PLANNER is None:
        # a dead agent must die loudly, never quietly play [] forever
        raise RuntimeError(f"oracle import failed: {_IMPORT_ERROR}")
    try:
        return _PLANNER.act(obs)
    except Exception:
        if os.environ.get("ORACLE_RAISE"):
            raise
        traceback.print_exc(file=sys.stderr)
        return []
