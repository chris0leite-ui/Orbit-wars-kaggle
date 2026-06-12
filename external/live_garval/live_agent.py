"""Local rebuild of live sub 53588922 (producer_plus_vetorf4p_sync_garval).

Env config copied verbatim from scripts/bundle_producer_plus.py
ENV_VARIANTS["vetorf4p_sync_garval"] on claude/awesome-clarke-ixy57v
(commit 3f66440). Eval-only opponent — the Rule 45 live-pair gate.
"""
import os
import sys

for k, v in {
    "PRODUCER_PLUS_MULTI_SIZE": "1",
    "PRODUCER_PLUS_OPP_PROJECTION": "1",
    "PRODUCER_PLUS_RESPONSE_VETO": "1",
    "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
    "PRODUCER_PLUS_REPLY_SEQ": "1",
    "PRODUCER_PLUS_FFA_SCORE": "1",
    "PRODUCER_PLUS_FFA_WEIGHTS": "strength",
    "PRODUCER_PLUS_SYNC": "1",
    "PRODUCER_PLUS_SOURCE_SAFETY": "0.5",
    "PRODUCER_PLUS_GARRISON_VALUE": "12",
}.items():
    os.environ.setdefault(k, v)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "live_garval_main", os.path.join(_HERE, "main.py"))
_main = importlib.util.module_from_spec(_spec)
sys.modules["live_garval_main"] = _main
_spec.loader.exec_module(_main)

agent = _main.agent
