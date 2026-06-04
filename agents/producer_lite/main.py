"""Standalone agent wrapper for `lib.producer_lite.producer_lite_policy`.

Lets the pure-python Producer port play as a full agent so it can be A/B'd
via scripts/clean_ab.py (fidelity gates: vs lite_greedy, vs baseline, vs the
full Producer). Not a submission — a local evaluation / opponent-model agent.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib.producer_lite import producer_lite_policy


def agent(obs, configuration=None):
    return producer_lite_policy(obs)
