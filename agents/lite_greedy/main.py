"""Standalone agent wrapper for `lib.opp_model.lite_greedy_policy`.

The weak (Tier-0) opponent model our chooser currently rolls out against,
exposed as a full agent so producer_lite can be A/B'd against it (the cheap
primary fidelity gate: a faithful Producer port must beat our weak model).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib.opp_model import lite_greedy_policy


def agent(obs, configuration=None):
    return lite_greedy_policy(obs)
