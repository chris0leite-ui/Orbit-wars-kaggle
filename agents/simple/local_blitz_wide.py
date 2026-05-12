"""local_blitz_wide — same as local_blitz but with LENGTH_SCALE = 25 (gentler falloff).

See agents/simple/local_blitz.py for the strategy description.
"""

from __future__ import annotations

from agents.simple import local_blitz as _base

LENGTH_SCALE = 25.0


def agent(obs):
    _base.LENGTH_SCALE = LENGTH_SCALE
    return _base.agent(obs)
