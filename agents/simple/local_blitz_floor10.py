"""local_blitz_floor10 — local_blitz with GARRISON_FLOOR = 10.

Holds back 10 ships at every source on every launch. Tests whether reserving
a meaningful defender garrison saves enough planets to flip outcomes.

Calls _base.propose_intents + _base.realize directly to bypass _base.agent's
constant-reset, which would otherwise clobber the variant's settings.
"""

from __future__ import annotations

from agents.simple import local_blitz as _base


def agent(obs):
    _base.LENGTH_SCALE = 15.0
    _base.GARRISON_FLOOR = 10
    return _base.realize(
        _base.propose_intents(obs), obs, mechanisms=_base.MECHANISMS,
    )
