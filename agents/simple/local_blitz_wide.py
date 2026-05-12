"""local_blitz_wide — LENGTH_SCALE = 25 (gentler distance penalty).

Calls _base.propose_intents + _base.realize directly to bypass _base.agent's
constant-reset, which would otherwise clobber the variant's settings.
"""

from __future__ import annotations

from agents.simple import local_blitz as _base


def agent(obs):
    _base.LENGTH_SCALE = 25.0
    _base.GARRISON_FLOOR = 0
    return _base.realize(
        _base.propose_intents(obs), obs, mechanisms=_base.MECHANISMS,
    )
