"""local_blitz_steep — same as local_blitz but with LENGTH_SCALE = 10 (steeper falloff).

See agents/simple/local_blitz.py for the strategy description.
"""

from __future__ import annotations

from agents.simple import local_blitz as _base

LENGTH_SCALE = 10.0


def agent(obs):
    # Mutate inside agent() so worker-reuse + module-import-once doesn't
    # leak parameters across variants (friction 2026-05-12: module-
    # mutation-patching-has-worker-reuse-race).
    _base.LENGTH_SCALE = LENGTH_SCALE
    return _base.agent(obs)
