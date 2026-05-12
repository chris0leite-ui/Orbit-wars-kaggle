"""v7.3 — archetype preset sibling candidates.

Generate four full-action bundles, each derived under one preset
archetype from `lib/v7_search::ARCHETYPE_PRESETS`:
- baseline   (v3.5.1: aggressive_fraction=0.7, top-1 per source)
- concentrated (Isaiah/bowwowforeach style: 0.95, top-1)
- saturation (flg/Ebi style: 0.5, top-3 per source)
- defensive  (reinforce-mission priority × 3, otherwise baseline)

Strongest hypothesis from top-10 fingerprint analysis: different board
states reward different archetypes. v3.5.1 is one fixed point in that
space; rollout-driven selection lets us pick the right archetype per
turn.
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="archetype",
        K=10,
        wallclock_ms=700.0,
    )
