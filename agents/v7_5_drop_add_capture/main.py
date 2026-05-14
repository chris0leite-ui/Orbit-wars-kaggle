"""v7.5 — drop-or-add-one chooser with `composite_capture_value`.

v7_4 lifted from 31 % → 40.6 % vs v7_0 by swapping the value head from
plain ship-delta to `composite_capture_value` (rewards real captures,
penalises wasted ships). It still loses because the drop-one action
space is monotonically narrower than the incumbent — the chooser can
suppress launches the proposer made but can't add launches the proposer
skipped.

v7_5 widens the action space: the new `drop_or_add_one` enumerator
emits incumbent + (drop each launch) + (add one top mission from each
idle owned source). Combined with the v7_4 value head, the chooser
can now both refine (drop wasteful launches) AND extend (cover idle
sources the proposer left out).

Same K=10 rollout and v3.5.1 opp model as v7_0; structurally only the
candidate enumerator + value head differ.
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import composite_capture_value


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_or_add_one",
        K=10,
        wallclock_ms=700.0,
        value_fn=composite_capture_value,
    )
