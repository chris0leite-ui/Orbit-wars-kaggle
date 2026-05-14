"""v7.6 — multi-launch from one source primitive (split-source).

v7_4 (40.6 %) and v7_5 (37.5 %) failed because the chooser's candidate
space — drop-one ± add-one — is structurally exhausted. v7_6 introduces
a genuinely new action primitive: the same source firing TWO launches
in one turn, one to its incumbent target and one to a runner-up.

The env explicitly supports the same `src_id` appearing multiple times
in an action: `process_moves` (in `orbit_wars.interpreter`) decrements
the source's garrison per launch, so two launches share a budget. We've
never used this pattern. The loss-mode diagnostic
(`audit/2026-05-13-v7-0-loss-modes.md`) noted top-10 has mean
garrison-at-launch ~11 vs midpack ~22 — top-10 drains sources to
multiple targets in one turn.

`enumerator_mode="drop_or_split"` is the union of drop-one variants
and per-source split variants. Uses `composite_capture_value` as the
leaf head (same as v7_4); this isolates the action-space delta.
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import composite_capture_value


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_or_split",
        K=10,
        wallclock_ms=700.0,
        value_fn=composite_capture_value,
    )
