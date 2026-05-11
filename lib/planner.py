"""settle_plan — pick the highest-score mission per source.

v0 solver (snipe-only). Per-source greedy: each source picks its own
top-score candidate independently of what other sources do. This is
intentionally a refactor over v2's strategy-level greedy — the
behavioural lever in v3.0 is *the mission framework*, not the solver:

- Mission candidates are now first-class typed objects, surfacing
  per-(source, target) score and class.
- v3.1+ adds reinforce / recapture / gang_up via new builders in
  lib/missions/; settle_plan will then arbitrate between mission classes
  rather than within a single class.

Why NOT enforce no-double-commit at v0: an earlier attempt rejected any
second mission to the same target this turn. That hurt parity with v2 on
dense boards (concentrated dual-source attacks ARE the right play when
the target's defense exceeds a single source's affordable fleet).
v3.1's gang_up will introduce coordinated multi-source arrivals through
a mission class, not a planner-level filter. Audit:
audit/2026-05-11-block-e-snipe-mvp.md (planner v0 rationale).

Pure function of (missions, world, model). World + model are accepted
for parity with future multi-class planners; the snipe-only solver
doesn't consult them today.
"""

from __future__ import annotations

from lib.intent import Intent, World
from lib.mission import Mission
from lib.world_model import WorldModel


def settle_plan(
    missions: list[Mission],
    world: World,
    model: WorldModel,
) -> list[Intent]:
    """Pick the highest-score mission per source. Returns Intents in
    source-priority order (highest top-candidate score first).
    """
    if not missions:
        return []

    # Bucket by source, take the top-score mission from each.
    by_src: dict[int, Mission] = {}
    for m in missions:
        prev = by_src.get(m.src_id)
        if prev is None or m.score > prev.score:
            by_src[m.src_id] = m

    # Stable order: highest-score source first.
    ordered = sorted(by_src.values(), key=lambda m: -m.score)
    return [m.to_intent() for m in ordered]
