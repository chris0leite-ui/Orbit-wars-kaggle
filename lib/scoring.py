"""Shared scoring primitives for ROI-family strategies.

These helpers project fleet arrival deterministically against the current
obs snapshot. Variants under `agents/simple/roi_*.py` build their score
function on top of these primitives — the goal is that any strategy that
wants "what does this target look like at arrival" gets one consistent
answer rather than re-implementing the projection per file.

Caveat: the projection is single-pass and uses the current `target.ships`
to seed fleet speed. Strategies that send larger fleets (post-
`arrival_size` inflation) actually arrive a hair sooner, so `eta_proxy`
is a slight over-estimate. That's deliberately conservative: scoring
penalises far targets a touch more than the truth, biasing selection
toward closer / safer captures.

`T_TOTAL` defaults to the Orbit Wars episode-step limit (500 per the
data/README spec). If a future game has a different limit, pass it
explicitly.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.fleet import speed as fleet_speed
from lib.geometry import dist

T_TOTAL_DEFAULT: int = 500


def eta_proxy(mine: Planet, target: Planet) -> int:
    """Conservative integer-turn ETA for a launch from `mine` to `target`.

    Uses `fleet_speed(target.ships + 1)` as the speed proxy — this is the
    speed of the minimum-cover fleet a strategy would send before
    `arrival_size` inflates it. Returns 0 for zero-distance pairs.
    """
    d = dist((mine.x, mine.y), (target.x, target.y))
    if d == 0.0:
        return 0
    v = fleet_speed(target.ships + 1)
    return int(math.ceil(d / v))


def projected_garrison(target: Planet, eta: int) -> int:
    """Predicted target garrison at arrival.

    Neutral targets (`owner == -1`) don't produce; their garrison stays
    flat. Owned targets (ours or enemy) grow by `production * eta`.
    """
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta


def s_needed(target: Planet, eta: int) -> int:
    """Minimum fleet size to capture `target` at arrival.

    Mirrors `lib.mechanism.arrival_size`'s formula so strategy-side gates
    use the same number the mechanism layer will end up sizing fleets
    against. The +1 ensures strict win on the combat resolver.
    """
    return projected_garrison(target, eta) + 1


def horizon(step: int, eta: int, t_total: int = T_TOTAL_DEFAULT) -> int:
    """Remaining turns the captured planet can produce for us.

    `H = max(0, T_total - step - eta)`. Late-game this collapses toward 0
    so production-weighted scores naturally pivot to cheap snipes /
    denial in the closing turns.
    """
    return max(0, t_total - step - eta)


def margin_multiplier(target: Planet, my_id: int) -> int:
    """Owner-flip multiplier for margin-based scoring.

    Captures contribute to the margin (my_ships - their_ships):
    - Self (already ours): 0 — reinforce moves don't change margin.
    - Neutral: 1 — we gain P/turn going forward.
    - Enemy: 2 — we gain P/turn AND deny them P/turn (zero-sum).
    """
    if target.owner == my_id:
        return 0
    if target.owner == -1:
        return 1
    return 2


__all__ = [
    "T_TOTAL_DEFAULT",
    "eta_proxy",
    "projected_garrison",
    "s_needed",
    "horizon",
    "margin_multiplier",
]
