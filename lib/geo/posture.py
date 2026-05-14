"""Posture arbiter — one of {OPENING, EXPAND, DEFEND, BREAK} per turn.

Single function `decide_posture(world, sense, model) -> Posture`.
Pure of (world, sense, model). No state.

Decision table (priority order; first match wins):

| Posture | Condition                                                         |
| ------- | ----------------------------------------------------------------- |
| OPENING | world.step <= 5                                                   |
| DEFEND  | max(threat_budget) >= 0.6 × garrison(threatened) AND on front     |
| BREAK   | We have ≥1.5× force concentration vs nearest enemy cluster        |
| EXPAND  | Default fallthrough                                               |

Posture is a discrete decision feeding mission-class multipliers in
`agents/geo/main.py`. The DEFEND/BREAK thresholds are heuristic; they
will be tuned by local A/B if the agent clears the gate.
"""

from __future__ import annotations

from enum import Enum

from lib.intent import World
from lib.world_model import WorldModel

from lib.geo.sense import SenseState


# ---------------------------------------------------------------------------
# Tunables (n=64 local A/B will refine these)
# ---------------------------------------------------------------------------

OPENING_STEP_LIMIT = 5
DEFEND_THREAT_RATIO = 0.6      # threat / garrison threshold for DEFEND
BREAK_FORCE_RATIO = 1.5        # our_cluster_strength / nearest_enemy for BREAK


class Posture(Enum):
    OPENING = "OPENING"
    EXPAND = "EXPAND"
    DEFEND = "DEFEND"
    BREAK = "BREAK"


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide_posture(world: World, sense: SenseState, model: WorldModel) -> Posture:
    """Pick a single posture for this turn."""
    if int(world.step) <= OPENING_STEP_LIMIT:
        return Posture.OPENING

    # DEFEND: a front planet under threat we can't shrug off.
    if sense.threat_budget and sense.front_pids:
        for pid, incoming in sense.threat_budget.items():
            if pid not in sense.front_pids:
                continue
            planet = world.planets_by_id.get(pid)
            if planet is None:
                continue
            garrison = max(1, int(planet.ships))
            if incoming >= DEFEND_THREAT_RATIO * garrison:
                return Posture.DEFEND

    # BREAK: we have force concentration vs nearest enemy cluster.
    if sense.my_clusters and sense.enemy_clusters:
        my_max = max(c.total_ships for c in sense.my_clusters)
        enemy_total = sum(c.total_ships for c in sense.enemy_clusters)
        # Average per-enemy-cluster strength so 4P doesn't dilute the signal.
        enemy_avg = enemy_total / max(1, len(sense.enemy_clusters))
        if my_max >= BREAK_FORCE_RATIO * enemy_avg and enemy_avg > 0:
            return Posture.BREAK

    return Posture.EXPAND
