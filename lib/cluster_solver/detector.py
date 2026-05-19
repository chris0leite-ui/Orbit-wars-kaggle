"""Detect 2-3 planet clusters that are isolated enough to solve in isolation.

A *solvable cluster* is a set of 2 or 3 planets where:

  - All pairs are within `INTRA_REACH` board units of each other
    (≈ 6 turns at a 30-ship fleet — close enough that decisions within
    the cluster have meaningful cross-effects).
  - Every cluster planet is ≥ `EXTRA_REACH` units away from any
    non-cluster non-neutral entity (planet OR in-flight fleet —
    captures or interference from outside can't reach within the
    search horizon).
  - At least one planet is owned by my_id (we have decisions to make).
  - At least one planet is NOT owned by my_id (there's something to
    capture or defend against).

The returned `ClusterDescriptor.isolated_obs` synthesises an obs dict
containing ONLY the cluster's planets plus a distant idle phantom enemy
(workaround for the env's `alive_players <= 1` terminal check). That obs
is what `lib.cluster_solver.minimax.solve()` consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from lib.trajectory_layer import World


# ---- tunables -----------------------------------------------------------

INTRA_REACH = 30.0       # max pairwise distance inside a cluster
EXTRA_REACH = 60.0       # min distance from cluster to any outside non-neutral
SEARCH_HORIZON = 12      # turns the cluster is safe from outside interference
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 3

# Phantom enemy placement (off in a corner, well away from any cluster).
PHANTOM_PLANET_ID_OFFSET = 10_000
PHANTOM_X, PHANTOM_Y = 5.0, 95.0
PHANTOM_SHIPS = 9        # below lite_greedy's 10-ship launch threshold
PHANTOM_PRODUCTION = 0   # stays at 9 forever → never launches; offset is constant
PHANTOM_RADIUS = 1.0


# ---- types --------------------------------------------------------------


@dataclass(frozen=True)
class ClusterDescriptor:
    planet_ids: tuple[int, ...]
    isolated_obs: dict
    search_horizon: int
    my_id: int
    opp_id: int
    source_step: int
    # Phantom planet id used in the isolated_obs (always represents opp_id).
    phantom_planet_id: int = -1


# ---- helpers ------------------------------------------------------------


def _planet_distance(p_a, p_b) -> float:
    return math.hypot(p_a.current_x - p_b.current_x,
                      p_a.current_y - p_b.current_y)


def _fleet_distance_to_planet(f, p) -> float:
    return math.hypot(f.current_x - p.current_x,
                      f.current_y - p.current_y)


def _cluster_external_clear(cluster, world, my_id) -> bool:
    """Return True iff no non-cluster non-neutral entity is within
    EXTRA_REACH of any cluster planet."""
    cluster_ids = {p.id for p in cluster}
    for p in world.planets:
        if p.id in cluster_ids or p.owner == -1:
            continue  # planets that aren't in the cluster, and not neutral
        for cp in cluster:
            if _planet_distance(cp, p) < EXTRA_REACH:
                return False
    for f in world.fleets:
        for cp in cluster:
            # Fleet sources can be cluster planets; that's fine for in-cluster
            # fleets. Only outside fleets matter.
            if f.from_planet_id in cluster_ids:
                continue
            if _fleet_distance_to_planet(f, cp) < EXTRA_REACH:
                return False
    return True


def _cluster_has_both_sides(cluster, my_id: int) -> bool:
    has_mine = any(p.owner == my_id for p in cluster)
    has_other = any(p.owner != my_id for p in cluster)  # neutral or opp counts
    return has_mine and has_other


def _build_isolated_obs(cluster, world, my_id: int, opp_id: int) -> dict:
    """Synthesize an obs containing only the cluster's planets + the
    cluster's in-flight fleets + a phantom idle enemy."""
    cluster_ids = {p.id for p in cluster}
    raw_planets: list[list] = []
    for p in cluster:
        raw_planets.append([
            int(p.id), int(p.owner),
            float(p.current_x), float(p.current_y),
            float(p.radius), int(p.ships), int(p.production),
        ])

    # Phantom planet ID = max id in cluster + offset, owned by opp_id.
    # Ensures the env's alive_players check stays > 1.
    has_opp_in_cluster = any(p.owner == opp_id for p in cluster)
    phantom_id = -1
    if not has_opp_in_cluster:
        phantom_id = max(p.id for p in cluster) + PHANTOM_PLANET_ID_OFFSET
        raw_planets.append([
            phantom_id, int(opp_id),
            PHANTOM_X, PHANTOM_Y,
            PHANTOM_RADIUS, PHANTOM_SHIPS, PHANTOM_PRODUCTION,
        ])

    raw_fleets: list[list] = []
    for f in world.fleets:
        if f.from_planet_id in cluster_ids and f.ships > 0:
            # Env fleet format: [id, owner, x, y, angle, from_id, ships].
            # NOTE: FleetView's field order has ships BEFORE from_planet_id,
            # but the env's raw fleet list is the opposite — see
            # `lib/game/interpreter.py:651-660`. Get it wrong and the
            # interpreter reads `ships=fleet[6]` as a planet id (0/1/…),
            # then `_log(ships)` blows up on a zero.
            raw_fleets.append([
                int(f.id), int(f.owner),
                float(f.current_x), float(f.current_y),
                float(f.angle),
                int(f.from_planet_id), int(f.ships),
            ])

    obs = {
        "player": int(my_id),
        "step": int(world.step),
        "planets": raw_planets,
        "fleets": raw_fleets,
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": float(world.omega),
        "initial_planets": [list(p) for p in raw_planets],
    }
    return obs, phantom_id


# ---- main entry ---------------------------------------------------------


def find_solvable_clusters(world: World, my_id: int | None = None,
                            opp_id: int | None = None,
                            ) -> list[ClusterDescriptor]:
    """Walk the world and return every 2-3 planet cluster that meets the
    isolation criteria.

    `my_id` defaults to `world.my_id`; `opp_id` defaults to the most
    common non-mine non-neutral owner (or 1 - my_id for a 2P game).
    """
    if my_id is None:
        my_id = world.my_id
    if opp_id is None:
        other_owners = [p.owner for p in world.planets
                        if p.owner not in (-1, my_id)]
        opp_id = max(set(other_owners), key=other_owners.count) if other_owners else (1 - my_id)

    planets = list(world.planets)
    out: list[ClusterDescriptor] = []

    for size in range(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE + 1):
        for combo in combinations(planets, size):
            # Quick reject by intra-cluster distance first (cheap).
            ok_pairs = True
            for a, b in combinations(combo, 2):
                if _planet_distance(a, b) > INTRA_REACH:
                    ok_pairs = False
                    break
            if not ok_pairs:
                continue
            if not _cluster_has_both_sides(combo, my_id):
                continue
            if not _cluster_external_clear(combo, world, my_id):
                continue

            obs, phantom_id = _build_isolated_obs(combo, world, my_id, opp_id)
            out.append(ClusterDescriptor(
                planet_ids=tuple(sorted(p.id for p in combo)),
                isolated_obs=obs,
                search_horizon=SEARCH_HORIZON,
                my_id=my_id,
                opp_id=opp_id,
                source_step=int(world.step),
                phantom_planet_id=phantom_id,
            ))
    return out
