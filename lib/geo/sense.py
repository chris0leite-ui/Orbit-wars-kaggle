"""Geometric sense layer for the `geo` agent.

Computes once per turn:
- my_clusters / enemy_clusters: single-link agglomerative groupings by mutual reach
- voronoi: nearest-cluster assignment for neutral planets (by ETA, with tie-band)
- front_pids: my planets within FRONT_RADIUS_TURNS of any enemy planet
- threat_budget: per-my-planet sum of incoming enemy ships within THREAT_HORIZON
- comet_claims: my-cluster index for comets I can reach before any enemy
  (also enforces H15: drop comets whose remaining lifetime <= my ETA)

Distance unit: ETA in turns at REINFORCE_GARRISON ship-size. ETA uses
`lib/fleet.py:eta_turns` (log-curve fleet speed). For rotating planets
we evaluate positions AT CURRENT STEP — clustering is recomputed every
turn, so positional drift is captured naturally.

The threat budget reuses `WorldModel.ledger` (in-flight fleets only) +
a heuristic "expected enemy launch" estimator for upcoming threats:
naive — does not predict future opponent decisions. Good enough for
the posture arbiter, which only needs "is this planet at risk?" not
exact ship counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lib.fleet import eta_turns
from lib.intent import World
from lib.world_model import WorldModel, comet_remaining_lifetime


# ---------------------------------------------------------------------------
# Tunables (calibrated by inspection; tunable via local A/B at n=64)
# ---------------------------------------------------------------------------

# Single-link clustering: two of my planets are in the same cluster if
# they're within MUTUAL_REACH_TURNS turns of each other at the reference
# fleet size below. 8 turns ≈ ~48 board units at small fleets.
MUTUAL_REACH_TURNS = 8
REINFORCE_GARRISON = 20  # reference fleet size for ETA computations

# Front detection: my planet is on the front iff at least one enemy
# planet is within FRONT_RADIUS_TURNS turns.
FRONT_RADIUS_TURNS = 10

# Voronoi tie-band: neutrals within VORONOI_TIE_TOL turns of being
# equally-reachable by two clusters are marked CONTESTED (-1).
VORONOI_TIE_TOL = 2

# Threat horizon: how far ahead we sum incoming enemy ships per planet.
THREAT_HORIZON = 20

# Cluster sentinel for "contested neutral" / "no owning cluster".
CONTESTED = -1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """A group of friendly (or hostile) planets in mutual reinforcement reach."""
    idx: int                          # stable index in the parent list
    owner: int                        # player id (mine, or one of the enemies)
    planet_ids: list[int]
    total_ships: int = 0
    total_production: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)


@dataclass
class SenseState:
    """Frozen geometric snapshot for a single turn."""
    my_clusters:    list[Cluster] = field(default_factory=list)
    enemy_clusters: list[Cluster] = field(default_factory=list)
    voronoi:        dict[int, int] = field(default_factory=dict)   # neutral_pid -> cluster idx (mine)
    front_pids:     set[int] = field(default_factory=set)          # my planets adjacent to enemy
    threat_budget:  dict[int, int] = field(default_factory=dict)   # my_pid -> enemy ships incoming
    comet_claims:   dict[int, int] = field(default_factory=dict)   # comet_pid -> my cluster idx
    pid_to_cluster: dict[int, int] = field(default_factory=dict)   # my_pid -> cluster idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planet_eta(p_src, p_dst, ships: int = REINFORCE_GARRISON) -> int:
    """ETA in turns from src to dst at the reference fleet size."""
    return eta_turns((p_src.x, p_src.y), (p_dst.x, p_dst.y), ships)


def _single_link_clusters(planets, owner: int, threshold_turns: int) -> list[Cluster]:
    """Single-link agglomerative on pairwise ETA distance."""
    n = len(planets)
    if n == 0:
        return []
    # Union-find with planet-list index.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _planet_eta(planets[i], planets[j]) <= threshold_turns:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[Cluster] = []
    for idx, members in enumerate(groups.values()):
        pids = [planets[i].id for i in members]
        total_ships = sum(int(planets[i].ships) for i in members)
        total_prod = sum(float(planets[i].production) for i in members)
        cx = sum(planets[i].x for i in members) / len(members)
        cy = sum(planets[i].y for i in members) / len(members)
        out.append(Cluster(
            idx=idx,
            owner=owner,
            planet_ids=pids,
            total_ships=total_ships,
            total_production=total_prod,
            centroid=(cx, cy),
        ))
    return out


def _min_eta_to_cluster(planets_by_id, src_pid: int, cluster: Cluster) -> int:
    """Min ETA from src to any planet in cluster."""
    src = planets_by_id[src_pid]
    return min(
        _planet_eta(src, planets_by_id[pid]) for pid in cluster.planet_ids
    )


def _min_eta_from_cluster(planets_by_id, cluster: Cluster, dst_pid: int) -> int:
    """Min ETA from any cluster member to dst (used for Voronoi)."""
    dst = planets_by_id[dst_pid]
    return min(
        _planet_eta(planets_by_id[pid], dst) for pid in cluster.planet_ids
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sense_state(world: World, model: WorldModel) -> SenseState:
    """Compute the geometric snapshot for this turn.

    Pure function of (world, model). Cost: O(planets^2) for clustering
    + O(planets × clusters) for Voronoi + O(planets) for threat. On a
    30-planet board, total is well under 5 ms.
    """
    state = SenseState()
    if not world.planets_by_id:
        return state

    my_id = world.my_id
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    enemy_planets_by_owner: dict[int, list] = {}
    neutral_planets = []
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            continue
        if p.owner == -1:
            neutral_planets.append(p)
        else:
            enemy_planets_by_owner.setdefault(p.owner, []).append(p)

    state.my_clusters = _single_link_clusters(my_planets, my_id, MUTUAL_REACH_TURNS)

    # Aggregate enemy clusters across all enemy seats. We tag each cluster
    # with its owner so 4P FFA can distinguish opponents if needed.
    enemy_clusters: list[Cluster] = []
    next_idx = 0
    for owner, ps in enemy_planets_by_owner.items():
        for c in _single_link_clusters(ps, owner, MUTUAL_REACH_TURNS):
            c.idx = next_idx
            next_idx += 1
            enemy_clusters.append(c)
    state.enemy_clusters = enemy_clusters

    # pid_to_cluster: my planet id -> my cluster idx
    for c in state.my_clusters:
        for pid in c.planet_ids:
            state.pid_to_cluster[pid] = c.idx

    pbi = world.planets_by_id

    # Voronoi over neutrals: nearest-cluster by min-ETA, with tie-band.
    if state.my_clusters and neutral_planets:
        for n in neutral_planets:
            etas = [(_min_eta_from_cluster(pbi, c, n.id), c.idx) for c in state.my_clusters]
            enemy_etas = [
                _min_eta_from_cluster(pbi, c, n.id) for c in state.enemy_clusters
            ]
            etas.sort()
            best_eta, best_idx = etas[0]
            # If an enemy cluster reaches this neutral strictly faster, it's
            # not in our Voronoi at all.
            if enemy_etas and min(enemy_etas) < best_eta - VORONOI_TIE_TOL:
                continue
            if enemy_etas and min(enemy_etas) <= best_eta + VORONOI_TIE_TOL:
                state.voronoi[n.id] = CONTESTED
                continue
            # Tie-band between two of MY clusters → still mine, take the closer.
            state.voronoi[n.id] = best_idx

    # Front planets: my planets within FRONT_RADIUS_TURNS of any enemy planet.
    enemy_pids_flat = [pid for c in state.enemy_clusters for pid in c.planet_ids]
    if enemy_pids_flat:
        for mp in my_planets:
            min_e = min(
                _planet_eta(mp, pbi[epid]) for epid in enemy_pids_flat
            )
            if min_e <= FRONT_RADIUS_TURNS:
                state.front_pids.add(mp.id)

    # Threat budget: per my-planet, sum enemy ships arriving within horizon.
    for mp in my_planets:
        arrivals = model.ledger.get(mp.id) or []
        total = 0
        for eta, owner, ships in arrivals:
            if owner == my_id or ships <= 0:
                continue
            if eta <= THREAT_HORIZON:
                total += int(ships)
        if total:
            state.threat_budget[mp.id] = total

    # Comet claims: for each comet, check if WE reach faster than any enemy.
    for cid in world.comet_ids:
        comet = pbi.get(cid)
        if comet is None or comet.owner == my_id:
            continue
        rem = comet_remaining_lifetime(cid, world)
        if rem is None:
            continue
        my_etas = [
            (_planet_eta(pbi[pid], comet), c.idx)
            for c in state.my_clusters
            for pid in c.planet_ids
        ]
        if not my_etas:
            continue
        my_eta, my_cluster = min(my_etas)
        if my_eta >= rem:
            continue  # H15: comet leaves before we arrive
        enemy_etas = [
            _planet_eta(pbi[pid], comet)
            for c in state.enemy_clusters
            for pid in c.planet_ids
        ]
        if enemy_etas and min(enemy_etas) <= my_eta:
            continue  # enemy reaches first
        state.comet_claims[cid] = my_cluster

    return state
