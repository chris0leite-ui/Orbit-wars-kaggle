"""Tablebase-hybrid agent: solver-driven on clustered sources, heuristic elsewhere.

Phase B v2. For each detected solvable cluster (2-3 planets isolated
from the rest of the board), runs the cluster minimax solver and emits
the solver's chosen action verbatim — either an explicit launch or no
emit at all. Non-clustered sources fall through to trajectory_roi.

Difference vs tablebase_veto (Phase B v1):
- Veto only suppresses bad heuristic launches from clustered sources.
- Hybrid replaces them with the solver's actual choice, including
  cases where the solver prefers different ship sizing than the
  heuristic on an agreed target.

Audit (audit/2026-05-19-tablebase-audit.md) showed the two differ on
at most 3/29 cases (AGREE-LAUNCH sizing). The real test of the hybrid
is on runtime clusters not in the audit set.
"""

from __future__ import annotations

from agents.trajectory_roi.main import agent as roi_agent
from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


HYBRID_BUDGET_MS = 50.0
HYBRID_MAX_DEPTH = 14


def agent(obs, configuration=None):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        return roi_agent(obs, configuration)
    opp_id = max(set(other_owners), key=other_owners.count)
    if my_id == opp_id:
        return roi_agent(obs, configuration)

    clusters = find_solvable_clusters(world, my_id, opp_id)
    if not clusters:
        return roi_agent(obs, configuration)

    my_planet_ids = {p.id for p in world.planets if p.owner == my_id}

    # For each MY-owned planet that appears in any cluster, find the
    # cluster with the highest solver value and use its action for
    # that source. Avoids duplicate emits when the detector returns
    # overlapping clusters (common when one planet is hub-positioned
    # near many neighbors — e.g., 19 clusters all containing the
    # same source at one observed turn 5).
    best_by_src: dict[int, tuple[float, list]] = {}  # src -> (value, emits)
    clustered_my_sources: set[int] = set()
    for c in clusters:
        my_in_cluster = [pid for pid in c.planet_ids if pid in my_planet_ids]
        if not my_in_cluster:
            continue
        result = solve(c.isolated_obs, c.my_id, c.opp_id,
                       max_depth=HYBRID_MAX_DEPTH,
                       budget_ms=HYBRID_BUDGET_MS)
        for pid in my_in_cluster:
            clustered_my_sources.add(pid)
            emits_from_src = [list(e) for e in result.best_action
                              if int(e[0]) == pid]
            prev = best_by_src.get(pid)
            if prev is None or result.value > prev[0]:
                best_by_src[pid] = (result.value, emits_from_src)

    solver_emits: list = []
    for pid, (_, emits_from_src) in best_by_src.items():
        solver_emits.extend(emits_from_src)

    roi_emits = roi_agent(obs, configuration)
    residual_emits = [e for e in roi_emits
                      if int(e[0]) not in clustered_my_sources]

    return solver_emits + residual_emits
