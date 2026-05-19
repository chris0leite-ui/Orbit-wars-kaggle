"""Tablebase-veto agent: trajectory_roi proposals filtered by cluster solver.

Phase B. Runs trajectory_roi's full pipeline to get proposed launches, then
for each launch originating from a planet inside a "solvable cluster"
(2-3 planets isolated from the rest of the board), runs the cluster
minimax solver on that cluster in isolation. If the solver chooses IDLE
for the cluster, every proposed launch from that cluster's planets is
suppressed. Non-clustered emits pass through unchanged.

Rationale: the Phase A.5 audit
(audit/2026-05-19-tablebase-audit.md) found heuristic vs solver
disagreements were 10/29 one-sided over-launches (zero under-launches,
zero target mismatches). The minimum-risk filter is to drop launches the
solver judges net-negative inside their isolated sub-game.
"""

from __future__ import annotations

from agents.trajectory_roi.main import agent as roi_agent
from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


VETO_BUDGET_MS = 50.0
VETO_MAX_DEPTH = 14   # iterative deepening; budget caps actual depth.
                      # Audit used depth=8; this lets borderline-ETA clusters
                      # converge deeper when time allows.


def agent(obs, configuration=None):
    proposed_emits = roi_agent(obs, configuration)
    if not proposed_emits:
        return proposed_emits

    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        return proposed_emits
    opp_id = max(set(other_owners), key=other_owners.count)
    if my_id == opp_id:
        return proposed_emits

    clusters = find_solvable_clusters(world, my_id, opp_id)
    if not clusters:
        return proposed_emits

    source_to_cluster: dict[int, int] = {}
    for idx, c in enumerate(clusters):
        for pid in c.planet_ids:
            source_to_cluster.setdefault(pid, idx)

    veto_idle: dict[int, bool] = {}
    survivors: list = []
    for emit in proposed_emits:
        src_id = int(emit[0])
        cluster_idx = source_to_cluster.get(src_id)
        if cluster_idx is None:
            survivors.append(emit)
            continue
        if cluster_idx not in veto_idle:
            c = clusters[cluster_idx]
            result = solve(c.isolated_obs, c.my_id, c.opp_id,
                           max_depth=VETO_MAX_DEPTH,
                           budget_ms=VETO_BUDGET_MS)
            veto_idle[cluster_idx] = (len(result.best_action) == 0)
        if veto_idle[cluster_idx]:
            continue
        survivors.append(emit)

    return survivors
