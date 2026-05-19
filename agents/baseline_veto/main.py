"""Cluster-veto wrapper on top of the live Kaggle submission.

Phase B v3. Wraps `submissions/baseline.py` (sub_id 52811320, the
hold-feasibility-filter bundle pushed 2026-05-19 12:54, μ≈1165.4) with
our cluster-veto filter. When the base agent proposes a launch from a
planet inside a "solvable cluster" (2-3 planet isolated sub-game) and
the cluster minimax solver judges that sub-game best played IDLE, the
launch is suppressed.

Why this rebase (vs the trajectory_roi-based veto):
trajectory_roi is dominated by the baseline; phases B v1 and v2 both
went 0/32 vs baseline. The cluster signal is mechanically correct
(audit) but couldn't recover the gap. This wrapper tests whether the
cluster signal adds value on top of the STRONGEST base we have.

Note: the imported `submissions.baseline` is the bundled file, not our
local agents/baseline/ package — the bundle includes the
hold-feasibility filter (commit a7f9383) which our local
agents/baseline/ doesn't yet have.
"""

from __future__ import annotations

import submissions.baseline as _base

from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


VETO_BUDGET_MS = 50.0
VETO_MAX_DEPTH = 14


def agent(obs, configuration=None):
    proposed_emits = _base.agent(obs, configuration)
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

    source_to_cluster_idx: dict[int, int] = {}
    for idx, c in enumerate(clusters):
        for pid in c.planet_ids:
            source_to_cluster_idx.setdefault(pid, idx)

    veto_idle: dict[int, bool] = {}
    survivors: list = []
    for emit in proposed_emits:
        src_id = int(emit[0])
        cidx = source_to_cluster_idx.get(src_id)
        if cidx is None:
            survivors.append(emit)
            continue
        if cidx not in veto_idle:
            c = clusters[cidx]
            r = solve(c.isolated_obs, c.my_id, c.opp_id,
                      max_depth=VETO_MAX_DEPTH,
                      budget_ms=VETO_BUDGET_MS)
            veto_idle[cidx] = (len(r.best_action) == 0)
        if veto_idle[cidx]:
            continue
        survivors.append(emit)
    return survivors
