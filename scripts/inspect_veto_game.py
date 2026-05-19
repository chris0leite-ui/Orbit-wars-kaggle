"""Turn-by-turn analysis of tablebase_veto vs trajectory_roi for one game.

Plays one game (focal=tablebase_veto, opp=trajectory_roi). For each turn,
captures (a) what trajectory_roi would have emitted (b) what tablebase_veto
emitted (c) which clusters were detected (d) which clusters fired the veto.
Aggregates: how many turns had a veto fire, what was suppressed, did the
suppressed planet stay idle or get used for something else, what was the
downstream win/loss outcome of vetoed launches.

Run: python scripts/inspect_veto_game.py [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle_environments import make

from agents.tablebase_veto.main import agent as veto_agent
from agents.trajectory_roi.main import agent as roi_agent
from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


VETO_BUDGET_MS = 50.0
VETO_MAX_DEPTH = 14


def trace_turn(obs, configuration, turn_idx):
    """Re-run the veto pipeline manually so we can see the per-step decisions."""
    roi_emits = roi_agent(obs, configuration)
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners or not roi_emits:
        return roi_emits, roi_emits, [], []
    opp_id = max(set(other_owners), key=other_owners.count)
    clusters = find_solvable_clusters(world, my_id, opp_id)
    if not clusters:
        return roi_emits, roi_emits, [], []

    src_to_cluster_idx: dict[int, int] = {}
    for idx, c in enumerate(clusters):
        for pid in c.planet_ids:
            src_to_cluster_idx.setdefault(pid, idx)

    veto_idle_by_idx: dict[int, tuple[bool, float, int]] = {}
    survivors: list = []
    vetoes_fired: list = []
    for emit in roi_emits:
        src_id = int(emit[0])
        cidx = src_to_cluster_idx.get(src_id)
        if cidx is None:
            survivors.append(emit)
            continue
        if cidx not in veto_idle_by_idx:
            c = clusters[cidx]
            r = solve(c.isolated_obs, c.my_id, c.opp_id,
                      max_depth=VETO_MAX_DEPTH, budget_ms=VETO_BUDGET_MS)
            veto_idle_by_idx[cidx] = (len(r.best_action) == 0,
                                       r.value, r.depth_reached)
        is_idle, value, depth = veto_idle_by_idx[cidx]
        if is_idle:
            vetoes_fired.append({
                "turn": turn_idx,
                "src_id": src_id,
                "emit": list(emit),
                "cluster_planet_ids": list(clusters[cidx].planet_ids),
                "solver_value": value,
                "solver_depth": depth,
            })
        else:
            survivors.append(emit)

    cluster_summary = [
        {"planet_ids": list(c.planet_ids),
         "solver_says_idle": veto_idle_by_idx.get(i, (None, None, None))[0]}
        for i, c in enumerate(clusters)
    ]
    return roi_emits, survivors, cluster_summary, vetoes_fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make("orbit_wars", configuration={"seed": args.seed})

    # We need the obs both sides see, so we step manually.
    # Use roi as P1 (vanilla); we manually wrap the trace on P0's turns.
    trainer = env.train([None, "/home/user/Orbit-wars-kaggle/agents/trajectory_roi/main.py"])
    obs = trainer.reset()

    config = env.configuration
    turns_with_clusters = 0
    turns_with_veto = 0
    total_vetoes = 0
    veto_log = []

    turn = 0
    while True:
        roi_emits, veto_emits, clusters, vetoes = trace_turn(obs, config, turn)
        if clusters:
            turns_with_clusters += 1
        if vetoes:
            turns_with_veto += 1
            total_vetoes += len(vetoes)
            for v in vetoes:
                veto_log.append(v)
                my_ships = next(
                    (int(p[5]) for p in obs.get("planets", [])
                     if int(p[0]) == v["src_id"]), None
                )
                print(f"  turn {turn:3d}: VETO src={v['src_id']:2d} "
                      f"ships_at_src={my_ships} "
                      f"dropped_emit={v['emit']} "
                      f"cluster={v['cluster_planet_ids']} "
                      f"solver_value={v['solver_value']:.1f} "
                      f"depth={v['solver_depth']}")

        obs, reward, done, info = trainer.step(veto_emits)
        turn += 1
        if done:
            break

    print()
    print(f"=== seed={args.seed} summary ===")
    print(f"total turns: {turn}")
    print(f"turns with solvable cluster: {turns_with_clusters}")
    print(f"turns where veto fired:      {turns_with_veto}")
    print(f"total veto count:            {total_vetoes}")
    print(f"final reward (focal P0): {reward}")


if __name__ == "__main__":
    main()
