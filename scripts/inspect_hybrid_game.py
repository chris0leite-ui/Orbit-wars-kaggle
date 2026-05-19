"""Turn-by-turn trace of tablebase_hybrid for one game.

For each turn captures: (a) clusters detected, (b) solver verdict per
cluster, (c) what trajectory_roi would have emitted, (d) what hybrid
emitted, (e) the per-source delta (which heuristic emits got replaced
or dropped). Print a per-turn line whenever the hybrid differs from
trajectory_roi.

Run: python scripts/inspect_hybrid_game.py [--seed N] [--vs path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle_environments import make

from agents.trajectory_roi.main import agent as roi_agent
from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


HYBRID_BUDGET_MS = 50.0
HYBRID_MAX_DEPTH = 14


def trace_turn(obs, configuration, turn_idx):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        roi = roi_agent(obs, configuration)
        return roi, roi, [], [], []
    opp_id = max(set(other_owners), key=other_owners.count)

    roi_emits = roi_agent(obs, configuration)
    clusters = find_solvable_clusters(world, my_id, opp_id)

    if not clusters:
        return roi_emits, roi_emits, [], [], []

    my_planet_ids = {p.id for p in world.planets if p.owner == my_id}

    best_by_src: dict[int, tuple[float, list]] = {}
    clustered_my_sources: set[int] = set()
    cluster_records = []
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
        cluster_records.append({
            "planet_ids": list(c.planet_ids),
            "solver_action": list(result.best_action),
            "solver_value": result.value,
            "solver_depth": result.depth_reached,
            "elapsed_ms": result.elapsed_ms,
        })

    solver_emits: list = []
    for pid, (_, emits_from_src) in best_by_src.items():
        solver_emits.extend(emits_from_src)

    residual_emits = [e for e in roi_emits
                      if int(e[0]) not in clustered_my_sources]
    hybrid_emits = solver_emits + residual_emits

    # Per-source delta vs trajectory_roi.
    roi_by_src: dict[int, list] = {}
    for e in roi_emits:
        roi_by_src.setdefault(int(e[0]), []).append(list(e))
    hybrid_by_src: dict[int, list] = {}
    for e in hybrid_emits:
        hybrid_by_src.setdefault(int(e[0]), []).append(list(e))

    src_deltas = []
    for src in clustered_my_sources:
        roi_e = roi_by_src.get(src, [])
        hyb_e = hybrid_by_src.get(src, [])
        if roi_e != hyb_e:
            src_deltas.append({
                "src": src,
                "roi": roi_e,
                "hybrid": hyb_e,
            })
    return roi_emits, hybrid_emits, cluster_records, src_deltas, list(clustered_my_sources)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vs", type=str,
                    default="/home/user/Orbit-wars-kaggle/submissions/baseline.py",
                    help="path to opponent agent .py")
    args = ap.parse_args()

    env = make("orbit_wars", configuration={"seed": args.seed})
    trainer = env.train([None, args.vs])
    obs = trainer.reset()
    config = env.configuration

    turn = 0
    turns_with_cluster = 0
    turns_hybrid_differs = 0
    n_idle_replaces = 0
    n_launch_replaces = 0
    n_passthrough = 0
    examples = []

    while True:
        roi, hyb, clusters, deltas, clustered_srcs = trace_turn(
            obs, config, turn)
        if clusters:
            turns_with_cluster += 1
        if deltas:
            turns_hybrid_differs += 1
            for d in deltas:
                roi_n = len(d["roi"])
                hyb_n = len(d["hybrid"])
                if hyb_n == 0 and roi_n > 0:
                    n_idle_replaces += 1
                elif hyb_n > 0 and roi_n == 0:
                    n_launch_replaces += 1
                else:
                    n_launch_replaces += 1
            if len(examples) < 50:
                ships_by_src = {int(p[0]): int(p[5])
                                for p in obs.get("planets", [])}
                cluster_summary = [
                    f"{c['planet_ids']}->{c['solver_action']} "
                    f"(v={c['solver_value']:.0f}, d={c['solver_depth']}, "
                    f"{c['elapsed_ms']:.0f}ms)"
                    for c in clusters
                ]
                examples.append({
                    "turn": turn,
                    "clusters": cluster_summary,
                    "deltas": deltas,
                    "ships": {s: ships_by_src.get(s) for s in clustered_srcs},
                })

        obs, reward, done, info = trainer.step(hyb)
        turn += 1
        if done:
            break

    print(f"=== seed={args.seed} vs {Path(args.vs).name} ===")
    print(f"total turns:                       {turn}")
    print(f"turns with solvable cluster:       {turns_with_cluster}")
    print(f"turns hybrid differs from roi:     {turns_hybrid_differs}")
    print(f"  idle replaces (roi launched, hybrid IDLE): {n_idle_replaces}")
    print(f"  launch replaces (different launch):        {n_launch_replaces}")
    print(f"final reward (focal P0):           {reward}")
    print()
    print("--- first few divergence turns ---")
    for ex in examples:
        print(f"turn {ex['turn']:3d}:")
        for cs in ex["clusters"]:
            print(f"   cluster {cs}")
        for d in ex["deltas"]:
            src = d["src"]
            ships = ex["ships"].get(src)
            print(f"   src={src:2d} ships={ships}: "
                  f"roi={d['roi']}  ->  hybrid={d['hybrid']}")


if __name__ == "__main__":
    main()
