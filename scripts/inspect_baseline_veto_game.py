"""Quick veto-firing count for baseline_veto over one game.

Run: python scripts/inspect_baseline_veto_game.py [--seed N] [--vs path]

Only counts: turns where clusters were detected, and turns where the
veto suppressed at least one base-agent launch. If both are 0,
baseline_veto is identical to baseline and the A/B will be noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle_environments import make

import submissions.baseline as _base
from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World


VETO_BUDGET_MS = 50.0
VETO_MAX_DEPTH = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vs", type=str,
                    default="/home/user/Orbit-wars-kaggle/submissions/baseline.py")
    args = ap.parse_args()

    env = make("orbit_wars", configuration={"seed": args.seed})
    trainer = env.train([None, args.vs])
    obs = trainer.reset()
    config = env.configuration

    turn = 0
    turns_with_cluster = 0
    turns_veto_fired = 0
    total_vetoes = 0
    base_total_emits = 0
    veto_drops = []

    while True:
        base_emits = _base.agent(obs, config)
        base_total_emits += len(base_emits)
        if not base_emits:
            obs, reward, done, info = trainer.step([])
            turn += 1
            if done:
                break
            continue

        world = World.from_obs(obs, config)
        my_id = world.my_id
        other = [p.owner for p in world.planets
                 if p.owner not in (-1, my_id)]
        if not other:
            obs, reward, done, info = trainer.step(base_emits)
            turn += 1
            if done:
                break
            continue
        opp_id = max(set(other), key=other.count)

        clusters = find_solvable_clusters(world, my_id, opp_id)
        survivors = list(base_emits)
        if clusters:
            turns_with_cluster += 1
            src_to_cidx: dict[int, int] = {}
            for idx, c in enumerate(clusters):
                for pid in c.planet_ids:
                    src_to_cidx.setdefault(pid, idx)

            veto_idle: dict[int, bool] = {}
            survivors = []
            fired_this_turn = False
            for emit in base_emits:
                sid = int(emit[0])
                cidx = src_to_cidx.get(sid)
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
                    total_vetoes += 1
                    fired_this_turn = True
                    if len(veto_drops) < 8:
                        veto_drops.append({
                            "turn": turn,
                            "src": sid,
                            "dropped": list(emit),
                            "cluster": list(clusters[cidx].planet_ids),
                        })
                    continue
                survivors.append(emit)
            if fired_this_turn:
                turns_veto_fired += 1

        obs, reward, done, info = trainer.step(survivors)
        turn += 1
        if done:
            break

    print(f"=== seed={args.seed} vs {Path(args.vs).name} ===")
    print(f"total turns:                       {turn}")
    print(f"total base emits across all turns: {base_total_emits}")
    print(f"turns with solvable cluster:       {turns_with_cluster}")
    print(f"turns veto fired:                  {turns_veto_fired}")
    print(f"total veto suppressions:           {total_vetoes}")
    print(f"final reward (focal P0):           {reward}")
    if veto_drops:
        print("--- first few drops ---")
        for d in veto_drops:
            print(f" turn {d['turn']:3d}: src={d['src']:2d} "
                  f"dropped={d['dropped']} cluster={d['cluster']}")


if __name__ == "__main__":
    main()
