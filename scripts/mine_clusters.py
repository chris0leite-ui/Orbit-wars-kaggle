"""Walk 2P replay files, extract solvable clusters at sample turns,
solve each, and dump the results to `audit/tablebase/clusters.jsonl`.

Each output record:
  {
    "source_replay": "episode-XXX.json",
    "source_step": 60,
    "source_seat": 0,
    "planet_ids": [4, 17, 22],
    "my_id": 0, "opp_id": 1,
    "phantom_planet_id": 10022,
    "isolated_obs": {...},
    "best_action": [[src_id, angle, ships], ...],
    "value": 47.0,
    "depth_reached": 8,
    "nodes_searched": 256,
    "elapsed_ms": 213,
  }

The audit script (`audit_clusters.py`) reads these records and
compares `best_action` to `trajectory_roi.agent(isolated_obs)`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.cluster_solver.detector import find_solvable_clusters       # noqa: E402
from lib.cluster_solver.minimax import solve, DEFAULT_MAX_DEPTH      # noqa: E402
from lib.trajectory_layer import World                                # noqa: E402


REPLAY_DIR = _REPO / "audit/live-episodes/52784853"
OUT_PATH = _REPO / "audit/tablebase/clusters.jsonl"
SAMPLE_TURNS = (20, 40, 60, 80, 100, 140, 180)
DEFAULT_BUDGET_MS = 800.0
DEFAULT_MAX_DEPTH_MINE = 8


def _replay_obs(seat_state: dict) -> dict | None:
    """Normalize a replay seat-state's observation to the fields
    `World.from_obs` needs."""
    obs = seat_state.get("observation", {})
    if not obs.get("planets"):
        return None
    return {
        "player": int(obs.get("player", 0)),
        "step": int(obs.get("step", 0)),
        "planets": [list(p) for p in obs.get("planets", [])],
        "fleets": [list(f) for f in obs.get("fleets", []) or []],
        "comets": list(obs.get("comets", []) or []),
        "comet_planet_ids": list(obs.get("comet_planet_ids", []) or []),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in
                            obs.get("initial_planets",
                                    obs.get("planets", []))],
        "next_fleet_id": int(obs.get("next_fleet_id", 0)),
    }


def _is_2p_replay(replay: dict) -> bool:
    steps = replay.get("steps", [])
    return bool(steps) and len(steps[0]) == 2


def mine(max_replays: int, max_clusters: int, budget_ms: float,
         max_depth: int, out_path: Path) -> int:
    n_written = 0
    t_start = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out_fp:
        n_replays_seen = 0
        for path in sorted(REPLAY_DIR.glob("episode-*-replay.json")):
            if n_replays_seen >= max_replays:
                break
            with open(path) as f:
                try:
                    replay = json.load(f)
                except Exception:
                    continue
            if not _is_2p_replay(replay):
                continue
            n_replays_seen += 1
            steps = replay["steps"]
            for turn in SAMPLE_TURNS:
                if turn >= len(steps):
                    continue
                for seat in (0, 1):
                    obs = _replay_obs(steps[turn][seat])
                    if obs is None:
                        continue
                    try:
                        world = World.from_obs(obs)
                    except Exception:
                        continue
                    clusters = find_solvable_clusters(world)
                    for c in clusters:
                        result = solve(c.isolated_obs, c.my_id, c.opp_id,
                                       max_depth=max_depth,
                                       budget_ms=budget_ms)
                        record = {
                            "source_replay": path.name,
                            "source_step": turn,
                            "source_seat": seat,
                            "planet_ids": list(c.planet_ids),
                            "my_id": c.my_id,
                            "opp_id": c.opp_id,
                            "phantom_planet_id": c.phantom_planet_id,
                            "isolated_obs": c.isolated_obs,
                            "best_action": result.best_action,
                            "value": result.value,
                            "depth_reached": result.depth_reached,
                            "nodes_searched": result.nodes_searched,
                            "elapsed_ms": result.elapsed_ms,
                        }
                        out_fp.write(json.dumps(record) + "\n")
                        out_fp.flush()
                        n_written += 1
                        if n_written >= max_clusters:
                            elapsed = time.perf_counter() - t_start
                            print(f"  hit max_clusters={max_clusters}, "
                                  f"{n_written} written in {elapsed:.1f}s")
                            return n_written
            print(f"  {n_replays_seen} replays processed, "
                  f"{n_written} clusters so far", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"Done: {n_written} clusters written to {out_path} "
          f"in {elapsed:.1f}s")
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=20)
    parser.add_argument("--max-clusters", type=int, default=200)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH_MINE)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    n = mine(args.max_replays, args.max_clusters, args.budget_ms,
             args.max_depth, args.out)
    return 0 if n >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
