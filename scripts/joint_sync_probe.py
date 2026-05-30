"""Phase-0 live census for synchronized-arrival JOINT coalitions.

Runs a few real games with BASELINE_JOINT_SYNC=1 and the champion config,
counting how often a synchronized coalition is actually FORMED and emitted
(near-leg sync_joint commits) vs how often the waiting leg then FIRES. The
open risk this answers: the favor head may value a stacked capture below
the ship-in-flight cost, in which case sync never emits and an A/B is a
no-op. If sync_commits == 0 across games, STOP and diagnose before A/B.

Usage: python scripts/joint_sync_probe.py [n_games]
"""
from __future__ import annotations

import os
import sys

# Champion production config + sync ON (matches the A/B focal).
for k, v in {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_LAUNCH_RULES": "1", "BASELINE_CAPTURE_HORIZON_K": "10",
    "BASELINE_JOINT": "1", "BASELINE_JOINT_SYNC": "1",
}.items():
    os.environ.setdefault(k, v)

from kaggle_environments import make  # noqa: E402
import agents.baseline.chooser_trajectory as ct  # noqa: E402
import agents.baseline.main as M  # noqa: E402

_orig_choose = ct.choose_trajectory
STATS = {"turns": 0, "sync_emitted": 0, "with_sync_turns": 0}


def _counting_choose(*a, **k):
    moves, commits = _orig_choose(*a, **k)
    STATS["turns"] += 1
    n = sum(1 for c in commits if c.get("sync_joint"))
    STATS["sync_emitted"] += n
    if n:
        STATS["with_sync_turns"] += 1
    return moves, commits


ct.choose_trajectory = _counting_choose


def main() -> int:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fired = 0
    for g in range(n_games):
        env = make("orbit_wars", configuration={"seed": 100 + g}, debug=False)
        env.run([M.agent, M.agent])
        states = env.steps[-1]
        statuses = [s["status"] for s in states]
        print(f"game {g} seed={100+g}: end statuses={statuses} "
              f"turns_so_far={STATS['turns']} sync_emitted={STATS['sync_emitted']}")
        if any(st == "ERROR" for st in statuses):
            print("  !!! ERROR status — agent crashed")
            return 2
    print("\n=== CENSUS ===")
    print(f"games={n_games}  total chooser turns={STATS['turns']}")
    print(f"sync coalitions emitted (near-leg commits)={STATS['sync_emitted']}"
          f"  on {STATS['with_sync_turns']} turns")
    if STATS["sync_emitted"] == 0:
        print("VERDICT: sync NEVER fires — favor likely under-values stacked "
              "captures vs ship-in-flight cost. STOP, diagnose before A/B.")
        return 1
    print("VERDICT: sync coalitions DO fire — proceed to A/B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
