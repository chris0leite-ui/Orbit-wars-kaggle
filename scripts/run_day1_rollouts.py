"""Day-1 baseline rollouts: shipped main.py vs random and self-play.

Records per-seed reward + final ship counts so the audit log can cite
load-bearing facts without re-running the simulator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]
BASELINE = str(REPO / "data" / "main.py")
SEEDS = [42, 1, 7, 13, 31, 100]


def final_ships(state):
    """Sum ships on planets + in fleets per player at the final step."""
    obs0 = state[0].observation
    planets = obs0.get("planets", [])
    fleets = obs0.get("fleets", [])
    by_owner: dict[int, int] = {}
    for p in planets:
        owner = p[1]
        ships = p[5]
        if owner >= 0:
            by_owner[owner] = by_owner.get(owner, 0) + ships
    for f in fleets:
        owner = f[1]
        ships = f[6]
        by_owner[owner] = by_owner.get(owner, 0) + ships
    return by_owner


def run_match(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    statuses = [s.status for s in final]
    ships = final_ships(final)
    n_steps = len(env.steps)
    return {
        "seed": seed,
        "rewards": rewards,
        "statuses": statuses,
        "final_ships": {str(k): v for k, v in sorted(ships.items())},
        "n_steps": n_steps,
    }


def winrate(results, ego_idx=0):
    wins = sum(1 for r in results if r["rewards"][ego_idx] == 1)
    losses = sum(1 for r in results if r["rewards"][ego_idx] == -1)
    draws = sum(1 for r in results if r["rewards"][ego_idx] == 0)
    return {"wins": wins, "losses": losses, "draws": draws, "n": len(results)}


def main():
    print("=== baseline (P0) vs random (P1) ===")
    bvr = [run_match([BASELINE, "random"], s) for s in SEEDS]
    for r in bvr:
        print(json.dumps(r))
    print("summary baseline-vs-random P0:", winrate(bvr, 0))

    print()
    print("=== baseline (P0) vs baseline (P1) — self-play / validation gate ===")
    bvb = [run_match([BASELINE, BASELINE], s) for s in SEEDS]
    for r in bvb:
        print(json.dumps(r))
    print("summary baseline-vs-baseline P0:", winrate(bvb, 0))

    out = {
        "baseline_vs_random": bvr,
        "baseline_vs_baseline": bvb,
        "seeds": SEEDS,
    }
    out_path = REPO / "audit" / "2026-05-09-day-1-rollouts.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
