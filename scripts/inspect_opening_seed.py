"""Inspect the opening behaviour on a specific seed.

Replays the env with our agent in seat 0 vs a baseline opponent, captures
every fleet emission for both seats during steps 0..OPENING_HORIZON, and
prints a side-by-side trace.

Usage:
    python scripts/inspect_opening_seed.py --seed 384458460 --opp v7_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402


def _load_agent(spec: str):
    """Load an agent: path to .py file OR module path under agents/."""
    if spec.endswith(".py"):
        return spec
    p1 = REPO / "agents" / spec / "main.py"
    if p1.exists():
        return str(p1)
    p2 = REPO / "submissions" / f"{spec}.py"
    if p2.exists():
        return str(p2)
    p3 = REPO / "agents" / "simple" / f"{spec}.py"
    if p3.exists():
        return str(p3)
    raise FileNotFoundError(f"could not resolve agent: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--me", default="analytical_phase_c")
    ap.add_argument("--opp", default="v7_0")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    me_path = _load_agent(args.me)
    opp_path = _load_agent(args.opp)
    print(f"# Seed {args.seed}: me={args.me} vs opp={args.opp}")
    print(f"# step={args.steps}")

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.reset(num_agents=2)
    initial_obs = env.state[0]["observation"]
    planets0 = initial_obs.get("planets", [])
    print(f"\n# Initial planets ({len(planets0)} total):")
    for p in planets0:
        pid, owner, x, y, r, ships, prod = p
        owner_str = "ME" if int(owner) == 0 else ("OPP" if int(owner) == 1 else "N")
        print(f"   p{pid:>3} {owner_str:<3}  pos=({x:6.2f},{y:6.2f}) "
              f"r={r:.2f}  ships={ships:>3}  prod={prod}")

    # Step the env turn by turn, observing actions.
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([me_path, opp_path])

    # env.steps[t] is a list of two agent-step dicts.
    print(f"\n# Per-turn emissions (steps 0..{args.steps - 1}):")
    print(f"{'step':>4}  {'me_owned':>8} {'opp_owned':>9}  "
          f"{'me_actions':>40}  {'opp_actions':>40}")
    for t in range(min(args.steps, len(env.steps))):
        step_data = env.steps[t]
        me_obs = step_data[0]["observation"]
        opp_obs = step_data[1]["observation"]
        me_action = step_data[0].get("action") or []
        opp_action = step_data[1].get("action") or []

        # Owned planets
        planets = me_obs.get("planets", [])
        me_owned = sum(1 for p in planets if int(p[1]) == 0)
        opp_owned = sum(1 for p in planets if int(p[1]) == 1)

        # Format actions
        me_str = " ".join(
            f"{int(m[0])}→{m[2]}s@{m[1]:.2f}" for m in me_action
        ) or "-"
        opp_str = " ".join(
            f"{int(m[0])}→{m[2]}s@{m[1]:.2f}" for m in opp_action
        ) or "-"
        print(f"{t:>4}  {me_owned:>8} {opp_owned:>9}  "
              f"{me_str[:40]:>40}  {opp_str[:40]:>40}")

    # Final score
    final_step = env.steps[-1]
    final_obs = final_step[0]["observation"]
    final_planets = final_obs.get("planets", [])
    me_planets = [p for p in final_planets if int(p[1]) == 0]
    opp_planets = [p for p in final_planets if int(p[1]) == 1]
    me_prod = sum(int(p[6]) for p in me_planets)
    opp_prod = sum(int(p[6]) for p in opp_planets)
    print(f"\n# Final: me_planets={len(me_planets)} prod={me_prod}, "
          f"opp_planets={len(opp_planets)} prod={opp_prod}")
    print(f"#         steps_played={len(env.steps)}")
    rewards = [s.get("reward") for s in final_step]
    print(f"#         rewards: {rewards}")


if __name__ == "__main__":
    main()
