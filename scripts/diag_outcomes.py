"""Classify v8_scavenge's wins/losses vs an opponent by game length +
endgame ship/planet stats. Helps identify failure patterns (early
collapse vs late-game drift vs single-turn blowout).

Usage:  python scripts/diag_outcomes.py v7_0 [n_seeds]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.v8_scavenge import main as v8_main
from fast import _load_callable


def play_one(seed: int, opp_path: str, swap: bool):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500})
    env.reset(num_agents=2)
    opp_agent = _load_callable(opp_path)

    me_idx = 1 if swap else 0
    state = env.steps[0]
    n_steps = 0
    final_obs = None
    while True:
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        if swap:
            try:
                a0 = opp_agent(obs0, env.configuration)
            except TypeError:
                a0 = opp_agent(obs0)
            a1 = v8_main.agent(obs1, env.configuration)
        else:
            a0 = v8_main.agent(obs0, env.configuration)
            try:
                a1 = opp_agent(obs1, env.configuration)
            except TypeError:
                a1 = opp_agent(obs1)
        state = env.step([a0, a1])
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE":
            final_obs = state[me_idx]["observation"] if isinstance(state[me_idx], dict) else state[me_idx].observation
            break
        if n_steps >= 500:
            final_obs = state[me_idx]["observation"] if isinstance(state[me_idx], dict) else state[me_idx].observation
            break

    me = me_idx
    raw_planets = final_obs.planets if hasattr(final_obs, "planets") else final_obs.get("planets", [])
    raw_fleets = final_obs.fleets if hasattr(final_obs, "fleets") else final_obs.get("fleets", [])
    my_planets = sum(1 for p in raw_planets if int(p[1]) == me)
    opp_planets = sum(1 for p in raw_planets if int(p[1]) != me and int(p[1]) >= 0)
    my_ships = sum(int(p[5]) for p in raw_planets if int(p[1]) == me)
    opp_ships = sum(int(p[5]) for p in raw_planets if int(p[1]) != me and int(p[1]) >= 0)
    for f in raw_fleets:
        if int(f[1]) == me:
            my_ships += int(f[6])
        elif int(f[1]) >= 0:
            opp_ships += int(f[6])

    r_me = state[me_idx].get("reward") if isinstance(state[me_idx], dict) else state[me_idx].reward
    win = bool(r_me and r_me > 0)
    return {
        "seed": seed,
        "swap": swap,
        "win": win,
        "n_steps": n_steps,
        "my_planets": my_planets,
        "opp_planets": opp_planets,
        "my_ships": my_ships,
        "opp_ships": opp_ships,
    }


def main():
    opp = sys.argv[1] if len(sys.argv) > 1 else "v7_0"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    opp_path = {
        "nearest": str(REPO / "agents/simple/nearest.py"),
        "v7_0": str(REPO / "submissions/v7_0_drop_one.py"),
        "v4_planner": str(REPO / "submissions/v4_planner.py"),
    }.get(opp, opp)

    print(f"== v8_scavenge vs {opp}, n_seeds={n_seeds} (each plays once each side) ==")
    print(f"{'seed':>4} {'side':>4} {'win?':>4} {'steps':>5} {'myP':>3}/{'opP':<3} {'myS':>5}/{'opS':<5}  category")
    print("-" * 70)
    results = []
    for seed in range(n_seeds):
        for swap in (False, True):
            t0 = time.perf_counter()
            r = play_one(seed, opp_path, swap)
            dt = time.perf_counter() - t0
            results.append(r)
            cat = []
            if r["n_steps"] < 80:
                cat.append("EARLY")
            elif r["n_steps"] < 200:
                cat.append("MID")
            else:
                cat.append("LATE")
            if r["my_planets"] == 0:
                cat.append("eliminated")
            elif r["opp_planets"] == 0:
                cat.append("won_by_elim")
            else:
                if r["win"]:
                    cat.append("won_by_ships")
                else:
                    cat.append("lost_by_ships")
            side = "P1" if swap else "P0"
            print(f"{r['seed']:>4} {side:>4} {('W' if r['win'] else 'L'):>4} {r['n_steps']:>5} "
                  f"{r['my_planets']:>3}/{r['opp_planets']:<3} "
                  f"{r['my_ships']:>5}/{r['opp_ships']:<5}  {' '.join(cat)} ({dt:.1f}s)")

    n = len(results)
    wins = sum(1 for r in results if r["win"])
    print(f"\nTotal: {wins}/{n} ({100*wins/n:.1f}%)")
    early_loss = sum(1 for r in results if not r["win"] and r["n_steps"] < 80)
    mid_loss = sum(1 for r in results if not r["win"] and 80 <= r["n_steps"] < 200)
    late_loss = sum(1 for r in results if not r["win"] and r["n_steps"] >= 200)
    print(f"  losses: early={early_loss}  mid={mid_loss}  late={late_loss}")
    elim_losses = sum(1 for r in results if not r["win"] and r["my_planets"] == 0)
    ship_losses = sum(1 for r in results if not r["win"] and r["my_planets"] > 0)
    print(f"  loss type: eliminated={elim_losses}  outscored={ship_losses}")


if __name__ == "__main__":
    main()
