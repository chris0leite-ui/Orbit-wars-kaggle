"""Opening-phase diagnostic probe.

Instruments v3 (or any focal agent) across games against multiple
opponents and logs per-turn state for the FIRST 30 steps. Used to
investigate the user's observation that "after a few steps the
opponent is already ahead of us in ship count before combat
encounters."

Per-turn log for the focal agent's POV:
  - step
  - my_id, opp_id
  - my_ships (planets + in-flight), opp_ships
  - my_planets count, opp_planets count
  - my_production rate (sum of production over owned planets), opp_production
  - my_launches_emitted this turn (count of items in agent's returned action)
  - first_launch_step (set when launches_emitted > 0 for the first time)

Output: per-game per-turn rows in JSON. Aggregates separately.

Usage:
  python -m scripts.opening_probe \\
    --focal v3_snipe \\
    --opponents v3_snipe v2 roi precision \\
    --seeds 8 \\
    --max-step 30 \\
    --out audit/tournaments/opening_probe_v1.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]


POLICY_PATHS = {
    "v3_snipe": "agents/v3_snipe/main.py",
    "v7_minimax": "agents/v7_minimax/main.py",
    "v2": "agents/v2/main.py",
    "roi": "agents/simple/roi.py",
    "baseline": "data/main.py",
    "precision": "agents/precision/main.py",
}


def _load(name: str):
    path = POLICY_PATHS.get(name)
    if path is None:
        raise ValueError(f"unknown policy: {name}")
    full = REPO / path
    spec = importlib.util.spec_from_file_location(f"_probe_{name}", full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _instrument(agent_fn, log_list, focal_player_id):
    """Wrap agent so it logs per-turn state from focal player's POV."""
    def wrapped(obs):
        action = agent_fn(obs)
        step = int(_obs_get(obs, "step", 0))
        my_id = int(_obs_get(obs, "player", 0))
        if my_id != focal_player_id:
            return action  # only log on focal player's turn-call
        planets = _obs_get(obs, "planets", []) or []
        fleets = _obs_get(obs, "fleets", []) or []
        opp_id = 1 - my_id
        my_ships_p = sum(int(p[5]) for p in planets if p[1] == my_id)
        my_ships_f = sum(int(f[6]) for f in fleets if f[1] == my_id)
        opp_ships_p = sum(int(p[5]) for p in planets if p[1] == opp_id)
        opp_ships_f = sum(int(f[6]) for f in fleets if f[1] == opp_id)
        log_list.append({
            "step": step,
            "my_ships_total": my_ships_p + my_ships_f,
            "my_ships_on_planets": my_ships_p,
            "my_ships_in_flight": my_ships_f,
            "opp_ships_total": opp_ships_p + opp_ships_f,
            "opp_ships_on_planets": opp_ships_p,
            "opp_ships_in_flight": opp_ships_f,
            "my_planets": sum(1 for p in planets if p[1] == my_id),
            "opp_planets": sum(1 for p in planets if p[1] == opp_id),
            "neutral_planets": sum(1 for p in planets if p[1] == -1),
            "my_production": sum(int(p[6]) for p in planets if p[1] == my_id),
            "opp_production": sum(int(p[6]) for p in planets if p[1] == opp_id),
            "my_launches": len(action),
            "my_fleets_count": sum(1 for f in fleets if f[1] == my_id),
            "opp_fleets_count": sum(1 for f in fleets if f[1] == opp_id),
        })
        return action
    return wrapped


def run_one(focal_agent, opp_agent, seed: int, focal_side: int,
            max_step: int, episode_steps: int = 500):
    """Run one game; return focal-perspective per-turn log up to max_step."""
    env = make("orbit_wars",
               configuration={"episodeSteps": episode_steps, "seed": seed},
               debug=False)
    log: list = []
    # Wrap only the focal agent
    focal_wrapped = _instrument(focal_agent, log, focal_player_id=focal_side)
    agents = [focal_wrapped, opp_agent] if focal_side == 0 else [opp_agent, focal_wrapped]
    t0 = time.time()
    env.run(agents)
    elapsed = time.time() - t0
    # Trim to opening window
    log_opening = [r for r in log if r["step"] <= max_step]
    final = env.steps[-1]
    final_rewards = [s.reward for s in final]
    return {
        "log": log_opening,
        "final_step": len(env.steps),
        "final_rewards": final_rewards,
        "elapsed_s": elapsed,
    }


def summarize_runs(runs):
    """Compute aggregate metrics across all runs."""
    # Per-step aggregates: average my_ships - opp_ships, average my_planets - opp_planets, etc.
    if not runs:
        return {}
    all_steps_data = {}  # step -> list of metric values across runs
    for run in runs:
        for row in run["log"]:
            s = row["step"]
            if s not in all_steps_data:
                all_steps_data[s] = []
            all_steps_data[s].append(row)

    per_step = {}
    for s in sorted(all_steps_data):
        rows = all_steps_data[s]
        ship_delta = [r["my_ships_total"] - r["opp_ships_total"] for r in rows]
        planet_delta = [r["my_planets"] - r["opp_planets"] for r in rows]
        prod_delta = [r["my_production"] - r["opp_production"] for r in rows]
        launches = [r["my_launches"] for r in rows]
        per_step[s] = {
            "n": len(rows),
            "mean_ship_delta": statistics.mean(ship_delta),
            "stdev_ship_delta": statistics.stdev(ship_delta) if len(ship_delta) > 1 else 0,
            "mean_planet_delta": statistics.mean(planet_delta),
            "mean_prod_delta": statistics.mean(prod_delta),
            "mean_launches_this_turn": statistics.mean(launches),
            "n_idle_turns": sum(1 for l in launches if l == 0),
            "first_launch_seen": any(l > 0 for l in launches),
        }
    # Find first step where mean_ship_delta turns NEGATIVE
    first_negative = None
    for s in sorted(per_step):
        if per_step[s]["mean_ship_delta"] < 0:
            first_negative = s
            break
    # Find when each side first launches (per run)
    my_first_launch_steps = []
    for run in runs:
        for row in run["log"]:
            if row["my_launches"] > 0:
                my_first_launch_steps.append(row["step"])
                break
    return {
        "per_step": per_step,
        "first_negative_ship_delta_step": first_negative,
        "first_launch_step_mean": statistics.mean(my_first_launch_steps) if my_first_launch_steps else None,
        "first_launch_step_median": statistics.median(my_first_launch_steps) if my_first_launch_steps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="v3_snipe")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-step", type=int, default=30)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"Opening probe: focal={args.focal} vs opponents={args.opponents}, "
          f"{args.seeds} seeds × 2 sides, max_step={args.max_step}", flush=True)

    focal_agent = _load(args.focal)
    opp_agents = {opp: _load(opp) for opp in args.opponents}
    print(f"  loaded {len(opp_agents) + 1} agents", flush=True)

    all_results = {}
    t_all = time.time()
    for opp_name, opp_fn in opp_agents.items():
        print(f"\n[{opp_name}]", flush=True)
        runs = []
        for seed in range(args.seeds):
            for side in (0, 1):
                t = time.time()
                run = run_one(focal_agent, opp_fn, seed, side,
                              args.max_step, args.episode_steps)
                run["opp"] = opp_name
                run["seed"] = seed
                run["focal_side"] = side
                runs.append(run)
                # Concise per-run print
                final_rew = run["final_rewards"]
                last_row = run["log"][-1] if run["log"] else {}
                opening_delta = last_row.get("my_ships_total", 0) - last_row.get("opp_ships_total", 0)
                print(f"  seed={seed} side={side} step{args.max_step}: "
                      f"ship_delta={opening_delta:+d} final_rew={final_rew} ({time.time()-t:.0f}s)",
                      flush=True)
        all_results[opp_name] = {
            "runs": runs,
            "summary": summarize_runs(runs),
        }
        s = all_results[opp_name]["summary"]
        print(f"  summary: first_neg_delta={s.get('first_negative_ship_delta_step')}, "
              f"first_launch_median={s.get('first_launch_step_median')}", flush=True)

    out = {
        "focal": args.focal,
        "opponents": list(opp_agents.keys()),
        "n_seeds": args.seeds,
        "max_step": args.max_step,
        "elapsed_total_s": round(time.time() - t_all, 0),
        "results": all_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes) in "
          f"{out['elapsed_total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
