"""Phase B foundation diagnostic — opp-model predictive accuracy.

Measures how well `predict_opp_multi_launch` predicts opp's actual
behavior over the next HORIZON ticks. For each turn of a real game
(analytical agent vs trajectory baseline at seat 1):

  predicted = predict_opp_multi_launch(world, me, num_seats)
             → set of (target_pid, eta_absolute) opp launches projected
  actual    = opp's actual launches in the next HORIZON ticks,
              with their resolved target_pid via predict_fleet_fate

Per-turn metrics:
  - target_jaccard:    |pred_targets ∩ actual_targets| / |pred ∪ actual|
  - count_mae:         |n_predicted - n_actual|
  - per-pid count_mae: mean across pids of |pred[pid] - actual[pid]|

Aggregated mean / median across 4 seeds × full game.

Informs Phase D's opp-model design: a target_jaccard ~60%+ suggests
the greedy-ROI heuristic is salvageable; <30% says we need a mirror-
trajectory or learned model.

Usage:
  python -m scripts.diag_opp_predict_accuracy --seeds 42 7 13 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from lib.intent import World
from lib.joint_solver.opp_projection import HORIZON, predict_opp_multi_launch
from lib.trajectory import predict_fleet_fate


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}


def _num_seats(planets, fleets) -> int:
    owners = {int(p.owner) for p in planets if int(p.owner) >= 0}
    owners.update(int(f.owner) for f in fleets if int(f.owner) >= 0)
    if not owners:
        return 2
    return max(2, max(owners) + 1)


def _resolve_target_of_launch(src_planet, angle, ships, world):
    """Ray-cast a hypothetical opp launch via predict_fleet_fate.

    Returns (target_pid, eta_relative) or None if no target."""
    try:
        fate = predict_fleet_fate(src_planet, None, angle, ships, world, wait_N=0)
    except Exception:
        return None
    if fate is None or getattr(fate, "outcome", "") != "target":
        return None
    tgt = getattr(fate, "target", None) or getattr(fate, "target_planet", None)
    if tgt is None:
        return None
    return (int(tgt.id), int(getattr(fate, "eta", 0)))


def run_one_game(seed: int, focal_path: str = "agents/analytical/main.py",
                 opp_path: str = "agents/baseline/main.py") -> dict:
    """Run one game; capture predicted vs actual opp launches per turn."""
    from kaggle_environments import make

    # Per-turn capture: predicted opp arrivals from focal's POV
    predicted_per_turn: dict[int, list] = {}  # step → list of (tgt_pid, eta_abs)

    def focal_wrapper(obs, configuration=None):
        """Calls predict_opp_multi_launch before delegating to analytical."""
        obs_d = _as_dict(obs)
        step = int(obs_d.get("step", 0) or 0)
        me = int(obs_d.get("player", 0))
        raw_planets = obs_d.get("planets", []) or []
        raw_fleets = obs_d.get("fleets", []) or []
        if raw_planets:
            planets = [Planet(*p) for p in raw_planets]
            fleets = [Fleet(*f) for f in raw_fleets]
            num_seats = _num_seats(planets, fleets)
            try:
                world = World.from_obs(obs_d)
                pred = predict_opp_multi_launch(world, me, num_seats)
            except Exception:
                pred = []
            predicted_per_turn[step] = [
                (int(tgt_pid), int(eta_abs))
                for (tgt_pid, eta_abs, _o, _s) in pred
            ]

        # Delegate to analytical agent
        from agents.analytical.main import agent as analytical_agent
        return analytical_agent(obs, configuration)

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([focal_wrapper, opp_path])
    except Exception as e:
        return {"seed": seed, "error": str(e)}

    # Extract opp's actual emissions per step and resolve targets via
    # ray-cast against the world state at the time of launch.
    actual_per_turn: dict[int, list] = {}  # step → list of (tgt_pid, eta_rel)
    for step_idx, step_state in enumerate(env.steps):
        if step_idx >= len(env.steps) - 1:
            break  # last step is terminal
        opp_state = step_state[1]
        opp_action = getattr(opp_state, "action", None) or []
        if not opp_action:
            actual_per_turn[step_idx] = []
            continue
        # The world state at step_idx is what opp saw when deciding
        opp_obs = _as_dict(opp_state.observation)
        if not opp_obs.get("planets"):
            actual_per_turn[step_idx] = []
            continue
        try:
            opp_world = World.from_obs(opp_obs)
        except Exception:
            actual_per_turn[step_idx] = []
            continue
        resolved = []
        for move in opp_action:
            if len(move) < 3:
                continue
            src_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            src_planet = opp_world.planets_by_id.get(src_id)
            if src_planet is None:
                continue
            res = _resolve_target_of_launch(src_planet, angle, ships, opp_world)
            if res is None:
                continue
            tgt_pid, eta_rel = res
            resolved.append((tgt_pid, eta_rel))
        actual_per_turn[step_idx] = resolved

    # Per-turn metric: for each turn t when focal predicted, compare
    # predicted target set vs actual target set over t..t+HORIZON.
    per_turn_metrics = []
    for step, pred_list in sorted(predicted_per_turn.items()):
        pred_targets = {pid for (pid, _eta) in pred_list}
        actual_targets: set[int] = set()
        actual_count = 0
        for t_offset in range(HORIZON):
            t = step + t_offset
            if t in actual_per_turn:
                for (pid, _eta_rel) in actual_per_turn[t]:
                    actual_targets.add(pid)
                    actual_count += 1
        union = pred_targets | actual_targets
        intersect = pred_targets & actual_targets
        jaccard = (len(intersect) / len(union)) if union else 1.0
        per_turn_metrics.append({
            "step": step,
            "n_predicted": len(pred_list),
            "n_actual_in_window": actual_count,
            "n_predicted_targets": len(pred_targets),
            "n_actual_targets": len(actual_targets),
            "target_jaccard": jaccard,
            "count_mae": abs(len(pred_list) - actual_count),
        })

    if per_turn_metrics:
        jaccards = [m["target_jaccard"] for m in per_turn_metrics]
        count_maes = [m["count_mae"] for m in per_turn_metrics]
        summary = {
            "n_predict_turns": len(per_turn_metrics),
            "mean_target_jaccard": sum(jaccards) / len(jaccards),
            "median_target_jaccard": sorted(jaccards)[len(jaccards) // 2],
            "mean_count_mae": sum(count_maes) / len(count_maes),
            "median_count_mae": sorted(count_maes)[len(count_maes) // 2],
            "mean_n_predicted": sum(m["n_predicted"] for m in per_turn_metrics) / len(per_turn_metrics),
            "mean_n_actual_in_window": sum(m["n_actual_in_window"] for m in per_turn_metrics) / len(per_turn_metrics),
        }
    else:
        summary = {"n_predict_turns": 0}

    return {
        "seed": seed,
        "focal": focal_path,
        "opp": opp_path,
        "n_turns_total": len(env.steps),
        "summary": summary,
        "per_turn": per_turn_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 1])
    ap.add_argument("--focal", default="agents/analytical/main.py")
    ap.add_argument("--opp", default="agents/baseline/main.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    games = []
    for seed in args.seeds:
        print(f"=== seed={seed} ===")
        result = run_one_game(seed, args.focal, args.opp)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            s = result["summary"]
            print(f"  turns={result['n_turns_total']} predict_turns={s.get('n_predict_turns', 0)}")
            if s.get("n_predict_turns", 0) > 0:
                print(f"  mean_target_jaccard={s['mean_target_jaccard']:.3f} "
                      f"median={s['median_target_jaccard']:.3f}")
                print(f"  mean_count_mae={s['mean_count_mae']:.2f} "
                      f"median={s['median_count_mae']:.2f}")
                print(f"  mean_n_pred={s['mean_n_predicted']:.2f} "
                      f"mean_n_actual={s['mean_n_actual_in_window']:.2f}")
        games.append(result)

    # Aggregate across seeds
    valid = [g for g in games if "error" not in g and g["summary"].get("n_predict_turns", 0) > 0]
    if valid:
        agg_jaccard = sum(g["summary"]["mean_target_jaccard"] for g in valid) / len(valid)
        agg_count_mae = sum(g["summary"]["mean_count_mae"] for g in valid) / len(valid)
        print(f"\n=== AGGREGATE ({len(valid)}/{len(games)} seeds) ===")
        print(f"  mean target_jaccard = {agg_jaccard:.3f}")
        print(f"  mean count_mae      = {agg_count_mae:.2f}")

    out_path = args.out
    if out_path is None:
        out_dir = Path("audit/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"opp-predict-accuracy-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    Path(out_path).write_text(json.dumps({"games": games}, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
