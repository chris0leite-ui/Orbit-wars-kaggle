"""Diagnostic: log per-launch (turn, src, ships_at_src_before, ships_launched)
for the orbitfix bundle, in 2P (vs v7_0) and 4P FFA self-play.

Hypothesis: chooser is min-investment — launches at <50% of source ships
in early/mid game even when surplus exists.

Output: per-game launch list + histogram of launch-fraction by phase.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match orbitfix env stack BEFORE any agent imports.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

from kaggle_environments import make
from fast import _load_callable


def play_and_trace(seed: int, num_seats: int, focal_idx: int, opp_path: str):
    """Play one game; return list of (turn, src_id, ships_before, ships_launched,
    target_planet_id_estimate, total_my_ships, total_my_planets) for focal."""
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": 500},
        debug=False,
    )
    env.reset(num_agents=num_seats)

    from agents.baseline.main import agent as orbitfix_agent
    opp = _load_callable(opp_path)
    agents_list = [opp] * num_seats
    agents_list[focal_idx] = orbitfix_agent

    state = env.steps[0]
    trace = []
    n_steps = 0
    while True:
        # Build action per seat.
        per_seat_actions = []
        for s_idx in range(num_seats):
            obs_s = state[s_idx]["observation"] if isinstance(state[s_idx], dict) else state[s_idx].observation
            try:
                a = agents_list[s_idx](obs_s, env.configuration)
            except TypeError:
                a = agents_list[s_idx](obs_s)
            per_seat_actions.append(a)

        # Capture focal's pre-step state.
        focal_obs = state[focal_idx]["observation"] if isinstance(state[focal_idx], dict) else state[focal_idx].observation
        focal_obs_d = focal_obs if isinstance(focal_obs, dict) else dict(focal_obs)
        planets = focal_obs_d.get("planets", []) or []
        me = focal_idx  # in 2P/4P, agent seat == player id
        # Planet = (id, owner, x, y, radius, ships, production)
        planet_by_id = {int(p[0]): p for p in planets}
        my_planets = [p for p in planets if int(p[1]) == me]
        total_my_ships = sum(int(p[5]) for p in my_planets)
        total_my_planets = len(my_planets)

        focal_actions = per_seat_actions[focal_idx]
        if focal_actions:
            for action in focal_actions:
                # action = [src_id, angle, ships]
                src_id = int(action[0])
                ships = int(action[2])
                src = planet_by_id.get(src_id)
                ships_before = int(src[5]) if src else -1
                trace.append({
                    "turn": n_steps,
                    "src_id": src_id,
                    "ships_before": ships_before,
                    "ships_launched": ships,
                    "fraction": (ships / ships_before) if ships_before > 0 else 0.0,
                    "total_my_ships": total_my_ships,
                    "total_my_planets": total_my_planets,
                })

        # Step.
        state = env.step(per_seat_actions)
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE" or n_steps >= 500:
            break
    return trace, n_steps


def summarize(trace, label):
    print(f"\n=== {label}  (launches={len(trace)}) ===")
    if not trace:
        print("  no launches")
        return
    # Histogram of launch-fraction.
    buckets = [0, 0.25, 0.50, 0.75, 0.90, 1.01]
    bucket_labels = ["<25%", "25-50%", "50-75%", "75-90%", "90-100%"]
    counts = [0] * 5
    for e in trace:
        f = e["fraction"]
        for i in range(5):
            if buckets[i] <= f < buckets[i + 1]:
                counts[i] += 1
                break
    print(f"  launch-fraction histogram (n={len(trace)}):")
    for lbl, c in zip(bucket_labels, counts):
        pct = 100.0 * c / len(trace)
        bar = "#" * int(pct / 2)
        print(f"    {lbl:>8s}  {c:4d} ({pct:5.1f}%)  {bar}")

    # Stats by phase.
    def phase(t):
        if t < 30:
            return "early(0-30)"
        if t < 80:
            return "mid(30-80)"
        return "late(80+)"
    by_phase = {}
    for e in trace:
        by_phase.setdefault(phase(e["turn"]), []).append(e)
    print(f"  by phase:")
    for ph_name in ["early(0-30)", "mid(30-80)", "late(80+)"]:
        es = by_phase.get(ph_name, [])
        if not es:
            print(f"    {ph_name:>14s}  no launches")
            continue
        fracs = [e["fraction"] for e in es]
        ships = [e["ships_launched"] for e in es]
        srcs = [e["ships_before"] for e in es]
        n = len(es)
        mean_frac = sum(fracs) / n
        mean_ships = sum(ships) / n
        mean_src = sum(srcs) / n
        print(f"    {ph_name:>14s}  n={n:3d}  mean_frac={mean_frac:.2f}  mean_launched={mean_ships:5.1f}  mean_src_before={mean_src:5.1f}")

    # Per-turn aggregate: total ships idle (total_my_ships - launched_this_turn).
    print(f"  per-turn aggregate (first 20 turns with launches):")
    by_turn = {}
    for e in trace:
        by_turn.setdefault(e["turn"], []).append(e)
    turns = sorted(by_turn.keys())[:20]
    for t in turns:
        es = by_turn[t]
        total_my = es[0]["total_my_ships"]
        launched = sum(e["ships_launched"] for e in es)
        idle = total_my - launched
        n_planets = es[0]["total_my_planets"]
        n_launches = len(es)
        print(f"    turn={t:3d}  my_planets={n_planets:2d}  my_ships={total_my:4d}  launched_this_turn={launched:4d}  fleets={n_launches}  idle_after={idle:4d}")


def main():
    print("[diag] orbitfix launch-fraction diagnostic")
    print(f"[diag] bundle stack: BASELINE_ORBITAL_SAFETY=1, JOINT_AGGR=1, REINFORCE=1\n")

    # 2P self-play (seed similar to local A/B harness).
    trace_2p, steps_2p = play_and_trace(
        seed=42, num_seats=2, focal_idx=0,
        opp_path="submissions/v7_0_drop_one.py",
    )
    summarize(trace_2p, f"2P vs v7_0  seed=42  steps={steps_2p}")

    # 4P FFA (mimics live ladder).
    trace_4p, steps_4p = play_and_trace(
        seed=1511945213, num_seats=4, focal_idx=0,
        opp_path="submissions/v7_0_drop_one.py",
    )
    summarize(trace_4p, f"4P FFA vs v7_0x3  seed=1511945213  steps={steps_4p}")


if __name__ == "__main__":
    main()
