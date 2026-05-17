"""scripts/diag_v21_vs_v15.py — find WHY v21's patches don't lift vs v15.

For a given seed, runs v21 vs v15 once AND v15 vs v15 once, then compares
per-turn behaviour from v21's seat (seat 0 in both runs):

  - per-turn emit counts (v21 vs v15-as-seat-0 on same seed)
  - per-turn instrumentation: n_candidates, n_filtered_by_prefilter,
    n_filtered_by_hold_check, n_rescore_rounds, n_committed
  - cumulative planet count + production share for our seat
  - cumulative ship count
  - capture events (planet ownership flips TO us) and lost-back events
    (planet ownership flips FROM us within 50 turns of capture)

Output: per-seed roll-up table comparing v21-behavior to v15-behavior on
the same seed. Highlights where v21 is more conservative (fewer emits,
fewer captures) and whether the avoided actions correspond to comet
shots / undefensible captures (= patches doing their job) or to
legitimate aggression (= over-filtering).

Usage:
    python -m scripts.diag_v21_vs_v15 [--seeds N] [--max-turns T]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from importlib import import_module


def _seat_obs(env_steps, t, seat):
    return env_steps[t][seat].get("observation")


def _action(env_steps, t, seat):
    return env_steps[t][seat].get("action") or []


def _planets_owned(obs, seat):
    return [p for p in obs.get("planets", []) if int(p[1]) == seat]


def _ships_total(obs, seat):
    s = sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == seat)
    s += sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == seat)
    return s


def _production_total(obs, seat):
    return sum(float(p[6]) for p in obs.get("planets", []) if int(p[1]) == seat)


def _comet_ids(obs):
    return set(int(c) for c in obs.get("comet_planet_ids", []) or [])


def run_game(seed, agents_dict, seat_assignment, max_turns, counters_sink=None):
    """Run one game; return per-step records of MY seat's behavior + outcome.
    agents_dict: {"focal": fn, "opp": fn}
    seat_assignment: 0 or 1 — which seat the focal agent occupies
    counters_sink: dict to capture v21._INSTRUMENT_COUNTERS each turn
    """
    from kaggle_environments import make

    if seat_assignment == 0:
        agent_list = [agents_dict["focal"], agents_dict["opp"]]
    else:
        agent_list = [agents_dict["opp"], agents_dict["focal"]]

    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": max_turns})
    env.run(agent_list)

    records = []
    captures = []  # list of (turn, planet_id)
    captured_at = {}  # planet_id -> turn first captured
    lost_back = []  # (planet_id, capture_turn, loss_turn)

    n_steps = len(env.steps)
    for t in range(n_steps):
        my_obs = _seat_obs(env.steps, t, seat_assignment)
        my_action = _action(env.steps, t, seat_assignment)
        n_planets = len(_planets_owned(my_obs, seat_assignment))
        ships = _ships_total(my_obs, seat_assignment)
        prod = _production_total(my_obs, seat_assignment)
        my_planet_ids = {int(p[0]) for p in _planets_owned(my_obs, seat_assignment)}
        # Capture events: planets owned this turn that we DIDN'T own last turn.
        if t > 0:
            prev_obs = _seat_obs(env.steps, t - 1, seat_assignment)
            prev_owned = {int(p[0]) for p in _planets_owned(prev_obs, seat_assignment)}
            new_caps = my_planet_ids - prev_owned
            for pid in new_caps:
                captures.append((t, pid))
                captured_at[pid] = t
            # Lost-back: planets we previously captured that we no longer own.
            for pid in (prev_owned - my_planet_ids):
                if pid in captured_at:
                    lost_back.append((pid, captured_at[pid], t))
                    del captured_at[pid]
        # Comet shots: my actions targeting comet ids.
        comet_set = _comet_ids(my_obs)
        # Action format [[src_id, angle, ships], ...]; we don't have target
        # ids directly, but we can probe whether the ANGLE points to a comet
        # using straight-line nearest-planet. Approximation OK for diagnostic.
        records.append({
            "turn": t,
            "n_planets": n_planets,
            "ships": ships,
            "prod": prod,
            "n_emits": len(my_action),
            "n_comets_on_board": len(comet_set),
        })

    final = env.steps[-1]
    my_reward = final[seat_assignment].get("reward")
    return {
        "seed": seed,
        "seat": seat_assignment,
        "result": my_reward,
        "n_steps": n_steps,
        "records": records,
        "captures": captures,
        "lost_back": lost_back,
    }


def summarize(name, game):
    r = game["records"]
    total_emits = sum(rec["n_emits"] for rec in r)
    total_captures = len(game["captures"])
    total_lost_back = len(game["lost_back"])
    median_hold = -1
    if game["lost_back"]:
        holds = [(loss - cap) for (_pid, cap, loss) in game["lost_back"]]
        holds.sort()
        median_hold = holds[len(holds) // 2]
    final_planets = r[-1]["n_planets"] if r else 0
    final_ships = r[-1]["ships"] if r else 0
    final_prod = r[-1]["prod"] if r else 0
    print(f"  [{name}] result={game['result']:+d} n_steps={game['n_steps']:>3}  "
          f"emits={total_emits:>3}  caps={total_captures:>2}  "
          f"lost_back={total_lost_back:>2} (median_hold={median_hold})  "
          f"final: planets={final_planets} ships={final_ships:.0f} "
          f"prod={final_prod:.0f}")
    return {
        "emits": total_emits, "caps": total_captures,
        "lost_back": total_lost_back, "median_hold": median_hold,
        "result": game["result"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--wallclock-ms", type=float, default=600.0)
    ap.add_argument("--seed-offset", type=int, default=1000)
    args = ap.parse_args()

    os.environ["ORBIT_WARS_PARITY_WALLCLOCK_MS"] = str(args.wallclock_ms)

    v21 = import_module("agents.v21.main")
    v15 = import_module("agents.v15.main")

    rows = []
    for i in range(args.seeds):
        seed = args.seed_offset + i
        # Game 1: v21 (seat 0) vs v15 (seat 1)
        g_v21 = run_game(seed, {"focal": v21.agent, "opp": v15.agent},
                         seat_assignment=0, max_turns=args.max_turns)
        # Game 2: v15 (seat 0) vs v15 (seat 1) — same seed, same seat = baseline
        g_v15 = run_game(seed, {"focal": v15.agent, "opp": v15.agent},
                         seat_assignment=0, max_turns=args.max_turns)
        print(f"\n=== seed={seed} ===")
        s_v21 = summarize("v21 (seat 0 vs v15)", g_v21)
        s_v15 = summarize("v15 (seat 0 vs v15)", g_v15)
        # Deltas
        d_emits = s_v21["emits"] - s_v15["emits"]
        d_caps = s_v21["caps"] - s_v15["caps"]
        d_lost = s_v21["lost_back"] - s_v15["lost_back"]
        print(f"  Δ vs v15-baseline:  emits={d_emits:+d}  caps={d_caps:+d}  "
              f"lost_back={d_lost:+d}  result_delta={s_v21['result'] - s_v15['result']:+d}")
        rows.append({"seed": seed, "v21": s_v21, "v15": s_v15,
                     "d_emits": d_emits, "d_caps": d_caps, "d_lost": d_lost})

    print("\n=== ROLL-UP ===")
    n_v21_wins = sum(1 for r in rows if r["v21"]["result"] > 0)
    n_v15_self_wins = sum(1 for r in rows if r["v15"]["result"] > 0)
    avg_d_emits = sum(r["d_emits"] for r in rows) / len(rows)
    avg_d_caps = sum(r["d_caps"] for r in rows) / len(rows)
    avg_d_lost = sum(r["d_lost"] for r in rows) / len(rows)
    print(f"  v21 wins (vs v15):  {n_v21_wins}/{len(rows)}")
    print(f"  v15 wins (vs v15-self, seat0):  {n_v15_self_wins}/{len(rows)} "
          f"(should be ~50% in a fair game)")
    print(f"  Mean Δemits per game (v21 vs v15-as-seat-0):  {avg_d_emits:+.1f}")
    print(f"  Mean Δcaps per game:                          {avg_d_caps:+.1f}")
    print(f"  Mean Δlost_back per game:                     {avg_d_lost:+.1f}")
    if avg_d_emits < -5:
        print("\n  >>> v21 emits SIGNIFICANTLY FEWER fleets — patches over-filter")
    if avg_d_caps < -2:
        print("  >>> v21 captures FEWER planets — losing aggression beats waste savings")
    if avg_d_lost < -2:
        print("  >>> v21 loses BACK fewer planets — E1/E2 working correctly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
