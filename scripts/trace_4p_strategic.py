"""Single 4P FFA game trace: agents/baseline/main.py (strategic head +
Phase F fixes + post-4P-regression bug-fix) vs 3× orbitfix bundle.

Output:
  - turn-0 board: planet inventory + ownership (incl. seat assignments)
  - per-turn focal launches with src + ships + target-angle
  - per-turn aggregate: my_planets, opp1_planets, opp2_planets, opp3_planets,
    my_total_ships, idle_ships
  - final outcome + final planet counts
"""
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Strategic head + Phase F + bug-fix config.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("BASELINE_VALUE_HEAD", "strategic")
os.environ.setdefault("BASELINE_HOLD_HORIZON", "20")
os.environ.setdefault("BASELINE_FORWARD_REACH_WEIGHT", "0.5")
os.environ.setdefault("BASELINE_FORWARD_REACH_HORIZON", "15")
os.environ.setdefault("BASELINE_FINISH_BONUS", "50")
os.environ.setdefault("BASELINE_FINISH_THRESHOLD", "200")

from kaggle_environments import make
from fast import _load_callable

SEED = int(os.environ.get("TRACE_4P_SEED", "0"))
EPISODE_STEPS = 250
OPP_BUNDLE = "submissions/baseline_joint_aggr_consolidated_orbitfix.py"


def _wants_config(fn):
    try:
        return len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        return True


def main():
    env = make(
        "orbit_wars",
        configuration={"seed": SEED, "episodeSteps": EPISODE_STEPS},
        debug=False,
    )
    env.reset(num_agents=4)

    from agents.baseline.main import agent as focal_agent
    opp = _load_callable(OPP_BUNDLE)

    agents_list = [focal_agent, opp, opp, opp]
    wants_config = [_wants_config(a) for a in agents_list]

    state = env.steps[0]
    obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
    obs0_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    planets0 = obs0_d.get("planets") or []
    print(f"=== 4P FFA trace  seed={SEED}  episode_steps={EPISODE_STEPS} ===")
    print(f"focal=strategic+TermABC+postF-fix  opps=3× orbitfix bundle")
    print(f"turn-0 planet inventory:")
    seat_counts = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
    for p in planets0:
        seat_counts[int(p[1])] = seat_counts.get(int(p[1]), 0) + 1
    print(f"  total planets: {len(planets0)}  ME={seat_counts.get(0, 0)}  "
          f"OPP1={seat_counts.get(1, 0)}  OPP2={seat_counts.get(2, 0)}  "
          f"OPP3={seat_counts.get(3, 0)}  neutral={seat_counts.get(-1, 0)}")

    print(f"\nturn | src(p,prod,ships) fire ships  | planets[me/o1/o2/o3]  ships[me/o1/o2/o3]  idle")
    print(f"-----+------------------------------+--------------------------------------------")

    n_steps = 0
    me = 0
    while True:
        obs_me = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        per_seat_actions = []
        for s_idx in range(4):
            obs_s = state[s_idx]["observation"] if isinstance(state[s_idx], dict) else state[s_idx].observation
            a = (agents_list[s_idx](obs_s, env.configuration)
                 if wants_config[s_idx] else agents_list[s_idx](obs_s))
            per_seat_actions.append(a)

        obs_me_d = obs_me if isinstance(obs_me, dict) else dict(obs_me)
        planets = obs_me_d.get("planets") or []
        planet_by_id = {int(p[0]): p for p in planets}
        owner_now = {int(p[0]): int(p[1]) for p in planets}
        # Counts per seat
        c_me = sum(1 for o in owner_now.values() if o == 0)
        c_o1 = sum(1 for o in owner_now.values() if o == 1)
        c_o2 = sum(1 for o in owner_now.values() if o == 2)
        c_o3 = sum(1 for o in owner_now.values() if o == 3)
        s_me = sum(int(p[5]) for p in planets if int(p[1]) == 0)
        s_o1 = sum(int(p[5]) for p in planets if int(p[1]) == 1)
        s_o2 = sum(int(p[5]) for p in planets if int(p[1]) == 2)
        s_o3 = sum(int(p[5]) for p in planets if int(p[1]) == 3)

        a_me = per_seat_actions[0]
        total_launched = 0
        if a_me:
            for action in a_me:
                src_id = int(action[0])
                ships = int(action[2])
                total_launched += ships
                src = planet_by_id.get(src_id)
                src_prod = int(src[6]) if src else -1
                src_ships = int(src[5]) if src else -1
                print(f"{n_steps:>4d} | P{src_id:>2d} prod={src_prod} ships={src_ships:>3d}  fire {ships:>3d}  | "
                      f"{c_me:>2d}/{c_o1:>2d}/{c_o2:>2d}/{c_o3:>2d}  "
                      f"{s_me:>4d}/{s_o1:>4d}/{s_o2:>4d}/{s_o3:>4d}  idle={s_me - total_launched:>4d}")
        else:
            if n_steps % 10 == 0:
                print(f"{n_steps:>4d} | (idle)                       | "
                      f"{c_me:>2d}/{c_o1:>2d}/{c_o2:>2d}/{c_o3:>2d}  "
                      f"{s_me:>4d}/{s_o1:>4d}/{s_o2:>4d}/{s_o3:>4d}")

        state = env.step(per_seat_actions)
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE" or n_steps >= EPISODE_STEPS:
            break

    final = env.steps[-1]
    rs = [s["reward"] for s in final]
    focal_won = rs[0] >= max(rs[1:])
    print(f"\n=== final: steps={n_steps}  rs={rs}  focal_won={focal_won} ===")
    obs_f = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    obs_f_d = obs_f if isinstance(obs_f, dict) else dict(obs_f)
    planets_f = obs_f_d.get("planets") or []
    print(f"  final ownership: me={sum(1 for p in planets_f if int(p[1])==0)} "
          f"o1={sum(1 for p in planets_f if int(p[1])==1)} "
          f"o2={sum(1 for p in planets_f if int(p[1])==2)} "
          f"o3={sum(1 for p in planets_f if int(p[1])==3)} "
          f"neutral={sum(1 for p in planets_f if int(p[1])==-1)}")


if __name__ == "__main__":
    main()
