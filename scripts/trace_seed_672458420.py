"""One-game trace: agents/baseline/main.py (BASELINE_VALUE_HEAD=strategic +
orbitfix env stack) vs submissions/baseline_joint_aggr_consolidated_orbitfix.py
on seed 672458420.

Captures per-turn focal launches and per-planet ownership over time so we
can answer PI's question: does favor_strategic capture the big-production
neutrals early instead of nibbling small ones?

Output:
  - per-turn launch log (turn, src, tgt, ships, src_prod, tgt_prod)
  - turn-0 board summary (production sorted desc) so we know which targets
    were "big-prod neutrals"
  - per-turn aggregate (n_owned, total_ships, big_prod_owned_count)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Orbitfix env stack PLUS strategic head.
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

from kaggle_environments import make
from fast import _load_callable

SEED = 672458420
EPISODE_STEPS = 250  # match PI standard procedure


def main():
    env = make(
        "orbit_wars",
        configuration={"seed": SEED, "episodeSteps": EPISODE_STEPS},
        debug=False,
    )
    env.reset(num_agents=2)

    from agents.baseline.main import agent as focal_agent
    opp = _load_callable(
        "submissions/baseline_joint_aggr_consolidated_orbitfix.py",
    )

    state = env.steps[0]

    # Turn-0 board snapshot for planet labelling.
    obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
    obs0_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    planets0 = obs0_d.get("planets") or []
    # Planet tuple: (id, owner, x, y, radius, ships, production)
    planet_prod = {int(p[0]): int(p[6]) for p in planets0}
    planet_owner0 = {int(p[0]): int(p[1]) for p in planets0}
    planet_ships0 = {int(p[0]): int(p[5]) for p in planets0}

    big_prod_threshold = sorted(planet_prod.values(), reverse=True)[5]
    big_prod_ids = {pid for pid, p in planet_prod.items() if p >= big_prod_threshold}

    print(f"=== seed {SEED}  episodeSteps={EPISODE_STEPS}  2P focal=strategic vs opp=orbitfix ===")
    print(f"top-prod planets at step 0 (id, prod, ships, owner):")
    for pid in sorted(planet_prod, key=lambda p: -planet_prod[p])[:10]:
        ownr = planet_owner0[pid]
        ownr_lbl = {0: "ME", 1: "OPP", -1: "neutral"}.get(ownr, f"p{ownr}")
        print(f"  P{pid:>2d}  prod={planet_prod[pid]:>2d}  ships={planet_ships0[pid]:>3d}  owner={ownr_lbl}")

    print(f"\nbig-prod set (top 6, threshold prod>={big_prod_threshold}): {sorted(big_prod_ids)}")
    print(f"\nturn | src(p,prod,ships) -> tgt(p,prod,ships,owner) ships  | my_planets total_ships big_prod_owned")
    print(f"-----+--------------------------------------------------------+----------------------------------")

    n_steps = 0
    me = 0
    while True:
        obs_me = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs_op = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation

        try:
            a_me = focal_agent(obs_me, env.configuration)
        except TypeError:
            a_me = focal_agent(obs_me)
        try:
            a_op = opp(obs_op, env.configuration)
        except TypeError:
            a_op = opp(obs_op)

        # Pre-step: capture board state for the focal.
        obs_me_d = obs_me if isinstance(obs_me, dict) else dict(obs_me)
        planets = obs_me_d.get("planets") or []
        planet_by_id = {int(p[0]): p for p in planets}
        owner_now = {int(p[0]): int(p[1]) for p in planets}
        my_planet_ids = {pid for pid, o in owner_now.items() if o == me}
        big_prod_owned = len(my_planet_ids & big_prod_ids)
        total_my_ships = sum(int(p[5]) for p in planets if int(p[1]) == me)

        if a_me:
            for action in a_me:
                src_id = int(action[0])
                ships = int(action[2])
                # Best-effort target inference: nearest planet to fleet
                # trajectory is tricky. We have angle; use it as a rough proxy.
                # Just log src and ships; agent doesn't return target id.
                src = planet_by_id.get(src_id)
                src_ships = int(src[5]) if src else -1
                src_prod = int(src[6]) if src else -1
                # Target from angle: we'd need the same aim logic. Skip — log
                # ships fired + src; the trace gives enough to verify whether
                # we're firing from sources at appropriate targets.
                tgt_str = f"angle={float(action[1]):.2f}"
                print(f"{n_steps:>4d} | P{src_id:>2d} prod={src_prod} ships={src_ships:>3d}  fire {ships:>3d}  {tgt_str:>18s} | mp={len(my_planet_ids):>2d} tot={total_my_ships:>4d} bp={big_prod_owned}")
        else:
            if n_steps % 10 == 0:
                print(f"{n_steps:>4d} | (idle) | mp={len(my_planet_ids):>2d} tot={total_my_ships:>4d} bp={big_prod_owned}")

        state = env.step([a_me, a_op])
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE" or n_steps >= EPISODE_STEPS:
            break

    # Final outcome
    final = env.steps[-1]
    r0 = final[0]["reward"]
    r1 = final[1]["reward"]
    outcome = "P0_WIN" if r0 > r1 else ("P1_WIN" if r1 > r0 else "DRAW")
    print(f"\n=== final: outcome={outcome}  steps={n_steps}  P0={r0}  P1={r1} ===")

    # Final board
    obs_f = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    obs_f_d = obs_f if isinstance(obs_f, dict) else dict(obs_f)
    planets_f = obs_f_d.get("planets") or []
    my_final = sum(1 for p in planets_f if int(p[1]) == 0)
    opp_final = sum(1 for p in planets_f if int(p[1]) == 1)
    big_owned_final = sum(1 for p in planets_f if int(p[1]) == 0 and int(p[0]) in big_prod_ids)
    print(f"  final my_planets={my_final}  opp_planets={opp_final}  big_prod_owned={big_owned_final}/{len(big_prod_ids)}")


if __name__ == "__main__":
    main()
