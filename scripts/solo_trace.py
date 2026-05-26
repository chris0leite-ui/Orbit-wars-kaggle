"""Solo turn-by-turn trace: ONE game, focal vs noop, print every emission.

Use to inspect how a new value head actually plays — what it captures,
when it fires, how the ship pile grows.

Example:
    python scripts/solo_trace.py \\
        --agent agents/baseline_integral \\
        --seed 0 --steps 250 --integral-t-end 250

Output per step (one line):
    turn | (src=Pxx prod=N ships=N -> tgt=Pxx)+ | planets[me/opp/N]  ships[me/opp+flight]  prod[me]  V

`V` is the integral leaf value (computed inline from the agent's obs at
that step), so we can see how the leaf's prediction evolves.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("focal_agent_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def _resolve(spec: str) -> Path:
    candidates = [Path(spec), REPO / spec, REPO / f"{spec}.py",
                  REPO / "agents" / spec, REPO / "agents" / f"{spec}.py",
                  REPO / "agents" / "simple" / f"{spec}.py"]
    for p in candidates:
        if p.is_dir():
            mp = p / "main.py"
            if mp.is_file():
                return mp.resolve()
        elif p.is_file():
            return p.resolve()
    raise SystemExit(f"cannot resolve agent: {spec}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="Focal agent spec.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--integral-t-end", type=int, default=None,
                    help="Override INTEGRAL_T_END before agent import.")
    ap.add_argument("--every", type=int, default=1,
                    help="Print idle turns every N steps (emissions always printed).")
    args = ap.parse_args()

    import os
    if args.integral_t_end is not None:
        os.environ["INTEGRAL_T_END"] = str(args.integral_t_end)

    focal_path = _resolve(args.agent)
    noop_path = REPO / "agents" / "simple" / "noop.py"
    focal = _load_agent(focal_path)
    noop = _load_agent(noop_path)

    # Inline integral leaf so we can score V at every step without importing
    # value.py (which is owned by the agent module).
    def integral_V(obs_dict, me: int, t_end: int) -> float:
        planets = obs_dict.get("planets") or []
        fleets = obs_dict.get("fleets") or []
        step = int(obs_dict.get("step", 0))
        remaining = max(0, t_end - step)
        ships: dict[int, float] = {}
        prod: dict[int, float] = {}
        for p in planets:
            o = int(p[1])
            if o < 0:
                continue
            ships[o] = ships.get(o, 0.0) + float(p[5])
            prod[o] = prod.get(o, 0.0) + float(p[6])
        for f in fleets:
            o = int(f[1])
            if o < 0:
                continue
            ships[o] = ships.get(o, 0.0) + float(f[6])

        def tot(o: int) -> float:
            return ships.get(o, 0.0) + prod.get(o, 0.0) * remaining

        opps = [o for o in set(ships) | set(prod) if o != me and o >= 0]
        return tot(me) - max((tot(o) for o in opps), default=0.0)

    from kaggle_environments import make
    env = make("orbit_wars",
               configuration={"seed": args.seed, "episodeSteps": args.steps},
               debug=False)
    env.reset(num_agents=2)

    state = env.steps[0]
    obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
    obs0_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    planets0 = obs0_d.get("planets") or []
    me_planet0 = [p for p in planets0 if int(p[1]) == 0]
    opp_planet0 = [p for p in planets0 if int(p[1]) == 1]

    t_end_used = int(os.environ.get("INTEGRAL_T_END", str(args.steps)))
    print(f"=== solo trace  agent={args.agent}  seed={args.seed}  "
          f"steps={args.steps}  T_END={t_end_used} ===")
    print(f"turn-0:  total_planets={len(planets0)}  "
          f"me={len(me_planet0)} ({sum(int(p[5]) for p in me_planet0)} ships, "
          f"prod={sum(int(p[6]) for p in me_planet0)})  "
          f"opp={len(opp_planet0)} ({sum(int(p[5]) for p in opp_planet0)} ships)  "
          f"neutral={sum(1 for p in planets0 if int(p[1]) == -1)}")
    print(f"\n  t  | actions                                              "
          f"| pl[me/op/N] | ships[me+fl/op] | prod | V_leaf")
    print(f"-----+------------------------------------------------------"
          f"+-------------+-----------------+------+--------")

    n_steps = 0
    while True:
        obs_me = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs_op = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        obs_me_d = obs_me if isinstance(obs_me, dict) else dict(obs_me)

        a_me = focal(obs_me)
        a_op = noop(obs_op)

        planets = obs_me_d.get("planets") or []
        fleets = obs_me_d.get("fleets") or []
        planet_by_id = {int(p[0]): p for p in planets}

        c_me = sum(1 for p in planets if int(p[1]) == 0)
        c_op = sum(1 for p in planets if int(p[1]) == 1)
        c_n  = sum(1 for p in planets if int(p[1]) == -1)
        s_me_pl = sum(int(p[5]) for p in planets if int(p[1]) == 0)
        s_me_fl = sum(int(f[6]) for f in fleets if int(f[1]) == 0)
        s_op    = sum(int(p[5]) for p in planets if int(p[1]) == 1) + \
                  sum(int(f[6]) for f in fleets if int(f[1]) == 1)
        prod_me = sum(int(p[6]) for p in planets if int(p[1]) == 0)
        V = integral_V(obs_me_d, me=0, t_end=t_end_used)

        if a_me:
            # Compose a compact actions string. Each action = (src, tgt, ships).
            parts = []
            for act in a_me:
                src_id = int(act[0]); tgt_id = int(act[1]); ships = int(act[2])
                src = planet_by_id.get(src_id)
                src_prod = int(src[6]) if src else -1
                parts.append(f"P{src_id}(pr{src_prod})->P{tgt_id}:{ships}")
            actions_str = " ".join(parts)[:54]
            print(f"{n_steps:>4d} | {actions_str:<54s} | "
                  f"{c_me:>2d}/{c_op:>2d}/{c_n:>2d}  | "
                  f"{s_me_pl:>4d}+{s_me_fl:>3d}/{s_op:>4d} | "
                  f"{prod_me:>4d} | {V:>7.1f}")
        elif n_steps % args.every == 0:
            print(f"{n_steps:>4d} | (idle)                                                "
                  f"| {c_me:>2d}/{c_op:>2d}/{c_n:>2d}  | "
                  f"{s_me_pl:>4d}+{s_me_fl:>3d}/{s_op:>4d} | "
                  f"{prod_me:>4d} | {V:>7.1f}")

        state = env.step([a_me, a_op])
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE" or n_steps >= args.steps:
            break

    final = env.steps[-1]
    obs_f = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    obs_f_d = obs_f if isinstance(obs_f, dict) else dict(obs_f)
    planets_f = obs_f_d.get("planets") or []
    fleets_f = obs_f_d.get("fleets") or []
    me_final_pl = sum(int(p[5]) for p in planets_f if int(p[1]) == 0)
    me_final_fl = sum(int(f[6]) for f in fleets_f if int(f[1]) == 0)
    me_planets_final = sum(1 for p in planets_f if int(p[1]) == 0)
    op_total = sum(int(p[5]) for p in planets_f if int(p[1]) == 1) + \
               sum(int(f[6]) for f in fleets_f if int(f[1]) == 1)
    op_planets_final = sum(1 for p in planets_f if int(p[1]) == 1)
    print(f"\n=== final step={n_steps}  me={me_final_pl}+{me_final_fl}={me_final_pl + me_final_fl} ships on "
          f"{me_planets_final} planets (prod={sum(int(p[6]) for p in planets_f if int(p[1]) == 0)})  "
          f"opp={op_total} on {op_planets_final}  neutral={sum(1 for p in planets_f if int(p[1]) == -1)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
