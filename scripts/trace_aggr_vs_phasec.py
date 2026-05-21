"""One-game trace: AGGR variant vs phase_c — per-turn diagnostic.

Usage:
    python scripts/trace_aggr_vs_phasec.py [--seed N] [--swap]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_agent(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def ships_by_owner(state):
    obs = state["observation"]
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    by_owner = {}
    plt_by_owner = {}
    for p in planets:
        o = int(p[1])
        if o < 0:
            continue
        by_owner[o] = by_owner.get(o, 0) + int(p[5])
        plt_by_owner.setdefault(o, []).append(int(p[0]))
    for f in fleets:
        o = int(f[1])
        if o < 0:
            continue
        by_owner[o] = by_owner.get(o, 0) + int(f[6])
    return by_owner, plt_by_owner


def trace(seed: int, swap: bool):
    from kaggle_environments import make
    aggr = load_agent(str(REPO / "submissions/baseline_joint_aggr.py"), "_aggr")
    pc = load_agent(str(REPO / "submissions/analytical_phase_c.py"), "_pc")

    if swap:
        agents = [pc, aggr]
        focal_idx = 1
        focal_name = "AGGR"
    else:
        agents = [aggr, pc]
        focal_idx = 0
        focal_name = "AGGR"
    opp_name = "phase_c"
    print(f"==== seed={seed}  P0={'AGGR' if not swap else 'phase_c'}  "
          f"P1={'phase_c' if not swap else 'AGGR'} ====")

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)

    # Drive turn-by-turn via env.step so we can intercept actions.
    env.reset(num_agents=2)
    # Print initial board summary.
    initial = env.steps[0][0]["observation"]
    planets_init = initial.get("planets", [])
    print(f"\nINITIAL: {len(planets_init)} planets")
    for o in (0, 1):
        owned = [p for p in planets_init if int(p[1]) == o]
        total_ships = sum(int(p[5]) for p in owned)
        total_prod = sum(int(p[6]) for p in owned)
        who = "AGGR" if (o == focal_idx) else "phase_c"
        print(f"  P{o} ({who}): {len(owned)} planets, {total_ships} ships, "
              f"{total_prod}/turn prod")

    # Use env.run with the agent callables but instrument via steps log
    # afterwards. Simpler and reproducible.
    env.run(agents)

    last_step = len(env.steps) - 1
    print(f"\n=== ran {last_step+1} turns ===")

    # Per-turn summary every 25 turns; full at end.
    aggr_emit_total = 0
    pc_emit_total = 0
    aggr_emit_ships = 0
    pc_emit_ships = 0
    multi_src_same_tgt_aggr = 0
    own_changes = []  # (turn, planet_id, from_owner, to_owner)
    prev_owners = {int(p[0]): int(p[1]) for p in planets_init}

    for t, step in enumerate(env.steps):
        if t == 0:
            continue
        # Each step is a list of agent states; action from prior turn is in
        # state["action"].
        for pid, st in enumerate(step):
            act = st.get("action")
            if not isinstance(act, list):
                continue
            emit_count = len(act)
            ship_total = sum(int(m[2]) for m in act if isinstance(m, list) and len(m) >= 3)
            if pid == focal_idx:
                aggr_emit_total += emit_count
                aggr_emit_ships += ship_total
                # Detect multi-source-same-target (AGGR feature).
                tgts_this_turn = []
                # Note: action doesn't carry target_id, only direction; can't
                # easily detect target. Skip.
            else:
                pc_emit_total += emit_count
                pc_emit_ships += ship_total

        # Owner changes this turn (use any agent's planet view; all agents see
        # same world).
        obs = step[0]["observation"]
        for p in obs.get("planets", []):
            pid = int(p[0])
            owner = int(p[1])
            prev = prev_owners.get(pid)
            if prev is not None and prev != owner:
                own_changes.append((t, pid, prev, owner))
            prev_owners[pid] = owner

    print(f"\nAGGR  total launches: {aggr_emit_total},  ships sent: {aggr_emit_ships}")
    print(f"phase_c total launches: {pc_emit_total},  ships sent: {pc_emit_ships}")
    print(f"\nOwnership changes: {len(own_changes)} total")
    # Tabulate captures by direction.
    aggr_caps = [c for c in own_changes if c[3] == focal_idx and c[2] != focal_idx]
    pc_caps = [c for c in own_changes if c[3] != focal_idx and c[2] == focal_idx]
    neutrals_taken_by_aggr = [c for c in own_changes if c[2] == -1 and c[3] == focal_idx]
    neutrals_taken_by_pc = [c for c in own_changes if c[2] == -1 and c[3] != focal_idx]
    print(f"  AGGR captures from phase_c: {len(aggr_caps)}")
    print(f"  phase_c captures from AGGR: {len(pc_caps)}")
    print(f"  AGGR took from neutral:     {len(neutrals_taken_by_aggr)}")
    print(f"  phase_c took from neutral:  {len(neutrals_taken_by_pc)}")

    final = env.steps[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    print(f"\nFINAL  P0 reward={r0}  P1 reward={r1}  "
          f"winner=P{0 if r0>r1 else (1 if r1>r0 else '?')}  "
          f"focal={'WIN' if final[focal_idx]['reward']>final[1-focal_idx]['reward'] else 'LOSS'}")
    by, _ = ships_by_owner(final[0])
    for o in sorted(by):
        who = "AGGR" if o == focal_idx else "phase_c"
        print(f"  P{o} ({who}) final ship total: {by[o]}")

    return own_changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args()
    trace(args.seed, args.swap)


if __name__ == "__main__":
    main()
