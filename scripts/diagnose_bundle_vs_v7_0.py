"""One-game step-by-step diagnostic: bundle (P0) vs v7_0 (P1), seed 42.

Captures per turn: my/opp planet count, ship totals (garrison + in
flight), production rate, bundle's actions, v7_0's actions, key events
(captures, fleet arrivals, eliminations).

Output: stdout table + summary of inflection points.

Usage:
    python scripts/diagnose_bundle_vs_v7_0.py [--seed 42] [--print-every 1]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make

from agents.bundle.main import agent as bundle_agent


def load_v7_0():
    spec = importlib.util.spec_from_file_location(
        "v7_0_loaded",
        ROOT / "submissions" / "v7_0_drop_one.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec — bundled file uses @dataclass which needs
    # the module live in sys.modules during class construction.
    sys.modules["v7_0_loaded"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def planet_summary(obs, me: int):
    """(my_planets, opp_planets, neutral, my_ships, opp_ships,
       my_prod, opp_prod) — garrisons only, no in-flight."""
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    my_p = opp_p = neut = 0
    my_s = opp_s = 0.0
    my_prod = opp_prod = 0.0
    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        if owner == me:
            my_p += 1
            my_s += ships
            my_prod += prod
        elif owner == -1:
            neut += 1
        else:
            opp_p += 1
            opp_s += ships
            opp_prod += prod
    return my_p, opp_p, neut, my_s, opp_s, my_prod, opp_prod


def fleet_summary(obs, me: int):
    """(my_fleet_count, my_in_flight_ships, opp_fleet_count, opp_in_flight)."""
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    my_n = opp_n = 0
    my_s = opp_s = 0
    for f in fleets:
        owner = int(f[1])
        ships = int(f[6])
        if owner == me:
            my_n += 1
            my_s += ships
        else:
            opp_n += 1
            opp_s += ships
    return my_n, my_s, opp_n, opp_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=1,
                    help="print every N turns (1 = every turn)")
    args = ap.parse_args()

    v7_0_agent = load_v7_0()
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.reset(num_agents=2)

    print(f"=== bundle (P0) vs v7_0 (P1)  seed={args.seed} ===")
    print(f"{'t':>3}  "
          f"{'my-p':>4} {'opp-p':>5} {'neu':>3}  "
          f"{'my-g':>5} {'opp-g':>5}  "
          f"{'my-pr':>5} {'opp-pr':>6}  "
          f"{'my-f':>4}+{'shp':<4} {'opp-f':>5}+{'shp':<4}  "
          f"{'b-act':>5} {'v-act':>5}  "
          f"{'b-ms':>5} {'v-ms':>5}")

    bundle_ms_list = []
    v7_0_ms_list = []
    inflections: list[tuple[int, str]] = []
    prev_my_planets = None
    prev_opp_planets = None

    t = 0
    while True:
        s0 = env.state[0]
        s1 = env.state[1]
        if s0.status != "ACTIVE":
            break
        obs0 = s0.observation
        obs1 = s1.observation

        # Take per-side metrics from each player's POV obs.
        my_p, opp_p, neut, my_g, opp_g, my_pr, opp_pr = planet_summary(obs0, 0)
        my_f, my_fs, opp_f, opp_fs = fleet_summary(obs0, 0)

        # Inflection: planet-count change.
        if prev_my_planets is not None:
            if my_p < prev_my_planets:
                inflections.append((t, f"bundle LOST {prev_my_planets - my_p} planet(s) "
                                       f"({prev_my_planets}→{my_p})"))
            elif my_p > prev_my_planets:
                inflections.append((t, f"bundle GAINED {my_p - prev_my_planets} planet(s) "
                                       f"({prev_my_planets}→{my_p})"))
        if prev_opp_planets is not None:
            if opp_p < prev_opp_planets:
                inflections.append((t, f"v7_0 LOST {prev_opp_planets - opp_p} planet(s) "
                                       f"({prev_opp_planets}→{opp_p})"))
            elif opp_p > prev_opp_planets:
                inflections.append((t, f"v7_0 GAINED {opp_p - prev_opp_planets} planet(s) "
                                       f"({prev_opp_planets}→{opp_p})"))
        prev_my_planets = my_p
        prev_opp_planets = opp_p

        # Call agents (note: each runs its own World.from_obs).
        t0 = time.perf_counter()
        a0 = bundle_agent(obs0)
        b_ms = (time.perf_counter() - t0) * 1000
        bundle_ms_list.append(b_ms)
        t0 = time.perf_counter()
        a1 = v7_0_agent(obs1)
        v_ms = (time.perf_counter() - t0) * 1000
        v7_0_ms_list.append(v_ms)

        if t % args.print_every == 0:
            print(f"{t:>3}  "
                  f"{my_p:>4} {opp_p:>5} {neut:>3}  "
                  f"{my_g:>5.0f} {opp_g:>5.0f}  "
                  f"{my_pr:>5.1f} {opp_pr:>6.1f}  "
                  f"{my_f:>4}+{my_fs:<4} {opp_f:>5}+{opp_fs:<4}  "
                  f"{len(a0):>5} {len(a1):>5}  "
                  f"{b_ms:>5.0f} {v_ms:>5.0f}")

        env.step([a0, a1])
        t += 1
        if t > 500:
            break

    # Final state.
    final_status = env.state[0].status
    final_reward = env.state[0].reward
    print(f"\nfinal: status={final_status} reward(P0)={final_reward} n_steps={t}")

    # Timing summary.
    if bundle_ms_list:
        bs = sorted(bundle_ms_list)
        vs = sorted(v7_0_ms_list)
        print(f"\nbundle turn-ms  p50={bs[len(bs)//2]:.0f}  "
              f"p95={bs[int(len(bs)*0.95)]:.0f}  max={max(bs):.0f}")
        print(f"v7_0   turn-ms  p50={vs[len(vs)//2]:.0f}  "
              f"p95={vs[int(len(vs)*0.95)]:.0f}  max={max(vs):.0f}")

    # Inflection summary.
    if inflections:
        print(f"\ninflections ({len(inflections)}):")
        for tt, msg in inflections:
            print(f"  t={tt:>3}: {msg}")


if __name__ == "__main__":
    main()
