"""Quick diagnostic — play 1 game and print v8_scavenge's per-turn candidate
ledger + emitted action. Helps spot when the chooser sees no positive-Δ
candidate (hold) vs when it actually launches.

Usage:  python scripts/diag_v8.py <seed>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.v8_scavenge import main as v8_main
from lib.intent import World
from lib.world_model import WorldModel
from fast import _load_callable


def play_and_diag(seed: int, opp: str = "nearest", max_turns: int = 30, only_when_active: bool = True):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500})
    env.reset(num_agents=2)
    # Use v8_scavenge as P0; specified opp as P1
    opp_path = {
        "nearest": str(REPO / "agents/simple/nearest.py"),
        "v7_0": str(REPO / "submissions/v7_0_drop_one.py"),
        "v4_planner": str(REPO / "submissions/v4_planner.py"),
    }.get(opp, opp)
    opp_agent = _load_callable(opp_path)

    state = env.steps[0]
    print(f"== seed={seed} ==")
    for turn in range(max_turns):
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        # Capture v8's full candidate ledger
        obs_d = v8_main._as_dict(obs0)
        me = int(obs_d.get("player", 0))
        raw_planets = obs_d.get("planets", []) or []
        planets = [Planet(*p) for p in raw_planets]
        my_planets = [p for p in planets if int(p.owner) == me]
        targets = [p for p in planets if int(p.owner) != me]
        world = World.from_obs(obs_d)
        model = WorldModel.from_world(world)
        omega = float(obs_d.get("angular_velocity", 0.0))

        candidates = []
        for src in my_planets:
            if int(src.ships) < v8_main.MIN_FLEET_SIZE:
                continue
            for tgt in v8_main._nearest_k(targets, src, v8_main.NUM_TARGETS_PER_SOURCE):
                for ships in v8_main._enumerate_ship_counts_basic(src, tgt, model, omega):
                    if ships < v8_main.MIN_FLEET_SIZE or ships > int(src.ships):
                        continue
                    angle, eta = v8_main._aim_and_eta(src, tgt, ships, omega)
                    delta = v8_main._marginal_value(src, tgt, ships, eta, world, model, me)
                    candidates.append((delta, int(src.id), int(tgt.id), ships, eta))
        candidates.sort(key=lambda c: -c[0])
        positive = [c for c in candidates if c[0] > 0]
        # Counts
        my_total_ships = sum(int(p.ships) for p in my_planets)
        opp_planets = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]
        opp_total_ships = sum(int(p.ships) for p in opp_planets)

        a0 = v8_main.agent(obs0, env.configuration)
        try:
            a1 = opp_agent(obs1, env.configuration)
        except TypeError:
            a1 = opp_agent(obs1)

        active = bool(a0)
        if (not only_when_active) or active or turn < 5 or len(candidates) > 0:
            print(f"\n--turn {turn}--  ships me={my_total_ships} opp={opp_total_ships}"
                  f"  my_planets={len(my_planets)} opp_planets={len(opp_planets)}")
            if not candidates:
                print("  NO candidates enumerated (no positive-Δ found)")
            else:
                print(f"  total candidates: {len(candidates)}  positive: {len(positive)}")
                for delta, src_id, tgt_id, ships, eta in candidates[:8]:
                    star = "*" if delta > 0 else " "
                    print(f"    {star} src={src_id:>2}->tgt={tgt_id:>2} ships={ships:>3} eta={eta:>2} Δ={delta:.1f}")
            print(f"  emitted: {a0}")

        state = env.step([a0, a1])
        # Check if game ended
        s0 = state[0] if isinstance(state[0], dict) else state[0]
        status = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status != "ACTIVE":
            print(f"\n GAME OVER turn={turn} status={status}")
            break


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    opp = sys.argv[2] if len(sys.argv) > 2 else "nearest"
    max_t = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    play_and_diag(seed, opp=opp, max_turns=max_t, only_when_active=False)
