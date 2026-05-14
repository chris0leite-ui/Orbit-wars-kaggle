"""diagnose_chooser.py — dump the chooser's scoring at a specific turn.

Runs a game up to a target turn, then runs main.agent() with verbose
instrumentation: lists every candidate (src, tgt, ships), prints its
analytic Δfavor (score_action), its rollout-leaf favor, and why it
was kept or rejected.

Used to find which formula term causes the under-launching we
observed in turns 0-30 vs v7_0 (5 favor-axis variants, all 0 / 96).

Usage:  python diagnose_chooser.py [seed] [target_turn]
"""
from __future__ import annotations

import math
import sys
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

import main
from favor import favor, favor_breakdown


def _capture_obs_at_turn(seed, target_turn, my_slot=0):
    """Run a game with my agent vs v7_0; return obs of `my_slot` at
    target_turn (BEFORE the action for that turn was decided)."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = ["baselines/v7_0.py", "baselines/v7_0.py"]
    agents[my_slot] = "main.py"
    env.run(agents)
    if target_turn >= len(env.steps):
        target_turn = len(env.steps) - 1
    return env.steps[target_turn][my_slot].observation, target_turn


def dump_chooser(seed, target_turn, my_slot=0):
    obs, t = _capture_obs_at_turn(seed, target_turn, my_slot)
    player = my_slot
    print(f"=== seed={seed}  turn={t}  player={player} ===\n")

    raw_planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    raw_fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    print(f"my_planets ({len(my_planets)}):")
    for p in my_planets:
        print(f"  P{p.id} owner={p.owner} pos=({p.x:5.1f},{p.y:5.1f}) "
              f"r={p.radius:.1f} ships={p.ships} prod={p.production}")
    print(f"opp/neutral ({len(targets)}):")
    for p in targets[:12]:
        print(f"  P{p.id} owner={p.owner} pos=({p.x:5.1f},{p.y:5.1f}) "
              f"r={p.radius:.1f} ships={p.ships} prod={p.production}")
    if len(targets) > 12:
        print(f"  ... ({len(targets) - 12} more)")

    bd = favor_breakdown(obs, player)
    print(f"\nfavor breakdown: {bd}")

    print("\n--- v4 fast_sim Δfavor candidates (> 0) ---")
    favor_before = favor(obs, player)
    num_seats = main._num_seats(planets, fleets)
    snap_base = main.from_obs(obs, num_seats=num_seats)
    baseline_favors = main._build_idle_baseline(snap_base, player, num_seats, 40)
    cands = main._enumerate_candidates(
        my_planets, targets, fleets, t, player,
        snap_base, num_seats, baseline_favors
    )
    print(f"   {len(cands)} candidates would be LAUNCHED")
    for i, (score, src, tgt, ships) in enumerate(cands[:15]):
        print(f"   #{i:2d}  P{src.id:2d}→P{tgt.id:2d}  ×{ships:3d}  "
              f"Δfavor = {score:+10.1f}  "
              f"(tgt: owner={tgt.owner} ships={tgt.ships} prod={tgt.production})")

    if not cands:
        print("\n   NO CANDIDATES. Per-pair Δfavor for diagnosis:")
        for src in my_planets:
            if src.ships < main.MIN_FLEET_SIZE:
                continue
            ranked = sorted(targets, key=lambda t: math.hypot(src.x - t.x, src.y - t.y))[:main.NUM_TARGETS_PER_SOURCE]
            for tgt in ranked:
                cap = main._capture_size_guess(src, tgt)
                if cap < main.MIN_FLEET_SIZE:
                    cap = main.MIN_FLEET_SIZE
                s = main.score_action(src, tgt, cap, t, player,
                                       snap_base=snap_base, num_seats=num_seats,
                                       baseline_favors=baseline_favors)
                print(f"     P{src.id:2d}→P{tgt.id:2d}  cap={cap:3d} (src.ships={src.ships}) "
                      f"  Δfavor = {s:+10.1f}  "
                      f"(tgt: owner={tgt.owner} ships={tgt.ships} prod={tgt.production})")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1003
    target_turn = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    dump_chooser(seed, target_turn)
