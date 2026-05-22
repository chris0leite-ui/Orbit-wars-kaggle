"""Deep diagnostic on a single early-game turn: dump all bundles coord
sees with their wait_N, ships, leaf-Δ, endgame_bonus, composite,
and what emit drops.

Also runs minimal on the same observation and prints what it would
emit, for direct comparison.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("COORD_DELTA_W", "1")

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet  # noqa: E402

from agents.coord.main import (  # noqa: E402
    CHEAP_FILTER_TOP_K,
    WALLCLOCK_BUDGET_MS,
    BundleKind,
    _as_dict,
    _num_seats,
    _greedy_primal,
    _composite_score,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    lagrangian_clear,
    tier2_score_bundles,
)
from agents.minimal.main import agent as minimal_agent  # noqa: E402
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def dump(seed: int = 0, turn: int = 0):
    """Drive a game to `turn`, then dump everything coord and minimal see."""
    from agents.coord.main import agent as coord_agent
    env = make("orbit_wars", configuration={"seed": int(seed)})
    env.reset(num_agents=2)
    # Advance to the target turn by stepping with coord vs minimal.
    for t in range(turn):
        if env.done:
            print(f"game ended early at turn {t}")
            return
        a0 = coord_agent(env.state[0].observation)
        a1 = minimal_agent(env.state[1].observation)
        env.step([a0, a1])
    # Use one fresh step with the simplest possible agents.
    obs = env.state[0].observation
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets:
        print("no own planets")
        return
    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    print(f"=== seed {seed} turn {turn}: me={me} ===")
    print(f"  my_planets: {len(my_planets)}")
    for p in my_planets:
        print(f"    p{p.id} ships={p.ships} prod={p.production}")
    print(f"  enemy planets: {len(other_planets)}")
    print(f"  in_flight: {len(raw_fleets)}")
    print()

    # Enumerate (no deadline; we want the full set).
    attacks = enumerate_attack_bundles(
        my_planets, other_planets, world, model, me, omega,
    )
    defends = enumerate_defend_bundles(
        my_planets, world, model, me, omega,
    )
    all_bundles = attacks + defends
    print(f"  raw enumerate: attack={len(attacks)} defend={len(defends)}")

    cheap = cheap_filter_bundles(
        all_bundles, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
    )
    print(f"  after cheap-filter: {len(cheap)} bundles (K={CHEAP_FILTER_TOP_K})")

    scored = tier2_score_bundles(
        cheap, snap_base, me, num_seats, world, model,
    )
    print(f"  after Tier-2: {len(scored)} bundles\n")

    # Sort by composite descending.
    scored_sorted = sorted(scored, key=lambda b: -_composite_score(b))

    print("  ALL TIER-2 SCORED BUNDLES (composite-desc):")
    print(f"  {'#':>3} {'kind':>7} {'tgt':>4} {'wait':>5} {'eta':>4}"
          f" {'ships':>6} {'leaf':>9} {'bonus':>9} {'composite':>10}"
          f" {'fire-now':>8}")
    for i, b in enumerate(scored_sorted):
        # Single-leg only in early game; combine multi-leg display.
        for L in b.legs:
            wait_s = "0" if L.wait_N == 0 else f"+{L.wait_N}"
            fire_now = "yes" if L.wait_N == 0 else "no"
            print(f"  {i+1:>3} {b.kind.name:>7} {b.target_id:>4} "
                  f"{wait_s:>5} {L.eta:>4} {L.ships:>6} "
                  f"{b.tier2_score:>9.2f} {b.endgame_bonus:>9.2f} "
                  f"{_composite_score(b):>10.2f} {fire_now:>8}")
        # If multi-leg, leave a blank line between bundles.
        if len(b.legs) > 1:
            print()

    print()
    # Run Lagrangian to see what it picks.
    selected = lagrangian_clear(scored, my_planets=my_planets)
    print(f"  Lagrangian selected: {len(selected)} bundles")
    for b in selected:
        for L in b.legs:
            wait_s = "0" if L.wait_N == 0 else f"+{L.wait_N}"
            print(f"    tgt={b.target_id} {b.kind.name} src={L.src_id} "
                  f"ships={L.ships} wait={wait_s} eta={L.eta} "
                  f"composite={_composite_score(b):.2f}")

    # What would emit do?
    from agents.coord.main import emit_bundle_actions
    moves = emit_bundle_actions(selected, world, model, me)
    print(f"  emit produced: {len(moves)} moves")
    for m in moves:
        print(f"    move: src={m[0]} angle={m[1]:.3f} ships={m[2]}")

    # What does minimal do on the same obs?
    print()
    print("  --- minimal on same obs ---")
    minimal_moves = minimal_agent(obs)
    print(f"  minimal moves: {len(minimal_moves)}")
    for m in minimal_moves:
        print(f"    move: src={m[0]} angle={m[1]:.3f} ships={m[2]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turn", type=int, default=0)
    args = parser.parse_args()
    dump(seed=args.seed, turn=args.turn)
