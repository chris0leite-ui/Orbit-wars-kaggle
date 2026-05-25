"""One-off probe: at seed 1622482326, what does the BUILDUP MILP plan?

Builds the initial world (step 0) and dumps:
- Planet inventory (id, owner, garrison, production, position)
- The 19-garrison neutrals specifically (the ones in the screenshot)
- The MILP pruning waterfall
- The final schedule (which targets it chose)
- For each 19-garrison neutral that did NOT make the schedule, why
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make

from lib.intent import World
from lib.world_model import WorldModel
from lib.joint_solver.opening_planner import (
    opening_plan, _Candidate, _build_candidates, _dist,
    TOP_TARGETS_PER_SOURCE, MIN_SOURCE_SHIPS, OPENING_HORIZON,
    OPP_RESPONSE_LAG, OPENING_DEFENDER_GUARD,
)
from lib.fleet import speed as fleet_speed
from agents.baseline.proposer import aim_and_eta
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


SEED = 1622482326


def main() -> None:
    env = make("orbit_wars", debug=False, configuration={"episodeSteps": 250})
    env.reset(num_agents=2)
    env.specification["seed"] = SEED
    # Reset with seed
    env.reset(num_agents=2)
    # Find an obs for player 0
    obs0 = env.state[0]["observation"]

    # Patch the seed by reconstructing
    env = make("orbit_wars", configuration={"episodeSteps": 250, "seed": SEED})
    env.reset(num_agents=2)
    obs0 = dict(env.state[0]["observation"])
    obs0["player"] = 0
    # ensure step int
    obs0["step"] = int(obs0.get("step", 0))

    world = World.from_obs(obs0)
    model = WorldModel.from_world(world)

    print(f"=== SEED {SEED}, step {world.step}, my_id={world.my_id} ===")
    print(f"omega = {world.omega:.4f}")

    # planet inventory
    print("\nPlanets (sorted by id):")
    print(f"  {'id':>3}  {'own':>3}  {'gar':>4}  {'prod':>4}   x       y")
    my_planets = []
    enemy_planets = []
    neutrals = []
    for p in sorted(world.planets_by_id.values(), key=lambda q: q.id):
        owner = int(p.owner)
        tag = "ME" if owner == world.my_id else ("OPP" if owner >= 0 else "NEU")
        print(f"  {p.id:>3}  {owner:>3}  {int(p.ships):>4}  {int(p.production):>4}   "
              f"{float(p.x):+6.1f}  {float(p.y):+6.1f}   {tag}")
        if owner == world.my_id:
            my_planets.append(p)
        elif owner >= 0:
            enemy_planets.append(p)
        else:
            neutrals.append(p)

    print(f"\nme={len(my_planets)} planets, opp={len(enemy_planets)}, neutrals={len(neutrals)}")

    # find the 19-garrison neutrals
    nineteens = [p for p in neutrals if int(p.ships) == 19]
    print(f"\n19-garrison neutrals: {[p.id for p in nineteens]}")
    for p in nineteens:
        print(f"  id={p.id} ships={int(p.ships)} prod={int(p.production)} "
              f"pos=({float(p.x):+.1f},{float(p.y):+.1f})")

    # MILP plan
    print("\n--- opening_plan() ---")
    op = opening_plan(world, model, world.my_id, num_seats=2)
    print(f"objective = {op.objective:.3f}")
    print(f"n_vars = {op.n_vars}, n_constraints = {op.n_constraints}")
    print(f"status = {op.status}")
    print("\nPruning waterfall:")
    for k, v in op.pruning_waterfall.items():
        print(f"  {k:35s} {v}")

    print(f"\nScheduled launches ({len(op.schedule)}):")
    for e in op.schedule:
        tgt = world.planets_by_id[e.tgt_id]
        print(f"  fire_step={e.fire_step:3d}  src={e.src_id} -> tgt={e.tgt_id} "
              f"(prod={int(tgt.production)}, gar={int(tgt.ships)})  "
              f"ships={e.ships:3d}  eta={e.eta:3d}  value={e.value:7.2f}")

    sched_targets = {e.tgt_id for e in op.schedule}
    nineteen_in_sched = [p.id for p in nineteens if p.id in sched_targets]
    print(f"\n19-garrison planets in schedule: {nineteen_in_sched}")
    if not nineteen_in_sched:
        print("→ NONE of the 19-garrison neutrals were scheduled by the MILP.")

    # Now drill into WHY the 19s were dropped.
    # Manually walk the per-source top-K targets for each source and check
    # whether each 19 is in the top-K and what its predicted opp-contest is.
    print("\n--- per-source top-K eligibility for the 19-garrison neutrals ---")
    nineteen_ids = {p.id for p in nineteens}
    all_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != world.my_id]
    for src in my_planets:
        scored = sorted(
            ((float(t.production) / (_dist(src, t) + 1.0), t) for t in all_targets),
            key=lambda x: x[0], reverse=True,
        )
        top = [t for _s, t in scored[:TOP_TARGETS_PER_SOURCE]]
        print(f"\nsource id={src.id} ships={int(src.ships)} prod={int(src.production)}")
        print(f"  top-{TOP_TARGETS_PER_SOURCE} by prod/(dist+1):")
        for s, t in scored[:TOP_TARGETS_PER_SOURCE + 4]:
            mark = "★" if int(t.id) in nineteen_ids else " "
            in_top = " (TOP)" if t in top else "  (out)"
            print(f"     {mark} id={t.id:3d} prod={int(t.production)} gar={int(t.ships):3d} "
                  f"d={_dist(src, t):6.1f}  score={s:.3f}{in_top}")

        # For each 19 that IS in the top, what does each step's analysis say?
        for tgt in nineteens:
            if tgt not in top:
                continue
            print(f"\n  19-planet id={tgt.id} feasibility at offset=0:")
            ships_est = max(OPENING_DEFENDER_GUARD, int(tgt.ships) + 1)
            res = aim_and_eta(src, tgt, ships_est, world.omega, wait_N=0)
            if res is None:
                print("    aim_and_eta returned None → unreachable")
                continue
            angle, eta_flight = res
            arrival = int(eta_flight)
            # opp-contest check
            best_opp = 0
            for p in world.planets_by_id.values():
                owner = int(p.owner)
                if owner == world.my_id or owner < 0:
                    continue
                ships_avail = int(p.ships) - OPENING_DEFENDER_GUARD
                if ships_avail < MIN_SOURCE_SHIPS:
                    continue
                d = math.hypot(float(p.x) - float(tgt.x),
                               float(p.y) - float(tgt.y))
                v = fleet_speed(ships_avail)
                if v <= 0:
                    continue
                opp_eta = int(math.ceil(d / v))
                contested = opp_eta <= arrival + OPP_RESPONSE_LAG
                print(f"    opp source id={p.id} ships_avail={ships_avail} "
                      f"d={d:.1f} opp_eta={opp_eta}  our_arrival={arrival} "
                      f"(slack={OPP_RESPONSE_LAG}) → contested={contested}")
                if contested and ships_avail > best_opp:
                    best_opp = ships_avail
            print(f"    our needed = {ships_est}, opp_could_contest = {best_opp}")

            # Trajectory feasibility
            try:
                fate = predict_fleet_fate(src, tgt, angle, ships_est, world, wait_N=0)
                print(f"    fate outcome: {getattr(fate, 'outcome', 'UNKNOWN')}")
            except Exception as ex:
                print(f"    fate raised: {ex!r}")


if __name__ == "__main__":
    main()
