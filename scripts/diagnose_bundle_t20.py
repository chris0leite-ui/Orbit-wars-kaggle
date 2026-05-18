"""Step bundle vs v7_0 to a target turn, then inspect WHY bundle's
chooser returns empty (when intuition says it should attack).

Outputs: top-K candidate bundles sorted by score, with breakdown.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make

from agents.bundle.main import (
    agent as bundle_agent, _build_searches, _carry_over, _as_obs_dict,
)
from lib.trajectory_layer import (
    Bundle, BundleEvaluator, BundleSearch, World, SunFilter,
    predict_opp_bundles_via_mirror_search,
)


def load_v7_0():
    spec = importlib.util.spec_from_file_location(
        "v7_0_loaded", ROOT / "submissions" / "v7_0_drop_one.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v7_0_loaded"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turn", type=int, default=20)
    args = ap.parse_args()

    v7_0 = load_v7_0()
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.reset(num_agents=2)

    # Step to target turn.
    for t in range(args.turn):
        if env.state[0].status != "ACTIVE":
            break
        a0 = bundle_agent(env.state[0].observation)
        a1 = v7_0(env.state[1].observation)
        env.step([a0, a1])

    obs0 = env.state[0].observation
    obs_d = _as_obs_dict(obs0)
    world = World.from_obs(obs_d)
    me = 0

    print(f"=== inspection at t={args.turn} ===")
    print(f"my planets:")
    for p in world.planets:
        if p.is_comet:
            continue
        marker = "OURS" if p.owner == 0 else ("OPP " if p.owner == 1 else "NEUT")
        print(f"  pid={p.id:2d} {marker} ships={int(p.ships):3d} "
              f"prod={p.production:.1f} at ({p.current_x:.1f}, {p.current_y:.1f})")
    print(f"in-flight fleets: {len(world.fleets)}")
    for f in world.fleets[:8]:
        marker = "MINE" if f.owner == 0 else "OPP "
        print(f"  fid={f.id:3d} {marker} ships={f.ships:3d} "
              f"from={f.from_planet_id} at ({f.current_x:.1f}, {f.current_y:.1f})")

    own_search, opp_search, mirror_depth = _build_searches()
    print(f"\nsearch knobs: max_depth={own_search.max_depth} "
          f"beam={own_search.beam_width} "
          f"cands={own_search.candidates_per_source} "
          f"horizon={own_search.evaluator.horizon} "
          f"reserve={own_search.reserve_ships_at_source}")

    # Mirror search (so candidates score against the realistic overlay).
    t0 = time.perf_counter()
    mirror = predict_opp_bundles_via_mirror_search(
        world, my_id=me, search=opp_search, depth=mirror_depth)
    print(f"\nmirror-search cost: {(time.perf_counter()-t0)*1000:.0f}ms; "
          f"opp launches predicted: {sum(len(b.launches) for b in mirror.values())}")
    for opp_id, b in mirror.items():
        print(f"  opp_id={opp_id} bundle has {len(b.launches)} launches:")
        for ls in b.launches[:5]:
            print(f"    src={ls.src_id} angle={ls.aim_angle:.2f} ships={ls.ships} "
                  f"launch_turn={ls.launch_turn}")

    # Score empty bundle.
    empty = Bundle()
    empty_score = own_search.evaluator.score(world, empty, my_id=me,
                                              opp_overlays=mirror)
    print(f"\nempty bundle score: total={empty_score.total:.1f} "
          f"ship_delta={empty_score.ship_delta:.1f} "
          f"planet_delta={empty_score.planet_delta:.1f} "
          f"prod_delta={empty_score.production_delta:.1f}")

    # Enumerate candidates and score each as SINGLE-spec bundle.
    sun = SunFilter(world, safety_margin=own_search.sun_safety)
    candidates = list(own_search._enumerate_candidates(world, me, sun))
    print(f"\nenumerated {len(candidates)} candidate launches "
          f"(single-launch bundles):")
    scored = []
    for spec in candidates:
        b = Bundle(launches=(spec,))
        try:
            sc = own_search.evaluator.score(world, b, my_id=me,
                                             opp_overlays=mirror)
            scored.append((sc.total, spec, sc))
        except Exception as e:
            scored.append((float("-inf"), spec, None))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'rank':>4} {'total':>9} {'Δtot':>9} {'sh-Δ':>6} {'pl-Δ':>6} "
          f"{'pr-Δ':>6}  src tgt-angle ships  launch_t")
    print(f"{' ':>4} {empty_score.total:>9.1f} {0:>9.1f} "
          f"{empty_score.ship_delta:>6.0f} {empty_score.planet_delta:>6.0f} "
          f"{empty_score.production_delta:>6.0f}  empty bundle")
    for i, (tot, spec, sc) in enumerate(scored[:15]):
        if sc is None:
            print(f"{i:>4} {'inf-feas':>9}")
            continue
        d = tot - empty_score.total
        print(f"{i:>4} {tot:>9.1f} {d:>+9.1f} "
              f"{sc.ship_delta:>6.0f} {sc.planet_delta:>6.0f} "
              f"{sc.production_delta:>6.0f}  "
              f"{spec.src_id:>3} {spec.aim_angle:>+6.2f}    "
              f"{spec.ships:>3}  {spec.launch_turn:>3}")

    chosen = own_search.search(world, my_id=me, opp_overlays=mirror,
                                deadline=time.perf_counter() + 5.0)
    print(f"\nfull beam search result: "
          f"{len(chosen.launches)} launches in chosen bundle")
    for ls in chosen.launches:
        print(f"  src={ls.src_id} angle={ls.aim_angle:.2f} "
              f"ships={ls.ships} launch_turn={ls.launch_turn}")


if __name__ == "__main__":
    main()
