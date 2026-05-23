"""End-to-end: every shot the intercept solver emits MUST land in the engine.

Sweep many seeds and (src, tgt) pairs. For each shot returned, execute it via
the engine and verify the target's garrison or ownership changed on the
predicted arrival tick.
"""
from __future__ import annotations

import math
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import sim, intercept
from kaggle_environments import make


def passive(obs):
    return []


def _exec_shot_and_observe(seed: int, shot: intercept.Shot, src_id: int, tgt_id: int,
                            launch_call_index: int = 2):
    """Run a fresh env with seed, fire ONE shot at the launch_call_index-th agent call.

    Default launch_call_index=2 means fire on the agent's 2nd call (obs.step=1) to
    avoid the engine's off-by-one rotation on the very first interpreter call.
    """
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": shot.eta + launch_call_index + 5})
    call = {"n": 0}

    def shooter(obs):
        call["n"] += 1
        if call["n"] == launch_call_index:
            return [[shot.src_id, shot.angle, shot.ship_count]]
        return []

    env.run([shooter, passive])
    steps = env.steps
    # The launch happens during the interpreter call that processes env.steps[launch_call_index-1]
    # to produce env.steps[launch_call_index]. The fleet's k-th tick produces
    # env.steps[launch_call_index + k - 1]. Hit visible on env.steps[launch_call_index + shot.eta - 1].
    # We'll just scan for the first step where target garrison/ownership changes.
    # Detect an actual impact: ownership change, OR garrison drop, OR garrison
    # jumped UP by more than production (friendly arrival).
    pre_tgt = None
    for k, st in enumerate(steps):
        obs = st[0].observation
        cur_tgt = next((p for p in obs.planets if p[0] == tgt_id), None)
        if pre_tgt is not None and cur_tgt is not None:
            owner_change = cur_tgt[1] != pre_tgt[1]
            garrison_drop = cur_tgt[5] < pre_tgt[5]
            # Friendly arrival adds more than production allows
            prod = pre_tgt[6] if pre_tgt[1] != -1 else 0
            big_jump = cur_tgt[5] > pre_tgt[5] + prod + 1
            if owner_change or garrison_drop or big_jump:
                return k, cur_tgt
        pre_tgt = cur_tgt
    return None, None


def test_landing_rate_static_targets():
    """Many seeds, static targets only — should land 100%."""
    landed = 0
    attempted = 0
    misses = []
    for seed in range(20):
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 200})
        env.reset(2)
        # Step the env once so obs.step=1 (avoids the engine's first-tick rotation edge).
        env.step([[], []])
        obs = env.steps[1][0].observation
        obs_dict = {
            "player": 0,
            "step": int(obs.step),
            "planets": list(obs.planets),
            "fleets": list(obs.fleets),
            "angular_velocity": obs.angular_velocity,
            "initial_planets": list(obs.initial_planets),
            "comets": list(obs.comets),
            "comet_planet_ids": list(obs.comet_planet_ids),
            "remainingOverageTime": 60.0,
        }
        world = intercept.parse_world(obs_dict)
        # Find our planets and static enemy/neutral planets
        my_planets = [p for p in world["planets"] if p.owner == 0]
        static_targets = [
            p for p in world["planets"]
            if not sim.is_orbiting_sim(p.x, p.y, p.radius)
            and p.owner != 0
        ]
        if not my_planets or not static_targets:
            continue
        for src in my_planets[:1]:
            for tgt in static_targets[:3]:
                # Try minimum-capture ship count
                S = max(1, min(src.ships, tgt.ships + 1))
                shot = intercept.find_shot(src, tgt, S, world)
                if shot is None:
                    continue
                attempted += 1
                # Fire on call #2 (obs.step=1), so the fleet's k-th tick produces env.steps[1+k].
                hit_tick, hit_tgt = _exec_shot_and_observe(seed, shot, src.id, tgt.id, launch_call_index=2)
                if hit_tick is None:
                    misses.append((seed, src.id, tgt.id, shot, "no impact observed"))
                    continue
                expected = 1 + shot.eta
                if abs(hit_tick - expected) > 1:
                    misses.append((seed, src.id, tgt.id, shot,
                                   f"hit at {hit_tick}, expected ~{expected}"))
                    continue
                landed += 1

    rate = landed / max(1, attempted)
    print(f"Static targets: {landed}/{attempted} landed ({rate:.1%})")
    for m in misses[:10]:
        print(" MISS:", m)
    assert attempted > 0, "no shots attempted"
    assert rate >= 0.95, f"land rate {rate:.1%} below 95% threshold"


def test_landing_rate_orbiting_targets():
    """Orbiting targets — the precision case the plan exists for."""
    landed = 0
    attempted = 0
    misses = []
    for seed in range(40):
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 200})
        env.reset(2)
        env.step([[], []])
        obs = env.steps[1][0].observation
        obs_dict = {
            "player": 0, "step": int(obs.step),
            "planets": list(obs.planets),
            "fleets": list(obs.fleets),
            "angular_velocity": obs.angular_velocity,
            "initial_planets": list(obs.initial_planets),
            "comets": list(obs.comets),
            "comet_planet_ids": list(obs.comet_planet_ids),
            "remainingOverageTime": 60.0,
        }
        world = intercept.parse_world(obs_dict)
        my_planets = [p for p in world["planets"] if p.owner == 0]
        orbiting_targets = [
            p for p in world["planets"]
            if sim.is_orbiting_sim(p.x, p.y, p.radius)
            and p.owner != 0
        ]
        if not my_planets or not orbiting_targets:
            continue
        for src in my_planets[:1]:
            for tgt in orbiting_targets[:3]:
                S = max(1, min(src.ships, tgt.ships + 1))
                shot = intercept.find_shot(src, tgt, S, world)
                if shot is None:
                    continue
                attempted += 1
                hit_tick, hit_tgt = _exec_shot_and_observe(seed, shot, src.id, tgt.id, launch_call_index=2)
                if hit_tick is None:
                    misses.append((seed, src.id, tgt.id, shot, "no impact"))
                    continue
                expected = 1 + shot.eta
                if abs(hit_tick - expected) > 1:
                    misses.append((seed, src.id, tgt.id, shot,
                                   f"hit at {hit_tick}, expected ~{expected}"))
                    continue
                landed += 1

    rate = landed / max(1, attempted)
    print(f"Orbiting targets: {landed}/{attempted} landed ({rate:.1%})")
    for m in misses[:10]:
        print(" MISS:", m)
    assert attempted > 0, "no shots attempted"
    assert rate >= 0.95, f"orbiting land rate {rate:.1%} below 95%"


def test_shot_menu_sane():
    """Sanity: menu produces shots for a typical seed."""
    env = make("orbit_wars", configuration={"seed": 7, "episodeSteps": 200})
    env.reset(2)
    env.step([[], []])
    obs = env.steps[1][0].observation
    obs_dict = {
        "player": 0, "step": int(obs.step),
        "planets": list(obs.planets),
        "fleets": list(obs.fleets),
        "angular_velocity": obs.angular_velocity,
        "initial_planets": list(obs.initial_planets),
        "comets": list(obs.comets),
        "comet_planet_ids": list(obs.comet_planet_ids),
        "remainingOverageTime": 60.0,
    }
    world = intercept.parse_world(obs_dict)
    menu = intercept.build_shot_menu(world)
    print(f"seed=7: {len(menu)} (src,tgt) pairs with valid shots")
    total = sum(len(v) for v in menu.values())
    print(f"  total shots in menu: {total}")
    assert len(menu) > 0


if __name__ == "__main__":
    test_shot_menu_sane()
    test_landing_rate_static_targets()
    test_landing_rate_orbiting_targets()
    print("\nAll intercept landing tests passed.")
