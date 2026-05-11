"""Verify landing rate across every (source × target) motion combination.

We bin shots by the static/orbiting status of the source AND the target,
and assert ≥95% land rate in each bin. Also covers cometary targets.
"""
from __future__ import annotations

import math
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import sim, intercept
from kaggle_environments import make


def _obs_at_step1(seed: int):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 200})
    env.reset(2)
    env.step([[], []])
    return env.steps[1][0].observation


def _world_from(obs):
    return intercept.parse_world({
        "player": 0, "step": int(obs.step),
        "planets": list(obs.planets),
        "fleets": list(obs.fleets),
        "angular_velocity": obs.angular_velocity,
        "initial_planets": list(obs.initial_planets),
        "comets": list(obs.comets),
        "comet_planet_ids": list(obs.comet_planet_ids),
        "remainingOverageTime": 60.0,
    })


def _exec_and_detect(seed: int, shot: intercept.Shot, tgt_id: int) -> tuple[int | None, list | None]:
    """Run a fresh env, fire shot on agent's 2nd call, detect impact."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": shot.eta + 10})
    call = {"n": 0}
    def shooter(o):
        call["n"] += 1
        if call["n"] == 2:
            return [[shot.src_id, shot.angle, shot.ship_count]]
        return []
    def passive(o):
        return []
    env.run([shooter, passive])
    pre = None
    for k, st in enumerate(env.steps):
        cur = next((p for p in st[0].observation.planets if p[0] == tgt_id), None)
        if pre is not None and cur is not None:
            if cur[1] != pre[1] or cur[5] < pre[5]:
                return k, cur
            prod = pre[6] if pre[1] != -1 else 0
            if cur[5] > pre[5] + prod + 1:
                return k, cur
        pre = cur
    return None, None


def _bin(p: intercept.PlanetView) -> str:
    if p.is_comet:
        return "comet"
    return "orbiting" if sim.is_orbiting(p.x, p.y, p.radius) else "static"


def _obs_at_step(seed: int, target_step: int):
    """Step env forward (both agents pass) until obs.step == target_step."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": target_step + 50})
    env.reset(2)
    while env.steps[-1][0].observation.step < target_step:
        env.step([[], []])
    return env.steps[-1][0].observation


def test_comet_targets():
    """Comets spawn at step 50. Shoot at them and verify shots land."""
    attempted = landed = 0
    misses = []
    for seed in range(20):
        # Step to right after first comet spawn (step ~51)
        obs = _obs_at_step(seed, 51)
        if not obs.comets:
            continue
        world = _world_from(obs)
        my = [p for p in world["planets"] if p.owner == 0]
        comets = [p for p in world["planets"] if p.is_comet]
        if not my or not comets:
            continue
        # Try each (src, comet) pair
        for src in my[:2]:
            for cm in comets[:2]:
                S = max(1, min(src.ships, cm.ships + 1))
                shot = intercept.find_shot(src, cm, S, world)
                if shot is None:
                    continue
                attempted += 1
                # Fire the shot at obs.step=51 (= agent's 52nd call). Build a
                # shooter that fires on call #52.
                fire_call = obs.step + 1  # 1-indexed agent call count
                episode_len = obs.step + shot.eta + 15
                env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_len})
                state = {"n": 0}
                # Bind shot fields as defaults so the closure doesn't refer to
                # the loop-variable `shot` (which gets reassigned each iter).
                def make_shooter(fc, sid, ang, sh, st):
                    def _agent(o):
                        st["n"] += 1
                        if st["n"] == fc:
                            return [[sid, ang, sh]]
                        return []
                    return _agent
                shooter = make_shooter(fire_call, shot.src_id, shot.angle, shot.ship_count, state)
                def passive(o):
                    return []
                env.run([shooter, passive])
                # Detect impact on the comet.
                pre = None
                hit_at = None
                for k, st in enumerate(env.steps):
                    cur = next((p for p in st[0].observation.planets if p[0] == cm.id), None)
                    if cur is None and pre is not None:
                        hit_at = k  # comet expired or destroyed
                        break
                    if pre is not None and cur is not None:
                        if cur[1] != pre[1] or cur[5] < pre[5]:
                            hit_at = k
                            break
                    pre = cur
                if hit_at is not None:
                    landed += 1
                else:
                    misses.append((seed, src.id, cm.id, shot))

    print(f"Comet targets: {landed}/{attempted} landed ({landed/max(1,attempted):.1%})")
    for m in misses[:5]:
        print(" MISS:", m)
    if attempted >= 5:
        rate = landed / attempted
        assert rate >= 0.85, f"comet land rate {rate:.1%} below 85%"
    else:
        print("(not enough comet attempts to gate)")


def test_all_motion_combinations():
    """For each (src-type, tgt-type) bin, count attempts vs landings."""
    bins = {}  # (src_bin, tgt_bin) -> [attempted, landed]

    # Seeds 0..29: try shots from every owned planet to up to 6 closest non-self planets
    for seed in range(30):
        obs = _obs_at_step1(seed)
        world = _world_from(obs)
        my = [p for p in world["planets"] if p.owner == 0]
        all_p = world["planets"]
        for src in my[:2]:  # at most 2 sources per seed to keep runtime bounded
            others = sorted(
                (p for p in all_p if p.id != src.id),
                key=lambda p: (p.x - src.x) ** 2 + (p.y - src.y) ** 2,
            )[:6]
            for tgt in others:
                # Ship count: just enough to capture (or all available, whichever smaller).
                garrison = tgt.ships + (tgt.production * 3 if tgt.owner != -1 else 0)
                S = max(1, min(src.ships, garrison + 1))
                shot = intercept.find_shot(src, tgt, S, world)
                if shot is None:
                    continue
                key = (_bin(src), _bin(tgt))
                bins.setdefault(key, [0, 0])
                bins[key][0] += 1
                hit_tick, _ = _exec_and_detect(seed, shot, tgt.id)
                if hit_tick is not None:
                    expected = 1 + shot.eta
                    if abs(hit_tick - expected) <= 1:
                        bins[key][1] += 1
                    # else: mispredicted-arrival, but still counts as a land?
                    # For strict precision: only count if within +/-1 of predicted tick.

    print("Landing rate per (source × target) motion combination:")
    for key, (att, land) in sorted(bins.items()):
        rate = land / att if att else 0
        print(f"  {key[0]:>8s} -> {key[1]:>8s}: {land:3d}/{att:3d}  ({rate:.1%})")

    # Assert ≥95% for every (src, tgt) bin with at least 5 attempts.
    failures = []
    for key, (att, land) in bins.items():
        if att < 5:
            continue
        rate = land / att
        if rate < 0.95:
            failures.append((key, rate, att))
    assert not failures, f"Bins below 95% land rate: {failures}"


if __name__ == "__main__":
    test_all_motion_combinations()
    test_comet_targets()
    print("\nAll motion-combination landing tests passed.")
