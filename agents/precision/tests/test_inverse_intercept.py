"""Verify find_shot_for_arrival lands at the requested step (or ±1)."""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import intercept, sim
from kaggle_environments import make


def _world_at_step1(seed: int):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 100})
    env.reset(2)
    env.step([[], []])
    obs = env.steps[1][0].observation
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


def _exec_and_get_hit_step(seed: int, shot: intercept.Shot, tgt_id: int, launch_call: int = 2) -> int | None:
    """Fire the shot at launch_call and report the step where the target's
    garrison drops (i.e. impact occurred)."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": shot.eta + launch_call + 6})
    state = {"n": 0}

    def make_shooter(fc, sid, ang, sh, st):
        def _a(o):
            st["n"] += 1
            if st["n"] == fc:
                return [[sid, ang, sh]]
            return []
        return _a

    shooter = make_shooter(launch_call, shot.src_id, shot.angle, shot.ship_count, state)
    env.run([shooter, lambda o: []])
    pre = None
    for k, st in enumerate(env.steps):
        cur = next((p for p in st[0].observation.planets if p[0] == tgt_id), None)
        if pre is not None and cur is not None:
            if cur[1] != pre[1] or cur[5] < pre[5]:
                return k
            prod = pre[6] if pre[1] != -1 else 0
            if cur[5] > pre[5] + prod + 1:
                return k
        pre = cur
    return None


def test_find_shot_for_arrival_lands_on_requested_step():
    """For each seed, request shots at several target steps; verify exact arrival."""
    attempted = exact = within_1 = 0
    for seed in range(15):
        w = _world_at_step1(seed)
        cur = w["step"]
        my = [p for p in w["planets"] if p.owner == 0]
        others = sorted(
            (p for p in w["planets"] if p.owner != 0),
            key=lambda p: (p.x - my[0].x) ** 2 + (p.y - my[0].y) ** 2,
        )[:3] if my else []
        if not my or not others:
            continue
        src = my[0]
        for tgt in others:
            # Try requesting arrival at several steps.
            base_shot = intercept.find_shot(src, tgt, min(src.ships, 50), w)
            if base_shot is None:
                continue
            # Pick target steps around the natural ETA.
            for offset in (base_shot.eta, base_shot.eta + 3, base_shot.eta + 6):
                target_step = cur + offset
                shot = intercept.find_shot_for_arrival(src, tgt, target_step, w)
                if shot is None:
                    continue
                attempted += 1
                # Fire on launch_call=2 (just after env step([[], []]) we already did).
                # The shot we got is computed at obs.step=1; firing on call 2 makes
                # it launch at obs.step=1, so impact at env.steps[1 + shot.eta].
                hit = _exec_and_get_hit_step(seed, shot, tgt.id, launch_call=2)
                if hit is None:
                    continue
                expected = 1 + shot.eta
                if hit == expected:
                    exact += 1
                if abs(hit - expected) <= 1:
                    within_1 += 1

    print(f"Inverse intercept: {exact}/{attempted} exact ({exact/max(1,attempted):.0%}), "
          f"{within_1}/{attempted} within ±1 ({within_1/max(1,attempted):.0%})")
    assert attempted > 0
    assert within_1 / attempted >= 0.97, f"within-1 rate {within_1/attempted:.1%} below 97%"


if __name__ == "__main__":
    test_find_shot_for_arrival_lands_on_requested_step()
    print("\nInverse-intercept tests passed.")
