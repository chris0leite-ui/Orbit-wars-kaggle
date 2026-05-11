"""Verify the prediction layer exposes per-planet arrivals and production projection."""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import intercept, prediction, sim
from kaggle_environments import make


def _world_at(seed: int, target_step: int = 10):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": target_step + 50})
    env.reset(2)
    while env.steps[-1][0].observation.step < target_step:
        env.step([[], []])
    obs = env.steps[-1][0].observation
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


def test_production_by_player():
    world = _world_at(7, 5)
    prod = prediction.production_by_player(world)
    print(f"Production by player at step 5: {prod}")
    # 2-player game: both players own home planet(s); -1 not in dict (neutrals filtered).
    assert 0 in prod and 1 in prod
    assert -1 not in prod
    # Each player has at least 1 home planet with production >= 1.
    assert prod[0] >= 1
    assert prod[1] >= 1


def test_arrivals_timeline_includes_planned_and_enemy():
    world = _world_at(7, 5)
    me = world["planet_by_id"]
    my = [p for p in world["planets"] if p.owner == 0]
    others = [p for p in world["planets"] if p.owner != 0]
    src = my[0]
    tgt = others[0]
    S = max(1, min(src.ships, tgt.ships + 1))
    shot = intercept.find_shot(src, tgt, S, world)
    assert shot is not None

    timeline = prediction.planet_arrivals_timeline(world, [shot])
    # Our shot should appear in the target's timeline at the correct step.
    arrs_at_tgt = timeline[tgt.id]
    assert any(
        a.step == world["step"] + shot.eta and a.owner == 0 and a.ships == shot.ship_count
        for a in arrs_at_tgt
    ), f"shot not in timeline; timeline[{tgt.id}] = {arrs_at_tgt}"
    print(f"Timeline at target {tgt.id}: {arrs_at_tgt}")


def test_planet_garrison_projection_accrues_production():
    """An owned planet untouched by combat should grow by production each step."""
    world = _world_at(7, 5)
    my = [p for p in world["planets"] if p.owner == 0]
    src = my[0]
    proj = prediction.planet_garrison_projection(world, [], src.id, horizon_steps=20)
    # Without any arrivals, ownership stays the same, ships grow by `production` each step.
    print(f"Garrison projection for {src.id} (prod={src.production}): {proj[:5]} ...")
    base = src.ships + src.production  # first step adds production
    for i, (step, owner, ships) in enumerate(proj):
        expected = src.ships + (i + 1) * src.production
        assert owner == 0
        assert ships == expected, f"step {step}: expected {expected} ships, got {ships}"


def test_capture_flips_production_to_us():
    """When we capture an enemy/neutral planet, its production should accrue to us thereafter."""
    world = _world_at(7, 5)
    my = [p for p in world["planets"] if p.owner == 0]
    neutrals = [p for p in world["planets"] if p.owner == -1]
    if not neutrals:
        return  # skip
    src = my[0]
    tgt = neutrals[0]
    S = max(1, min(src.ships, tgt.ships + 1))
    shot = intercept.find_shot(src, tgt, S, world)
    if shot is None:
        return
    proj = prediction.planet_garrison_projection(world, [shot], tgt.id, horizon_steps=shot.eta + 5)
    # Before arrival: neutral. After: owner=0, then ships grow by production.
    arrival_step = world["step"] + shot.eta
    pre_arrival = [t for t in proj if t[0] < arrival_step][-1]
    post_arrival = [t for t in proj if t[0] >= arrival_step][0]
    print(f"Pre-arrival ({pre_arrival[0]}): owner={pre_arrival[1]} ships={pre_arrival[2]}")
    print(f"Post-arrival ({post_arrival[0]}): owner={post_arrival[1]} ships={post_arrival[2]}")
    assert pre_arrival[1] == -1, "should still be neutral before our fleet lands"
    assert post_arrival[1] == 0, "should be ours after capture"

    # Steps after arrival should grow by tgt.production each tick.
    after = sorted([t for t in proj if t[0] > arrival_step])
    if len(after) >= 2:
        delta = after[1][2] - after[0][2]
        assert delta == tgt.production, f"production delta {delta} != tgt.production {tgt.production}"
        print(f"Production confirmed: +{tgt.production} ships/step on captured planet")


def test_introspection_is_cheap():
    """Per-step tracking shouldn't blow the time budget."""
    import time
    world = _world_at(7, 30)
    my = [p for p in world["planets"] if p.owner == 0]
    if not my:
        return
    t0 = time.perf_counter()
    res = prediction.rollout(world, [], horizon_steps=200, track_per_step=True)
    elapsed = time.perf_counter() - t0
    print(f"rollout(track_per_step=True, horizon=200): {elapsed*1000:.1f}ms, "
          f"step_state has {len(res.step_state)} entries, "
          f"production_per_player has {len(res.production_per_player)} entries")
    assert elapsed < 0.1, f"rollout took {elapsed*1000:.0f}ms (>100ms threshold)"


if __name__ == "__main__":
    test_production_by_player()
    test_arrivals_timeline_includes_planned_and_enemy()
    test_planet_garrison_projection_accrues_production()
    test_capture_flips_production_to_us()
    test_introspection_is_cheap()
    print("\nAll prediction-introspection tests passed.")
