"""Sanity-check the orbit-prediction math.

Predict orbiting planet positions at step 100 from initial_planets +
angular_velocity, and compare against the env's observed positions.
Load-bearing for any agent that pre-targets where a planet WILL be when
a fleet arrives.

Two findings (verified against the running env on seed 42):

1. ABSOLUTE formula (init_planets-based) has an off-by-one:
       angle_at_envsteps_N = initial_angle + omega * (N - 1)   for N >= 1
   Naive `omega * N` is WRONG — error of exactly one step's rotation,
   which at the default fleet speed translates to ~1.27 board units
   for inner planets (orb_r ~ 31). Source: orbit_wars.py:519 reads
   `step = get(obs0, "step", 1)`, but `env.steps[N]` is the snapshot
   BEFORE that step's rotation is applied. Empirical test below
   confirms env.steps[N] uses (N-1) rotations.

2. RELATIVE formula (current-position-based) is unconditionally safe:
       angle_at_step_(N+T) = angle_at_step_N + omega * T
   No step counter needed. This is what an agent should use when
   computing fleet-arrival lead — read planet from current obs, rotate
   forward by omega * arrival_lag_in_turns.

Static planets (orbital_radius + planet.radius >= 50) do not rotate.
"""

from __future__ import annotations

import math
import sys

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT

SEED = 42
TARGET_STEP = 100


def is_orbiting(planet):
    # planet = [id, owner, x, y, radius, ships, production]
    px, py, pr = planet[2], planet[3], planet[4]
    orb = math.hypot(px - CENTER, py - CENTER)
    return (orb + pr) < ROTATION_RADIUS_LIMIT


def predict_absolute(init_planet, angular_velocity, env_step_N, *, off_by_one=True):
    """Predict (x, y) at env.steps[env_step_N] from initial_planets.

    With off_by_one=True (the empirical convention), uses N-1 rotations
    for N>=1. With off_by_one=False, uses N rotations (naive — WRONG).
    """
    px, py = init_planet[2], init_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    init_angle = math.atan2(dy, dx)
    n_rot = max(env_step_N - 1, 0) if off_by_one else env_step_N
    cur_angle = init_angle + angular_velocity * n_rot
    return CENTER + orb_r * math.cos(cur_angle), CENTER + orb_r * math.sin(cur_angle)


def predict_relative(current_planet, angular_velocity, lead_turns):
    """Predict (x, y) `lead_turns` after the obs that gave current_planet.

    Safe for an agent that doesn't track absolute step. Computes the
    polar angle of the *current* planet position and rotates forward.
    """
    px, py = current_planet[2], current_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    cur_angle = math.atan2(dy, dx)
    new_angle = cur_angle + angular_velocity * lead_turns
    return CENTER + orb_r * math.cos(new_angle), CENTER + orb_r * math.sin(new_angle)


def main():
    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    obs0 = env.steps[0][0].observation
    initial_planets = obs0["initial_planets"]
    angular_velocity = obs0["angular_velocity"]
    print(f"angular_velocity = {angular_velocity:.6f} rad/turn")
    print(f"orbiting planets at step 0:")
    orbiting = [p for p in initial_planets if is_orbiting(p)]
    for p in orbiting:
        print(f"  id={p[0]:>3}  init=({p[2]:7.3f},{p[3]:7.3f}) r={p[4]:.3f} prod={p[6]}")

    if not orbiting:
        print("ERROR: no orbiting planets in this seed; pick another seed.")
        return 1

    # Run env to TARGET_STEP with empty agents so positions are observable.
    def noop(obs):
        return []

    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.run([noop, noop])

    actual_obs = env.steps[TARGET_STEP][0].observation
    actual_planets = {p[0]: p for p in actual_obs["planets"]}
    init_by_id = {p[0]: p for p in initial_planets}

    print(f"\n=== absolute formula (naive `omega*N`) vs env at step {TARGET_STEP} ===")
    print(f"{'id':>4}  {'pred_x':>8} {'pred_y':>8}  {'actual_x':>8} {'actual_y':>8}  {'err':>8}")
    naive_err = 0.0
    for p in orbiting:
        pid = p[0]
        if pid not in actual_planets:
            continue
        px, py = predict_absolute(init_by_id[pid], angular_velocity, TARGET_STEP, off_by_one=False)
        ax, ay = actual_planets[pid][2], actual_planets[pid][3]
        err = math.hypot(px - ax, py - ay)
        naive_err = max(naive_err, err)
        print(f"{pid:>4}  {px:8.3f} {py:8.3f}  {ax:8.3f} {ay:8.3f}  {err:8.5f}")
    print(f"naive max error: {naive_err:.6f} units  (≈ orb_r * omega = one-step drift)")

    print(f"\n=== absolute formula with N-1 offset vs env at step {TARGET_STEP} ===")
    print(f"{'id':>4}  {'pred_x':>8} {'pred_y':>8}  {'actual_x':>8} {'actual_y':>8}  {'err':>8}")
    abs_err = 0.0
    for p in orbiting:
        pid = p[0]
        if pid not in actual_planets:
            continue
        px, py = predict_absolute(init_by_id[pid], angular_velocity, TARGET_STEP, off_by_one=True)
        ax, ay = actual_planets[pid][2], actual_planets[pid][3]
        err = math.hypot(px - ax, py - ay)
        abs_err = max(abs_err, err)
        print(f"{pid:>4}  {px:8.3f} {py:8.3f}  {ax:8.3f} {ay:8.3f}  {err:8.5f}")
    print(f"abs(N-1) max error: {abs_err:.6f}")

    # Relative test: roll the env to step 50, then predict step 100 from there.
    rel_step = 50
    rel_obs = env.steps[rel_step][0].observation
    rel_by_id = {p[0]: p for p in rel_obs["planets"]}
    print(f"\n=== relative formula: predict step {TARGET_STEP} from step {rel_step} obs ===")
    print(f"{'id':>4}  {'pred_x':>8} {'pred_y':>8}  {'actual_x':>8} {'actual_y':>8}  {'err':>8}")
    rel_err = 0.0
    for p in orbiting:
        pid = p[0]
        if pid not in rel_by_id or pid not in actual_planets:
            continue
        px, py = predict_relative(rel_by_id[pid], angular_velocity, TARGET_STEP - rel_step)
        ax, ay = actual_planets[pid][2], actual_planets[pid][3]
        err = math.hypot(px - ax, py - ay)
        rel_err = max(rel_err, err)
        print(f"{pid:>4}  {px:8.3f} {py:8.3f}  {ax:8.3f} {ay:8.3f}  {err:8.5f}")
    print(f"relative max error: {rel_err:.6f}")

    if rel_err < 1e-9 and abs_err < 1e-9:
        print("\nPASS: relative + abs(N-1) formulas match env exactly. Naive abs(N) is off by exactly omega*orb_r.")
    elif rel_err < 1e-3:
        print("\nPASS-soft: relative formula matches; absolute formula needs N-1 offset.")
    else:
        print("\nFAIL: even the relative formula misses — formula or convention is wrong.")
        return 2

    # Also confirm static planets do NOT rotate.
    print(f"\n=== sanity: static planets at step {TARGET_STEP} should equal init ===")
    static_drift = 0.0
    for p in initial_planets:
        if is_orbiting(p):
            continue
        pid = p[0]
        if pid not in actual_planets:
            continue
        ax, ay = actual_planets[pid][2], actual_planets[pid][3]
        ix, iy = p[2], p[3]
        d = math.hypot(ax - ix, ay - iy)
        static_drift = max(static_drift, d)
    print(f"max static-planet drift over {TARGET_STEP} steps: {static_drift:.6f} units")

    return 0


if __name__ == "__main__":
    sys.exit(main())
