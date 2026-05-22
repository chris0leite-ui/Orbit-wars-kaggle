"""Probe: does aim_orbiting miss the retrograde solution for antipodal targets?

For each (my-planet, orbiting-non-comet-target) pair at seed 42 step 0:
  1. Run aim_orbiting normally (converges from target's current position).
  2. Run a hand-iterated retrograde variant — initialize the lead-target at
     predict_relative(target, omega, half_period) (antipodal point), so the
     fixed-point converges to a DIFFERENT solution if one exists.
  3. Compare ETAs.

If the retrograde solution exists and has a smaller ETA, the current
proposer/aim_orbiting is leaving a fast intercept on the table.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make

from lib.aim import aim_orbiting, estimate_eta, MAX_ITERATIONS, CONVERGENCE_XY_TOL
from lib.orbit import predict_relative
from lib.intent import World


def _is_orbiting(planet_list) -> bool:
    """Orbiting iff the planet is NOT at the sun center (radius>0 to it)."""
    CENTER = 50.0
    return math.hypot(planet_list[2] - CENTER, planet_list[3] - CENTER) > 1.0


def aim_orbiting_seeded(src_xy, src_radius, target_tuple, target_radius,
                        ships, omega, initial_target_xy):
    """Hand-iterated aim_orbiting with an EXPLICIT initial-guess for the
    lead-target position. Same fixed-point as aim_orbiting but the start
    is `initial_target_xy` instead of (target.x, target.y).

    Returns (angle, eta, converged_xy) or None.
    """
    tx, ty = initial_target_xy
    last_eta = None
    for _ in range(MAX_ITERATIONS):
        eta = estimate_eta(src_xy, src_radius, (tx, ty), target_radius, ships)
        if eta is None:
            return None
        ntx, nty = predict_relative(target_tuple, omega, eta)
        if (last_eta is not None
                and abs(ntx - tx) < CONVERGENCE_XY_TOL
                and abs(nty - ty) < CONVERGENCE_XY_TOL):
            angle = math.atan2(nty - src_xy[1], ntx - src_xy[0])
            return angle, eta, (ntx, nty)
        tx, ty = ntx, nty
        last_eta = eta
    return None  # didn't converge


def angular_distance(p_src, p_tgt) -> float:
    """Angle subtended from sun center."""
    CENTER = 50.0
    a_src = math.atan2(p_src[3] - CENTER, p_src[2] - CENTER)
    a_tgt = math.atan2(p_tgt[3] - CENTER, p_tgt[2] - CENTER)
    d = abs(a_src - a_tgt)
    return min(d, 2 * math.pi - d)


def probe(seed: int, me_seat: int = 0, ships: int = 25):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.steps[0][me_seat].observation
    omega = float(obs.angular_velocity)
    planets = list(obs.planets)
    world = World.from_obs(obs)
    comet_ids = world.comet_ids
    half_period = math.pi / abs(omega) if omega != 0 else 0.0
    print(f"seed={seed}  omega={omega:.4f} rad/turn  half_period={half_period:.1f} turns")
    print(f"planets: {len(planets)}  orbiting_non_comet: "
          f"{sum(1 for p in planets if _is_orbiting(p) and int(p[0]) not in comet_ids)}")

    my_planets = [p for p in planets if int(p[1]) == me_seat and _is_orbiting(p)
                  and int(p[0]) not in comet_ids]
    targets = [p for p in planets if int(p[1]) != me_seat and _is_orbiting(p)
               and int(p[0]) not in comet_ids]

    print(f"\nmy orbiting planets: {len(my_planets)}   "
          f"orbiting non-comet targets: {len(targets)}\n")

    # Build all pairs, sort by angular distance descending (most antipodal first).
    pairs = []
    for src in my_planets:
        for tgt in targets:
            d_ang = angular_distance(src, tgt)
            pairs.append((d_ang, src, tgt))
    pairs.sort(key=lambda x: -x[0])

    print(f"{'src':>4} {'tgt':>4} {'ang_dist':>9} {'std_eta':>8} "
          f"{'std_ang':>8} {'retro_eta':>10} {'retro_ang':>10} "
          f"{'winner':>7} {'speedup':>8}")
    print("-" * 86)

    n_retro_faster = 0
    n_total = 0
    n_no_retro = 0
    speedups = []
    for d_ang, src, tgt in pairs[:20]:
        src_xy = (float(src[2]), float(src[3]))
        # Standard aim_orbiting — fixed-point starts at (tgt.x, tgt.y).
        std = aim_orbiting(src_xy, float(src[4]), tgt, float(tgt[4]), ships, omega)
        if std is None:
            std_eta = None
            std_ang = None
        else:
            std_ang, _xy, std_eta = std

        # Retrograde-seeded aim — fixed-point starts at the antipodal point
        # (where the target will be after half a period).
        anti_xy = predict_relative(tgt, omega, half_period)
        retro = aim_orbiting_seeded(src_xy, float(src[4]), tgt, float(tgt[4]),
                                    ships, omega, anti_xy)
        if retro is None:
            retro_eta = None
            retro_ang = None
            n_no_retro += 1
        else:
            retro_ang, retro_eta, _ = retro

        # Determine if retrograde really is a different solution (not the
        # same fixed point converged via a different route).
        if std and retro:
            same_solution = (abs(std_eta - retro_eta) < 0.5
                             and abs(std_ang - retro_ang) < 0.05)
        else:
            same_solution = False

        winner = "—"
        speedup = ""
        if std and retro and not same_solution:
            n_total += 1
            if retro_eta < std_eta:
                n_retro_faster += 1
                winner = "RETRO"
                sp = std_eta / max(0.5, retro_eta)
                speedup = f"{sp:.2f}×"
                speedups.append(sp)
            else:
                winner = "std"
                sp = retro_eta / max(0.5, std_eta)
                speedup = f"{sp:.2f}×"

        std_eta_s = f"{std_eta:.1f}" if std_eta is not None else "—"
        std_ang_s = f"{std_ang:+.2f}" if std_ang is not None else "—"
        retro_eta_s = f"{retro_eta:.1f}" if retro_eta is not None else "—"
        retro_ang_s = f"{retro_ang:+.2f}" if retro_ang is not None else "—"
        print(f"{int(src[0]):>4} {int(tgt[0]):>4} {d_ang:>9.2f} "
              f"{std_eta_s:>8} {std_ang_s:>8} {retro_eta_s:>10} {retro_ang_s:>10} "
              f"{winner:>7} {speedup:>8}")

    print()
    print(f"distinct solution-pairs: {n_total}  (of which retrograde faster: "
          f"{n_retro_faster})")
    print(f"retrograde solution did not converge: {n_no_retro}")
    if speedups:
        avg = sum(speedups) / len(speedups)
        mx = max(speedups)
        print(f"avg retrograde speedup: {avg:.2f}×    max: {mx:.2f}×")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ships", type=int, default=25)
    ap.add_argument("--seat", type=int, default=0)
    args = ap.parse_args()
    probe(args.seed, me_seat=args.seat, ships=args.ships)
