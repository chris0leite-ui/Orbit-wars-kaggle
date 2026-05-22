"""Probe: does aim_orbiting return systematically smaller ETAs for targets
on the INCOMING side (rotating toward source) than for targets on the
OUTGOING side (rotating away from source)?

For each (my-planet, orbiting-non-comet-target) pair at step 0:
  1. Compute SIGNED angular distance from source — sign matches omega
     direction. Positive = target is angularly AHEAD in rotation
     direction = target is rotating away (outgoing arc).
     Negative = target is angularly BEHIND = rotating toward source
     (incoming arc).
  2. Run aim_orbiting → get the actual eta.
  3. Compare eta vs |signed_distance|. If incoming-arc (negative-signed)
     targets have smaller ETAs at the same |distance| than outgoing-arc
     (positive-signed) targets, the asymmetry the PI describes is real
     and detectable.

Then we can ask: does the chooser's scoring (cheap_marginal_value)
have enough sensitivity to this eta gap to pick differently?
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make

from lib.aim import aim_orbiting
from lib.intent import World


CENTER = 50.0


def signed_angular_distance(src_xy, tgt_xy, omega: float) -> float:
    """Signed angular distance from src to tgt, measured in the rotation
    direction (sign of omega).

    Returns:
      positive  → target is AHEAD of source in rotation direction
                  (rotating AWAY from source — outgoing arc)
      negative  → target is BEHIND source in rotation direction
                  (rotating TOWARD source — incoming arc)
    """
    a_src = math.atan2(src_xy[1] - CENTER, src_xy[0] - CENTER)
    a_tgt = math.atan2(tgt_xy[1] - CENTER, tgt_xy[0] - CENTER)
    d = a_tgt - a_src
    # Normalize to (-pi, pi].
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d * (1.0 if omega > 0 else -1.0)


def _is_orbiting(p) -> bool:
    return math.hypot(p[2] - CENTER, p[3] - CENTER) > 1.0


def probe(seed: int, ships: int = 25, seat: int = 0):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.steps[0][seat].observation
    omega = float(obs.angular_velocity)
    planets = list(obs.planets)
    world = World.from_obs(obs)
    comet_ids = world.comet_ids

    my = [p for p in planets if int(p[1]) == seat and _is_orbiting(p)
          and int(p[0]) not in comet_ids]
    targets = [p for p in planets if int(p[1]) != seat and _is_orbiting(p)
               and int(p[0]) not in comet_ids]

    print(f"seed={seed}  omega={omega:+.4f} rad/turn  ships={ships}")
    print(f"  my orbiting planets: {len(my)}   targets: {len(targets)}")

    rows = []
    for src in my:
        for tgt in targets:
            sd = signed_angular_distance((src[2], src[3]), (tgt[2], tgt[3]), omega)
            res = aim_orbiting(
                (float(src[2]), float(src[3])), float(src[4]),
                tgt, float(tgt[4]), ships, omega,
            )
            if res is None:
                eta = None
            else:
                _angle, _xy, eta = res
            rows.append((int(src[0]), int(tgt[0]), sd, eta))

    # Sort by signed angular distance ASCENDING (most-incoming first).
    rows.sort(key=lambda r: r[2])

    print(f"\n  {'src':>3} {'tgt':>3} {'signed_ang':>11} {'|ang|':>7} "
          f"{'side':>9} {'eta':>6}")
    print("  " + "-" * 50)
    for src_id, tgt_id, sd, eta in rows:
        side = "INCOMING" if sd < 0 else ("OUTGOING" if sd > 0 else "—")
        eta_s = f"{eta:.1f}" if eta is not None else "—"
        print(f"  {src_id:>3} {tgt_id:>3} {sd:>+11.3f} {abs(sd):>7.3f} "
              f"{side:>9} {eta_s:>6}")

    # Asymmetry stats: bin by |signed_distance|, compare avg eta on each side.
    incoming = [(abs(r[2]), r[3]) for r in rows if r[2] < 0 and r[3] is not None]
    outgoing = [(abs(r[2]), r[3]) for r in rows if r[2] > 0 and r[3] is not None]

    print(f"\n  incoming arc (target rotating TOWARD source): "
          f"n={len(incoming)}  avg eta={_avg([e for _, e in incoming]):.1f}")
    print(f"  outgoing arc (target rotating AWAY from source): "
          f"n={len(outgoing)}  avg eta={_avg([e for _, e in outgoing]):.1f}")

    # Per-|ang|-band comparison (binsize=0.5 rad).
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.6), (2.6, math.pi + 0.01)]
    print(f"\n  {'|ang_dist|':>14}  {'n_in':>5} {'avg_eta_in':>11}  "
          f"{'n_out':>5} {'avg_eta_out':>12}  {'gap':>7}")
    for lo, hi in bins:
        in_etas = [e for d, e in incoming if lo <= d < hi]
        out_etas = [e for d, e in outgoing if lo <= d < hi]
        ai = _avg(in_etas)
        ao = _avg(out_etas)
        gap = ao - ai if (in_etas and out_etas) else None
        gap_s = f"{gap:+.1f}" if gap is not None else "—"
        ai_s = f"{ai:.1f}" if in_etas else "—"
        ao_s = f"{ao:.1f}" if out_etas else "—"
        print(f"  [{lo:>4.1f}, {hi:>4.1f})  {len(in_etas):>5} {ai_s:>11}  "
              f"{len(out_etas):>5} {ao_s:>12}  {gap_s:>7}")


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ships", type=int, default=25)
    ap.add_argument("--seat", type=int, default=0)
    args = ap.parse_args()
    probe(args.seed, ships=args.ships, seat=args.seat)
