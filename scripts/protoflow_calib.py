"""protoflow_calib — synthetic-situation calibration for the action-space field.

PI method (2026-06-04): play simple opponents AND hand-built synthetic states to
learn how to define/calibrate the field. A synthetic state isolates ONE decision
so we can read the field's valuations directly instead of inferring them from
noisy games. Each scenario prints the ranked action field (importance per move)
and what the agent emits, then checks a desired property.

Key physics under test: fleet speed RISES with ship count
  speed = 1 + 5*(ln(ships)/ln(1000))**1.5
so a small fleet is a SLOW fleet. A well-defined field should make slow/small
launches low-value on their own, so dribbling 2-3 ships becomes improbable and
"wait, mass a faster fleet, strike" wins where it should.

Run:  python scripts/protoflow_calib.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.protoflow.main as proto
from lib.fleet import speed as fleet_speed


def make_obs(planets, fleets=None, player=0, step=20, omega=0.0):
    """planets: list of [id, owner, x, y, radius, ships, production].
    fleets:  list of [id, owner, x, y, angle, from_planet_id, ships]."""
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in (fleets or [])],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
        "remainingOverageTime": 60.0,
    }


def radius(prod):
    return 1.0 + math.log(prod)


def show(name, obs, want):
    proto.reset_trace()
    moves = proto.agent(obs)
    field = proto.get_last_field()
    print(f"\n=== {name} ===")
    print(f"  want: {want}")
    print("  field (ranked by importance):")
    for f in field[:8]:
        own = {-1: "neutral"}.get(f["tgt_owner"], f"P{f['tgt_owner']}")
        spd = fleet_speed(f["ships"])
        print(f"    src{f['src']:>2} -> tgt{f['tgt']:>2} [{own:>7} prod={f['prod']}]  "
              f"ships={f['ships']:>3} speed={spd:.2f}  ttc={f['ttc']:>4}  imp={f['imp']}")
    if not field:
        print("    (field empty)")
    print(f"  EMITTED: {moves if moves else '(hold)'}")


def main():
    # S1 — DRIBBLE vs MASS. One home planet with small spare next to a cheap
    # neutral. A 2-ship launch is slow (speed ~1.2). Does the field prefer the
    # slow trickle, or does waiting to mass a faster fleet rank higher / does it
    # hold? We expect a well-calibrated field to NOT dribble.
    home = [0, 0, 20.0, 50.0, radius(3), 3, 3]
    neutral = [1, -1, 33.0, 50.0, radius(2), 2, 2]   # dist 13, cheap
    far_enemy = [2, 1, 80.0, 50.0, radius(3), 30, 3]
    show("S1 dribble-vs-mass (home has 3 ships)",
         make_obs([home, neutral, far_enemy]),
         "small slow launch should be low-value; ideally hold or send decisively")

    # Same geometry but home has accumulated a real strike force.
    home_big = [0, 0, 20.0, 50.0, radius(3), 24, 3]
    show("S1b same target, home has 24 ships (fast fleet available)",
         make_obs([home_big, neutral, far_enemy]),
         "now a decisive fast capture should be high-value and emitted")

    # S2 — CONVERGENCE NEEDED. A defended neutral that NO single planet can take
    # alone, but two planets arriving the same turn can (combat sums them).
    a = [0, 0, 30.0, 40.0, radius(3), 18, 3]
    b = [1, 0, 30.0, 60.0, radius(3), 18, 3]
    defended = [2, -1, 50.0, 50.0, radius(4), 30, 4]   # needs ~31; neither solo (18)
    enemy = [3, 1, 90.0, 50.0, radius(3), 20, 3]
    show("S2 convergence-needed (two 18-ship planets vs a 30-garrison target)",
         make_obs([a, b, defended, enemy]),
         "a 2-source same-arrival cohort should form (combat-rule-1 summation)")

    # S3 — OVERREACH. A juicy target far away that the enemy clearly wins the
    # race to. The field should not send (low winnability / past reach ceiling).
    home3 = [0, 0, 10.0, 50.0, radius(3), 40, 3]
    juicy_far = [1, -1, 92.0, 50.0, radius(5), 5, 5]   # dist 82, enemy adjacent
    enemy_adj = [2, 1, 88.0, 50.0, radius(5), 40, 5]
    show("S3 overreach (juicy target far away, enemy adjacent)",
         make_obs([home3, juicy_far, enemy_adj]),
         "should NOT send the far losing-race shot")


if __name__ == "__main__":
    main()
