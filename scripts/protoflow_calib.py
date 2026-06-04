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
    return moves, field


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    # NOTE: the sun is at (50,50) with radius 10 and destroys any fleet that
    # crosses it. All planets below sit well clear of the sun AND have a clear
    # line of sight to their targets, so trajectories are not silently rejected.

    # S1 — HOLD vs FIRE (the real dribble test). A single home that CANNOT solo-
    # capture a defended target right now (needs ~26, has 10), with no own planet
    # nearby to create reinforcement noise. The only field entries are wait-then-
    # mass candidates (no affordable fire-now capture), so the agent should HOLD
    # and accumulate -- NOT dribble a doomed 10-ship fleet that bounces. A distant
    # enemy means the eventual massed strike still wins the race (worth waiting).
    home = [0, 0, 15.0, 15.0, radius(3), 10, 3]        # 10 ships, prod 3
    defended = [1, -1, 38.0, 28.0, radius(3), 25, 3]   # garrison 25, dist ~26; floor ~26 > 10
    far_enemy = [2, 1, 85.0, 85.0, radius(3), 20, 3]   # far -> the held mass still wins the race
    moves, field = show("S1 hold-vs-fire (home has 10, target needs ~26)",
         make_obs([home, defended, far_enemy]),
         "no affordable fire-now capture -> HOLD and accumulate (no dribble)")
    # The field may still SHOW a sub-floor fire-now entry; the test is that the
    # agent does not FIRE it (it holds and accumulates a capturing fleet instead).
    _check("S1", not moves, f"emitted={moves or '(hold)'} (want hold to accumulate)")

    # S1b — same target, home has accumulated a real strike force. Now a decisive
    # fast capture is affordable and high-value -> it should fire.
    home_big = [0, 0, 15.0, 15.0, radius(3), 30, 3]    # 30 ships, can solo now
    moves, field = show("S1b same target, home has 30 ships (decisive fleet ready)",
         make_obs([home_big, defended, far_enemy]),
         "now a decisive fast capture is affordable and should be emitted")
    _check("S1b", bool(moves), f"emitted={moves or '(hold)'}")

    # S2 — CONVERGENCE NEEDED. A defended neutral that NO single planet can take
    # alone, but two planets arriving the same turn can (combat sums them). The
    # two sources are placed symmetrically so both legs share an arrival turn.
    a = [0, 0, 15.0, 30.0, radius(3), 18, 3]
    b = [1, 0, 15.0, 8.0, radius(3), 18, 3]
    defended = [2, -1, 40.0, 19.0, radius(4), 26, 4]   # floor ~27; neither solo (18), both (36) yes
    enemy = [3, 1, 88.0, 80.0, radius(3), 20, 3]
    moves, field = show("S2 convergence-needed (two 18-ship planets vs a 26-garrison target)",
         make_obs([a, b, defended, enemy]),
         "a 2-source same-arrival cohort should form (combat-rule-1 summation)")
    legs_to_def = [m for m in moves if int(m[0]) in (0, 1)]
    _check("S2", len(legs_to_def) >= 2,
           f"emitted={moves or '(hold)'} (want >=2 legs converging on target 2)")

    # S3 — OVERREACH. A juicy target so far the flight exceeds the reach ceiling
    # AND the enemy adjacent wins the race. The field should not send it.
    home3 = [0, 0, 10.0, 80.0, radius(3), 40, 3]
    juicy_far = [1, -1, 95.0, 80.0, radius(5), 5, 5]   # dist 85 (> reach ceiling), enemy adjacent
    enemy_adj = [2, 1, 90.0, 80.0, radius(5), 40, 5]
    moves, field = show("S3 overreach (juicy target far away, enemy adjacent)",
         make_obs([home3, juicy_far, enemy_adj]),
         "should NOT send the far losing-race shot")
    _check("S3", not moves, f"emitted={moves or '(hold)'} (want hold)")


if __name__ == "__main__":
    main()
