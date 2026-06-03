"""Replay pre-check for the race-gate lever (build spec Step 1).

For each downloaded replay + our seat, at sampled steps, reconstruct the
world and measure, using ONLY the existing primitives (no behaviour change):

  (1) Of the launches WE actually made, how many were `race_loss` (arrive
      at/after the opponent can contest) — the waste the race-gate removes.
  (2) Of all (source, target) pairs we COULD launch this turn, how many are
      `race_win` opportunities whose arrival ETA exceeds the current
      state-driven K — i.e. captures the live launch gate silently drops but
      the race-gate would admit. This is the value the gate unlocks.

The point: confirm the lever has signal before wiring anything (Rule 47 /
the contest-aware-conversion build sequence). Read-only / offline.

Usage:
  BASELINE_STATE_DRIVEN_K=1 BASELINE_STATE_K_ORBITAL_LEAD=1 \
    python scripts/probe_race_gate_replay.py /tmp/replays/episode-*.json
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

# Match the shipped agent config so "current K" is what actually ran.
os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
os.environ.setdefault("BASELINE_STATE_DRIVEN_K", "1")
os.environ.setdefault("BASELINE_STATE_K_ORBITAL_LEAD", "1")

from agents.baseline.proposer import MIN_FLEET_SIZE, aim_and_eta  # noqa: E402
from agents.baseline.launch_rules import (  # noqa: E402
    capture_horizon_k,
    resolve_launch_target,
)
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel, predict_arrival_contest  # noqa: E402

SAMPLE_EVERY = 5          # sample every Nth step (cost)
NEAREST_TARGETS = 6       # candidate targets per source (mirrors proposer)
HOLD_WINDOW = 15          # turns after landing to check we still own the target


def sufficient_contest_tick(world, model, tgt, me, defender_ships, omega):
    """Lever 3 v1 — earliest tick the opponent can land ENOUGH force to retake
    `tgt` from a defender holding `defender_ships`, vs the shipped model's
    "any fleet arrives" tick. Accumulates enemy garrisons in arrival-tick order
    (same-tick combat sums); the first tick cumulative arrivable force exceeds
    the defender is the sufficient-contest tick. Returns None if the opponent
    can never mass enough (truly bankable). Reuses aim_and_eta (Rule 47)."""
    arrivals = []
    for e in world.planets_by_id.values():
        if int(e.owner) in (int(me), -1) or int(e.ships) <= 0 or int(e.id) == int(tgt.id):
            continue
        _a, eta = aim_and_eta(e, tgt, int(e.ships), omega, world=world)
        arrivals.append((int(eta), int(e.ships)))
    arrivals.sort()
    cum = 0
    for eta, force in arrivals:
        cum += force
        if cum > defender_ships:
            return eta
    return None


def classify_sufficient(world, model, tgt, me, our_arrival, defender_ships,
                        omega, hold_margin=2):
    tick = sufficient_contest_tick(world, model, tgt, me, defender_ships, omega)
    if tick is None:
        return "bankable"
    if our_arrival + hold_margin < tick:
        return "race_win"
    return "race_loss"


def our_seat(d):
    info = d.get("info", {})
    names = info.get("TeamNames") or info.get("teamNames") or []
    return names.index("ChrisLeiteScha") if "ChrisLeiteScha" in names else 0


def _nearest(targets, src, k):
    return sorted(
        targets, key=lambda t: math.hypot(t.x - src.x, t.y - src.y)
    )[:k]


def analyze(path):
    d = json.load(open(path))
    seat = our_seat(d)
    steps = d["steps"]
    nseats = len(steps[0])

    actual_by_class = {"race_win": 0, "bankable": 0, "race_loss": 0}
    # Hold-outcome confound-sweep (Rule 41): for each committed capture, did we
    # still own the target HOLD_WINDOW turns later? If race_loss captures are
    # actually lost far more often than race_win ones, the classifier is
    # accurate and the gate is safe; if hold-rates match, it is over-pessimistic.
    held = {"race_win": [0, 0], "bankable": [0, 0], "race_loss": [0, 0]}  # [held, total]
    # Same hold-outcome sweep, but classified by the sufficiency-aware model.
    held_suf = {"race_win": [0, 0], "bankable": [0, 0], "race_loss": [0, 0]}
    # Of race_loss captures we HELD: were they actually attacked (intent), or
    # never contested at all (capability>>intent — the opponent-behaviour gap)?
    rl_held_attacked = [0, 0]  # [attacked, total_held]
    actual_total = 0
    unlocked_race_wins = 0          # race-win opps with eta > current K
    unlocked_prod = 0               # production summed over those opps
    sampled = 0

    for si in range(len(steps)):
        st = steps[si]
        action = st[seat].get("action") or []
        do_scan = (si % SAMPLE_EVERY == 0)
        if not action and not do_scan:
            continue
        obs = st[seat]["observation"]
        try:
            world = World.from_obs(dict(obs))
            model = WorldModel.from_world(world)
        except Exception:
            continue
        if do_scan:
            sampled += 1
        omega = float(getattr(world, "omega", 0.0))
        planets = list(world.planets_by_id.values())
        my_src = [p for p in planets if int(p.owner) == seat
                  and int(p.ships) >= MIN_FLEET_SIZE]
        targets = [p for p in planets if int(p.owner) != seat]

        # (1) classify the launches we actually emitted this step. Resolve the
        # TRUE target by ray-casting the launch angle (same primitive the gate
        # uses), and only classify captures of non-our planets — and only
        # those the fleet can actually take (ships > garrison), so the
        # race_loss count reflects committed captures, not under-strength pokes.
        for mv in action:
            src = world.planets_by_id.get(int(mv[0]))
            if src is None:
                continue
            hit_pid, step, outcome = resolve_launch_target(
                src, mv[1], int(mv[2]), world,
            )
            if hit_pid is None:
                continue
            tgt = world.planets_by_id.get(int(hit_pid))
            if tgt is None or int(tgt.owner) == seat:
                continue  # reinforcement / unknown — not a capture
            if int(mv[2]) <= int(tgt.ships):
                continue  # under-strength poke, not a committed capture
            ac = predict_arrival_contest(model, world, int(hit_pid), int(step), seat)
            actual_by_class[ac.race_class] = actual_by_class.get(ac.race_class, 0) + 1
            actual_total += 1
            # Sufficiency-aware (Lever 3) classification of the same capture.
            defender = max(1, int(mv[2]) - int(ac.predicted_garrison))
            suf_class = classify_sufficient(
                world, model, tgt, seat, int(step), defender, omega,
            )
            # Did the launch land AND did we still hold it HOLD_WINDOW later?
            land_si = si + int(step)
            check_si = land_si + HOLD_WINDOW
            if check_si < len(steps):
                future = steps[check_si][seat]["observation"]["planets"]
                owner_then = next(
                    (int(p[1]) for p in future if int(p[0]) == int(hit_pid)), None,
                )
                rec = held[ac.race_class]
                rec[1] += 1
                if owner_then == seat:
                    rec[0] += 1
                rec2 = held_suf[suf_class]
                rec2[1] += 1
                if owner_then == seat:
                    rec2[0] += 1
                # For race_loss captures we HELD: was the planet ever actually
                # attacked during the window (owner flipped away, or ships
                # dropped while we owned it)? "Never attacked" = the opponent
                # could contest but didn't (intent gap).
                if ac.race_class == "race_loss" and owner_then == seat:
                    rl_held_attacked[1] += 1
                    attacked = False
                    prev = None
                    for w in range(land_si, check_si + 1):
                        pl = next((p for p in steps[w][seat]["observation"]["planets"]
                                   if int(p[0]) == int(hit_pid)), None)
                        if pl is None:
                            continue
                        if int(pl[1]) != seat:
                            attacked = True
                            break
                        if prev is not None and float(pl[5]) < prev:
                            attacked = True
                            break
                        prev = float(pl[5])
                    if attacked:
                        rl_held_attacked[0] += 1

        # (2) race-win opportunities the current K would drop.
        if not do_scan:
            continue
        for src in my_src:
            for tgt in _nearest(targets, src, NEAREST_TARGETS):
                ships = min(int(src.ships), int(tgt.ships) + 1)
                if ships < MIN_FLEET_SIZE:
                    continue
                _ang, eta = aim_and_eta(src, tgt, ships, omega, world=world)
                k = capture_horizon_k(
                    si, tgt_id=int(tgt.id), world=world, model=model, me=seat,
                )
                if int(eta) <= k:
                    continue  # current gate already admits it
                ac = predict_arrival_contest(model, world, int(tgt.id), int(eta), seat)
                if ac.race_class == "race_win":
                    unlocked_race_wins += 1
                    unlocked_prod += int(tgt.production)

    ep = path.split("-")[1] if "-" in path else path
    print(f"\nep {ep} ({nseats}P, seat {seat}, {sampled} sampled steps)")
    print(f"  our launches classified: total={actual_total} "
          f"race_win={actual_by_class['race_win']} "
          f"bankable={actual_by_class['bankable']} "
          f"race_loss={actual_by_class['race_loss']}  "
          f"<- race_loss = wasted fleets the gate removes")
    print(f"  race-win opps DROPPED by current K (eta>K): {unlocked_race_wins} "
          f"(summed production {unlocked_prod})  <- value the gate unlocks")
    def _hr(rec):
        return f"{rec[0]}/{rec[1]} ({100*rec[0]/rec[1]:.0f}%)" if rec[1] else "0/0 (—)"
    print(f"  hold@{HOLD_WINDOW} WORST-CASE   : "
          f"race_win {_hr(held['race_win'])}  "
          f"bankable {_hr(held['bankable'])}  "
          f"race_loss {_hr(held['race_loss'])}")
    print(f"  hold@{HOLD_WINDOW} SUFFICIENCY  : "
          f"race_win {_hr(held_suf['race_win'])}  "
          f"bankable {_hr(held_suf['bankable'])}  "
          f"race_loss {_hr(held_suf['race_loss'])}  "
          f"<- Lever 3: want race_loss hold-rate LOW & separated")
    print(f"  race_loss-but-HELD captures actually attacked: {_hr(rl_held_attacked)} "
          f"<- low = opponent COULD contest but didn't (intent gap)")


def main():
    paths = sys.argv[1:] or sorted(glob.glob("/tmp/replays/episode-*.json"))
    print(f"config: STATE_DRIVEN_K={os.environ['BASELINE_STATE_DRIVEN_K']} "
          f"ORBITAL_LEAD={os.environ['BASELINE_STATE_K_ORBITAL_LEAD']}")
    for p in paths:
        try:
            analyze(p)
        except Exception as e:
            print(f"  {p}: ERROR {e}")


if __name__ == "__main__":
    main()
