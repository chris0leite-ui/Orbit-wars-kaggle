"""Verify hypothesis (C)+(E): rear planets are always-idle because
their solo launches would bounce, but joint launches with a neighbor
could capture.

Walks live replays, identifies always-idle our planets (>=100 own-steps,
0 launches), picks a mid-game snapshot, and for each idle planet:
  - For each of its nearest 6 non-our targets, compute solo launch ETA
    and predicted target garrison at arrival. Report bounce vs capture.
  - Also enumerate joint launches: idle planet + nearest own neighbor
    launching at near-simultaneous arrival. Report joint outcome.

Output: a table showing solo vs joint capture viability per (idle src,
target) pair, plus aggregate stats.

Reuses lib.fleet.speed for ETA, no agent re-execution.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.fleet import speed as fleet_speed

# How many own-steps must a planet have without launching to count as
# always-idle. The replays we look at are ~200-500 steps; 100 catches
# planets that should have had opportunities but never used them.
ALWAYS_IDLE_MIN_OWN_STEPS = 80


def planet_dict(p) -> dict:
    return {
        "id": int(p[0]),
        "owner": int(p[1]),
        "x": float(p[2]),
        "y": float(p[3]),
        "ships": int(p[5]),
        "production": int(p[6]),
    }


def find_always_idle(replay: dict, seat: int, my_id: int) -> dict:
    """Return {pid: own_steps_count} for planets that were ours for
    >= ALWAYS_IDLE_MIN_OWN_STEPS but launched 0 times."""
    own_steps = defaultdict(int)
    launches = defaultdict(int)
    for step_data in replay["steps"]:
        if seat >= len(step_data):
            continue
        obs = step_data[seat].get("observation") or {}
        action = step_data[seat].get("action") or []
        for p in obs.get("planets", []):
            if int(p[1]) == my_id:
                own_steps[int(p[0])] += 1
        for a in action:
            try:
                launches[int(a[0])] += 1
            except (TypeError, ValueError, IndexError):
                pass
    return {pid: cnt for pid, cnt in own_steps.items()
            if cnt >= ALWAYS_IDLE_MIN_OWN_STEPS and launches.get(pid, 0) == 0}


def pick_midgame_step(replay: dict, idle_pid: int, my_id: int,
                       seat: int) -> int | None:
    """Find a representative midgame step: idle planet is ours, has
    >=30 ships, and at least one non-our planet remains. Pick the
    earliest such step (catches the moment chooser SHOULD have
    launched first)."""
    n = len(replay["steps"])
    for s in range(max(1, n // 4), n - 1):
        if seat >= len(replay["steps"][s]):
            continue
        obs = replay["steps"][s][seat].get("observation") or {}
        planets = obs.get("planets", [])
        has_non_our = any(int(p[1]) != my_id and int(p[1]) >= -1
                          for p in planets)
        if not has_non_our:
            continue
        # Find idle planet at this step
        for p in planets:
            if int(p[0]) == idle_pid and int(p[1]) == my_id and int(p[5]) >= 30:
                return s
    return None


def analyse_solo_vs_joint(replay: dict, idle_pid: int, my_id: int,
                          step_idx: int, seat: int) -> dict:
    obs = replay["steps"][step_idx][seat]["observation"]
    planets_raw = obs.get("planets", [])
    planets = [planet_dict(p) for p in planets_raw]
    by_id = {p["id"]: p for p in planets}
    idle = by_id.get(idle_pid)
    if idle is None or idle["owner"] != my_id:
        return {"error": "idle planet missing or no longer ours at step"}

    my_planets = [p for p in planets if p["owner"] == my_id
                  and p["id"] != idle_pid]
    non_our = [p for p in planets if p["owner"] != my_id]

    def dist(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    nearest_targets = sorted(non_our, key=lambda t: dist(idle, t))[:6]
    nearest_neighbors = sorted(my_planets, key=lambda q: dist(idle, q))[:3]

    rows = []
    can_solo = 0
    can_joint = 0
    for tgt in nearest_targets:
        d = dist(idle, tgt)
        # Reserve a small constant; max launch from idle planet
        idle_max = max(0, idle["ships"] - 2)
        spd = fleet_speed(idle_max) if idle_max > 0 else 1.0
        eta = int(math.ceil(d / max(spd, 0.1)))

        # Predicted target ships at arrival (only production accrual;
        # ignores opp launches and arriving fleets — a SIMPLIFICATION
        # mirroring cheap_marginal_value's pred_ships_at path).
        # Neutrals (owner=-1) do NOT accrue production per engine rule.
        if tgt["owner"] == -1:
            pred_tgt_ships = tgt["ships"]
        else:
            pred_tgt_ships = tgt["ships"] + eta * tgt["production"]

        solo_caps = idle_max > pred_tgt_ships
        if solo_caps:
            can_solo += 1

        # Joint: assume the nearest own-neighbor also launches at the
        # same step toward the same target. Pessimistic ETA = max of
        # the two solo ETAs (we wait for the slower one).
        # Total ships = idle_max + neighbor_max.
        joint_caps = False
        joint_detail = ""
        for nb in nearest_neighbors:
            d_nb = dist(nb, tgt)
            nb_max = max(0, nb["ships"] - 2)
            if nb_max <= 0:
                continue
            spd_nb = fleet_speed(nb_max)
            eta_nb = int(math.ceil(d_nb / max(spd_nb, 0.1)))
            joint_eta = max(eta, eta_nb)
            joint_ships = idle_max + nb_max
            if tgt["owner"] == -1:
                pred_joint = tgt["ships"]
            else:
                pred_joint = tgt["ships"] + joint_eta * tgt["production"]
            if joint_ships > pred_joint:
                joint_caps = True
                joint_detail = (f"with pid {nb['id']} ({nb_max}s) at "
                                f"d={d_nb:.0f} eta={eta_nb} → "
                                f"{joint_ships}>{pred_joint}")
                break
        if joint_caps:
            can_joint += 1

        rows.append({
            "tgt_id": tgt["id"],
            "tgt_owner": tgt["owner"],
            "tgt_d": round(d, 1),
            "tgt_prod": tgt["production"],
            "tgt_ships_now": tgt["ships"],
            "eta": eta,
            "pred_tgt_at_arrival": int(pred_tgt_ships),
            "idle_max_ships": idle_max,
            "solo_captures": solo_caps,
            "joint_captures": joint_caps,
            "joint_detail": joint_detail,
        })

    return {
        "idle_pid": idle_pid,
        "idle_position": (round(idle["x"], 1), round(idle["y"], 1)),
        "idle_production": idle["production"],
        "idle_ships": idle["ships"],
        "step": step_idx,
        "rows": rows,
        "summary": f"solo {can_solo}/{len(rows)}  joint {can_joint}/{len(rows)}",
    }


def main():
    base = REPO / "audit" / "live-episodes" / "52754310"
    eps = sorted(base.glob("episode-*-replay.json"))
    if not eps:
        print(f"no episodes at {base}", file=sys.stderr)
        return 1

    total = defaultdict(int)  # aggregate solo/joint counts
    n_idle_planets = 0
    n_episodes = 0
    print(f"# Verification of (C)+(E) on 52754310 live episodes\n")
    print("Always-idle = our planet for >= 80 steps, 0 launches.")
    print("Per-snapshot: planet's max-ship moment + nearest 6 non-our targets.")
    print()

    for ep_path in eps:
        rep = json.load(open(ep_path))
        teams = rep.get("info", {}).get("TeamNames", [])
        our = [i for i, t in enumerate(teams) if t == "ChrisLeiteScha"]
        if not our:
            continue
        seat = our[0]
        my_id = int(rep["steps"][0][seat]["observation"].get("player", seat))
        idle = find_always_idle(rep, seat, my_id)
        if not idle:
            continue
        n_episodes += 1
        print(f"## {ep_path.name}  ({len(rep['steps'])} steps)")
        for idle_pid, own_steps in sorted(idle.items()):
            step = pick_midgame_step(rep, idle_pid, my_id, seat)
            if step is None:
                continue
            res = analyse_solo_vs_joint(rep, idle_pid, my_id, step, seat)
            if "error" in res:
                continue
            n_idle_planets += 1
            print(f"\n  idle pid {idle_pid} @ step {step} "
                  f"(own_steps={own_steps}, prod={res['idle_production']}, "
                  f"ships={res['idle_ships']}, "
                  f"xy={res['idle_position']})")
            print(f"    {'tgt':>3}  own  d  prod  now  eta  pred  solo  joint")
            for r in res["rows"]:
                marker = "@ENEMY" if r["tgt_owner"] >= 0 else "@NEUT"
                solo = "CAP" if r["solo_captures"] else "BNC"
                joint = "CAP" if r["joint_captures"] else "BNC"
                total[("solo", "cap" if r["solo_captures"] else "bnc")] += 1
                total[("joint", "cap" if r["joint_captures"] else "bnc")] += 1
                print(f"    {r['tgt_id']:>3}  {marker}  {r['tgt_d']:>4}  "
                      f"{r['tgt_prod']:>2}  {r['tgt_ships_now']:>3}  "
                      f"{r['eta']:>3}  {r['pred_tgt_at_arrival']:>3}  "
                      f"{solo:>4}  {joint:>4}  "
                      f"{r['joint_detail']}")
            print(f"    SUMMARY: {res['summary']}")
        print()

    print()
    print(f"# Aggregate ({n_idle_planets} idle planets across {n_episodes} episodes)")
    solo_cap = total[("solo", "cap")]
    solo_bnc = total[("solo", "bnc")]
    joint_cap = total[("joint", "cap")]
    joint_bnc = total[("joint", "bnc")]
    n_pairs = solo_cap + solo_bnc
    if n_pairs == 0:
        print("(no (idle, target) pairs analyzed)")
        return 0
    print(f"  solo  capture: {solo_cap}/{n_pairs} = {100*solo_cap/n_pairs:.1f}%")
    print(f"  joint capture: {joint_cap}/{n_pairs} = {100*joint_cap/n_pairs:.1f}%")
    delta = joint_cap - solo_cap
    print(f"  delta (joint - solo): +{delta} captures "
          f"({100*delta/n_pairs:.1f}pp lift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
