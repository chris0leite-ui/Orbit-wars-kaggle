"""scripts/trace_duel.py — per-tick economy/launch trace of one 2P game.

Why this exists: the Producer-matchup rebuild needs CURVES, not just
win/loss. For each tick this logs, per player: planet count, production
sum, garrison sum, in-flight sum, largest in-flight stack; plus capture
events (who took which planet, what the garrison cost) and every launch
(owner, size) reconstructed from fleet-set diffs.

Liveness asserts (audit 2026-06-12): both players must launch at least
one fleet AND the game must run > 30 steps, else exit(2) loudly — a dead
opponent sweeps exactly like a dominated one.

Usage:
    python scripts/trace_duel.py P0.py P1.py --seed 300 --out /tmp/t.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("p0")
    ap.add_argument("p1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([args.p0, args.p1])

    import math
    CENTER, SUN_R, BOARD = 50.0, 10.0, 100.0

    def _speed(ships):
        if ships <= 1:
            return 1.0
        return min(6.0, 1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5)

    rows = []
    captures = []
    launches = []
    deaths = []          # fleet removals classified: sun / oob / planet
    prev_owner: dict[int, int] = {}
    prev_ships: dict[int, int] = {}
    prev_fids: set[int] = set()
    prev_fleet_by_id: dict[int, list] = {}

    for t, step in enumerate(env.steps):
        obs = step[0]["observation"]
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []

        # classify fleets that vanished since last step
        cur_ids = {f[0] for f in fleets}
        planet_xy = {p[0]: (float(p[2]), float(p[3]), float(p[4]), p[1])
                     for p in planets}
        for fid, f in prev_fleet_by_id.items():
            if fid in cur_ids:
                continue
            x, y, ang, ships = float(f[2]), float(f[3]), float(f[4]), int(f[6])
            v = _speed(ships)
            nx, ny = x + math.cos(ang) * v, y + math.sin(ang) * v
            # nearest planet to the segment end (positions are this step's)
            near = None
            for pid, (px, py, pr, po) in planet_xy.items():
                dd = math.hypot(nx - px, ny - py) - pr
                if near is None or dd < near[0]:
                    near = (dd, pid, po)
            if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                kind = "oob"
            elif math.hypot(nx - CENTER, ny - CENTER) < SUN_R + 1.0:
                kind = "sun"
            elif near is not None and near[0] < 3.0:
                kind = "planet"
            else:
                kind = "lost"     # comet vanished under it, or moved planet
            deaths.append({"t": t, "owner": f[1], "ships": ships,
                           "kind": kind,
                           "pid": near[1] if kind == "planet" else None,
                           "tgt_owner": near[2] if kind == "planet" else None})

        agg = {o: {"planets": 0, "prod": 0, "garrison": 0,
                   "inflight": 0, "max_stack": 0, "n_fleets": 0}
               for o in (0, 1)}
        for p in planets:
            pid, owner, ships, prod = p[0], p[1], int(p[5]), int(p[6])
            if owner in agg:
                a = agg[owner]
                a["planets"] += 1
                a["prod"] += prod
                a["garrison"] += ships
            po = prev_owner.get(pid)
            if po is not None and po != owner and owner in (0, 1):
                captures.append({"t": t, "pid": pid, "from": po, "to": owner,
                                 "cost_garrison": prev_ships.get(pid, -1),
                                 "prod": prod})
            prev_owner[pid] = owner
            prev_ships[pid] = ships

        fids = set()
        prev_fleet_by_id = {f[0]: f for f in fleets}
        for f in fleets:
            fid, owner, ships = f[0], f[1], int(f[6])
            fids.add(fid)
            if owner in agg:
                a = agg[owner]
                a["inflight"] += ships
                a["n_fleets"] += 1
                if ships > a["max_stack"]:
                    a["max_stack"] = ships
            if fid not in prev_fids and owner in (0, 1):
                launches.append({"t": t, "owner": owner, "ships": ships})
        prev_fids = fids

        rows.append({"t": t,
                     "p0": agg[0], "p1": agg[1],
                     "total0": agg[0]["garrison"] + agg[0]["inflight"],
                     "total1": agg[1]["garrison"] + agg[1]["inflight"]})

    final = env.steps[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    n_l0 = sum(1 for l in launches if l["owner"] == 0)
    n_l1 = sum(1 for l in launches if l["owner"] == 1)

    # liveness asserts
    if len(env.steps) <= 30 or n_l0 == 0 or n_l1 == 0:
        print(json.dumps({"LIVENESS_FAIL": True, "n_steps": len(env.steps),
                          "launches0": n_l0, "launches1": n_l1,
                          "r0": r0, "r1": r1}))
        return 2

    out = {"seed": args.seed, "p0": args.p0, "p1": args.p1,
           "n_steps": len(env.steps), "r0": r0, "r1": r1,
           "launches0": n_l0, "launches1": n_l1,
           "rows": rows, "captures": captures, "launches": launches,
           "deaths": deaths}
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps({"seed": args.seed, "n_steps": len(env.steps),
                      "r0": r0, "r1": r1,
                      "launches": [n_l0, n_l1]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
