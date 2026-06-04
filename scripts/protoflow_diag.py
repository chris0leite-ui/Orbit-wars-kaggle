"""protoflow_diag — diagnose the mid/late-game collapse vs the Producer.

The probe shows we reach a strong position and still go 0/12. This walks the RAW
per-step game state (env.steps, absolute owners) to test one hypothesis: we lose by
REACTIVE WHACK-A-MOLE -- we out-build into a material lead, then the Producer
dictates a sequence of concentrated assaults we answer piecemeal, losing each
planet we reinforce and bleeding out. Metrics per game (focal = player 0):

  * material arc: our (garrison + in-flight) ships vs the Producer's, at our peak
    and at the end -- settles whether the drop is real combat loss or launch outflow.
  * territory arc: peak planet count -> final, and the collapse window length.
  * thrash: planets we sent a 'def' cohort to that we then LOST within 12 turns.
  * lead time: turns between an enemy fleet first AIMING at a planet and the planet
    flipping -- short lead = we cannot consolidate in time.
  * initiative: reactive ('def') vs offensive ('wave' on enemy) share of our launches
    in the last third, and whether the Producer's planet count ever dropped (did we
    ever make THEM react).

Usage:  python scripts/protoflow_diag.py [--seeds 4]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents.protoflow.main as proto  # noqa: E402
from fast import _load_callable  # noqa: E402

PRODUCER = "agents/producer/producer_agent.py"
THRASH_WINDOW = 12   # a planet lost within this many turns of a 'def' to it = thrash


def obs_of(step_state):
    """Player 0's observation dict from one env.steps entry."""
    o = step_state[0]["observation"]
    return o


def fleet_aims_at(fl, planets):
    """Cheap test: which planet (if any) a fleet is heading toward (nearest along its
    heading). fl = [id, owner, x, y, angle, from, ships]."""
    fx, fy, ang = float(fl[2]), float(fl[3]), float(fl[4])
    dx, dy = math.cos(ang), math.sin(ang)
    best, best_proj = None, 1e9
    for p in planets:
        px, py = float(p[2]) - fx, float(p[3]) - fy
        proj = px * dx + py * dy            # distance along heading
        if proj <= 0:
            continue
        perp = abs(px * dy - py * dx)        # offset from the ray
        if perp <= float(p[4]) + 1.5 and proj < best_proj:
            best, best_proj = int(p[0]), proj
    return best


def run_one(seed: int):
    from kaggle_environments import make
    proto.reset_trace()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([proto.agent, _load_callable(str(REPO / PRODUCER))])
    trace = proto.get_trace()

    me, opp = 0, 1
    steps = env.steps
    series = []                       # per step: (step, my_mat, opp_mat, my_planets, opp_planets)
    owner_at = {}                     # pid -> owner, previous step
    first_aim = {}                    # pid -> earliest step an enemy fleet aimed at it
    losses = []                       # (pid, step_lost, lead_time)
    opp_planets_min = 99

    for i, st in enumerate(steps):
        o = obs_of(st)
        planets = o["planets"]
        fleets = o.get("fleets", [])
        my_mat = sum(int(p[5]) for p in planets if int(p[1]) == me) + \
                 sum(int(f[6]) for f in fleets if int(f[1]) == me)
        opp_mat = sum(int(p[5]) for p in planets if int(p[1]) == opp) + \
                  sum(int(f[6]) for f in fleets if int(f[1]) == opp)
        myp = sum(1 for p in planets if int(p[1]) == me)
        opc = sum(1 for p in planets if int(p[1]) == opp)
        my_prod = sum(int(p[6]) for p in planets if int(p[1]) == me)
        opp_prod = sum(int(p[6]) for p in planets if int(p[1]) == opp)
        opp_planets_min = min(opp_planets_min, opc)
        series.append((i, my_mat, opp_mat, myp, opc, my_prod, opp_prod))

        # enemy fleets aiming at our planets -> earliest aim time (lead-time clock)
        my_pids = {int(p[0]) for p in planets if int(p[1]) == me}
        for f in fleets:
            if int(f[1]) != opp:
                continue
            tgt = fleet_aims_at(f, planets)
            if tgt in my_pids and tgt not in first_aim:
                first_aim[tgt] = i

        # ownership flips me -> opp
        for p in planets:
            pid, own = int(p[0]), int(p[1])
            prev = owner_at.get(pid)
            if prev == me and own == opp:
                lead = i - first_aim.get(pid, i)
                losses.append((pid, i, lead))
            owner_at[pid] = own

    # peak territory + collapse window
    peak_i, peak_planets = max(((s[0], s[3]) for s in series), key=lambda x: x[1])
    peak_mat = next(s[1] for s in series if s[0] == peak_i)
    peak_opp_mat = next(s[2] for s in series if s[0] == peak_i)
    peak_row = next(s for s in series if s[0] == peak_i)
    peak_opc, peak_myprod, peak_oppprod = peak_row[4], peak_row[5], peak_row[6]
    # earliest step our controlled production falls behind the Producer's, and by when
    behind_i = next((s[0] for s in series if s[6] > s[5]), None)
    final = series[-1]

    # thrash: 'def' launches per planet -> lost within window
    def_steps = defaultdict(list)
    for t in trace:
        for lc in t["launches"]:
            if lc.get("kind") == "def":
                def_steps[lc["tgt"]].append(t["step"])
    thrash = 0
    for pid, slost, _lead in losses:
        if any(0 <= slost - ds <= THRASH_WINDOW for ds in def_steps.get(pid, [])):
            thrash += 1

    # reactive vs offensive share in the last third
    n = len(trace)
    late = trace[int(n * 2 / 3):]
    late_def = late_off = 0
    for t in late:
        for lc in t["launches"]:
            if lc.get("kind") == "def":
                late_def += 1
            elif lc.get("kind") == "wave" and lc.get("tgt_owner") not in (-1,):
                late_off += 1

    won = final[3] > final[4]
    leads = [l for _, _, l in losses]
    return {
        "seed": seed, "won": won,
        "peak_i": peak_i, "peak_planets": peak_planets, "final_planets": final[3],
        "peak_mat": peak_mat, "peak_opp_mat": peak_opp_mat,
        "final_mat": final[1], "final_opp_mat": final[2],
        "collapse_window": final[0] - peak_i,
        "n_lost": len(losses), "thrash": thrash,
        "med_lead": (sorted(leads)[len(leads) // 2] if leads else None),
        "late_def": late_def, "late_off": late_off,
        "opp_planets_min": opp_planets_min,
        "n_steps": final[0],
        "peak_opc": peak_opc, "peak_myprod": peak_myprod, "peak_oppprod": peak_oppprod,
        "behind_i": behind_i,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    rows = [run_one(s) for s in range(args.seeds)]
    print(f"\nprotoflow vs Producer — collapse diagnosis ({args.seeds} seeds, focal=P0)\n")
    hdr = ("seed res | planets us/opp@peak | prod us/opp@peak | behind@ | "
           "mat us/opp@peak | lost(thrash) | late off/def")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        res = "W" if r["won"] else "L"
        print(f"{r['seed']:>4} {res}  | "
              f"{r['peak_planets']:>2}/{r['peak_opc']:<2}@{r['peak_i']:>3}        | "
              f"{r['peak_myprod']:>3}/{r['peak_oppprod']:<3}        | "
              f"{str(r['behind_i']):>4}   | "
              f"{r['peak_mat']:>4}/{r['peak_opp_mat']:<4}      | "
              f"{r['n_lost']:>2}({r['thrash']:>2})      | "
              f"{r['late_off']:>3}/{r['late_def']:<3}")
    # aggregate read
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    behinds = [r["behind_i"] for r in rows if r["behind_i"] is not None]
    print(f"\nmean: peak planets us/opp={avg('peak_planets'):.1f}/{avg('peak_opc'):.1f}  "
          f"prod us/opp={avg('peak_myprod'):.0f}/{avg('peak_oppprod'):.0f}  "
          f"mat us/opp={avg('peak_mat'):.0f}/{avg('peak_opp_mat'):.0f}  "
          f"fell behind by step~{(sum(behinds)/len(behinds)):.0f}  "
          f"lost={avg('n_lost'):.1f} thrash={avg('thrash'):.1f}  "
          f"late off/def={avg('late_off'):.1f}/{avg('late_def'):.1f}")
    print("\nRead: peak_mat us>opp but final us<opp -> we out-build then lose the war; "
          "high thrash -> we reinforce planets we then lose (whack-a-mole); "
          "low medLead -> assaults commit too late for us to consolidate; "
          "late off~0 & oppMin~=start -> we never seize initiative (Producer never reacts).")


if __name__ == "__main__":
    main()
