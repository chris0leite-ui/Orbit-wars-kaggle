"""mine_4p_carving.py — deep-dive the 4P live-episode corpus.

For each 4P episode (focal = our single seat), walk the replay and extract,
per capture-of-our-planet event:
  - the garrison on the planet the step before it fell,
  - ships we launched OUT of that planet in the preceding DRAIN_WINDOW steps
    (self-drain signature: the planner stripped the garrison, then it fell),
  - which opponent took it.

Per episode: elimination step, distinct rivals who captured our planets in
the final phase, our attack-vs-reinforce launch split while our planet count
was declining, and the same stats for wins as contrast.

Planet row: [id, owner, x, y, radius, ships, production]
Fleet  row: [id, owner, x, y, angle, from_planet_id, ships]

Usage: python scripts/mine_4p_carving.py [corpus_dir]
"""
from __future__ import annotations

import glob
import json
import statistics
import sys

CORPUS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "audit/live-episodes/53384340"
)
TEAM = "ChrisLeiteScha"
DRAIN_WINDOW = 8      # steps before a capture in which our outflow counts
FINAL_PHASE = 40      # steps before elimination for the multi-front count


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def analyze_episode(d):
    info = d.get("info", {})
    teams = info.get("TeamNames") or []
    seats = [i for i, t in enumerate(teams) if t == TEAM]
    if len(seats) != 1:
        return None  # self-match or unknown
    me = seats[0]
    steps = d["steps"]
    n_players = len(steps[0])
    if n_players != 4:
        return None
    rewards = d.get("rewards") or [None] * n_players
    if any(r is None for r in rewards):
        return None
    won = rewards[me] == max(rewards) and rewards.count(max(rewards)) == 1

    # Per-step board state from seat-0 observation (full board).
    boards = []     # list of dict planet_id -> (owner, ships)
    new_fleets = [] # per step: list of (owner, from_pid, ships)
    seen_fleet_ids = set()
    for s in steps:
        obs = s[0].get("observation", {})
        planets = obs.get("planets") or []
        boards.append({int(p[0]): (int(p[1]), float(p[5])) for p in planets})
        nf = []
        for f in obs.get("fleets") or []:
            fid = int(f[0])
            if fid not in seen_fleet_ids:
                seen_fleet_ids.add(fid)
                nf.append((int(f[1]), int(f[5]), float(f[6])))
        new_fleets.append(nf)

    T = len(boards)

    def my_planets(t):
        return [pid for pid, (o, _s) in boards[t].items() if o == me]

    # elimination step: last step where we own >= 1 planet.
    elim = None
    for t in range(T - 1, -1, -1):
        if my_planets(t):
            elim = t + 1 if t + 1 < T else None
            break
    eliminated = elim is not None and not my_planets(T - 1)

    peak_count = max(len(my_planets(t)) for t in range(T))
    peak_step = max(range(T), key=lambda t: len(my_planets(t)))

    # outflow[t][pid] = ships launched by us FROM pid at step t
    outflow = [dict() for _ in range(T)]
    for t, nf in enumerate(new_fleets):
        for owner, from_pid, ships in nf:
            if owner == me:
                outflow[t][from_pid] = outflow[t].get(from_pid, 0.0) + ships

    # capture events of OUR planets
    captures = []  # (t, pid, garrison_before, recent_outflow, taker)
    for t in range(1, T):
        prev, cur = boards[t - 1], boards[t]
        for pid, (o_prev, s_prev) in prev.items():
            if o_prev != me:
                continue
            if pid not in cur:
                continue
            o_cur = cur[pid][0]
            if o_cur != me:
                recent = sum(
                    outflow[u].get(pid, 0.0)
                    for u in range(max(0, t - DRAIN_WINDOW), t)
                )
                captures.append((t, pid, s_prev, recent, o_cur))

    end_step = (T - 1) if not eliminated else (elim if elim is not None else T - 1)
    final_lo = max(0, end_step - FINAL_PHASE)
    final_caps = [c for c in captures if c[0] >= final_lo]
    rivals_final = {c[4] for c in final_caps if c[4] >= 0 and c[4] != me}

    # launch split while declining (after peak): ships sent to planets we own
    # at launch time = reinforce/regroup; ships from planets... we don't know
    # targets, so proxy: outflow from FRONT planets vs total. Instead report
    # total outflow after peak and how much of it left planets that later fell
    # within DRAIN_WINDOW.
    out_after_peak = 0.0
    out_lost_soon = 0.0
    fell_at = {}
    for t, pid, *_ in captures:
        fell_at.setdefault(pid, []).append(t)
    for t in range(peak_step, T):
        for pid, ships in outflow[t].items():
            out_after_peak += ships
            if any(t < ft <= t + DRAIN_WINDOW for ft in fell_at.get(pid, [])):
                out_lost_soon += ships

    drained = [c for c in captures if c[3] >= 5.0]
    return {
        "won": won,
        "eliminated": eliminated,
        "elim_step": end_step if eliminated else None,
        "n_steps": T,
        "peak_count": peak_count,
        "peak_step": peak_step,
        "n_captures_of_ours": len(captures),
        "n_drained_captures": len(drained),
        "garrison_at_fall": [c[2] for c in captures],
        "recent_outflow_at_fall": [c[3] for c in captures],
        "rivals_final": len(rivals_final),
        "out_after_peak": out_after_peak,
        "out_lost_soon": out_lost_soon,
    }


def main():
    files = sorted(glob.glob(f"{CORPUS}/episode-*-replay.json"))
    losses, wins = [], []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        r = analyze_episode(d)
        if r is None:
            continue
        (wins if r["won"] else losses).append(r)

    def report(tag, group):
        if not group:
            print(f"{tag}: none")
            return
        n = len(group)
        caps = sum(g["n_captures_of_ours"] for g in group)
        drained = sum(g["n_drained_captures"] for g in group)
        garr = [x for g in group for x in g["garrison_at_fall"]]
        outf = [x for g in group for x in g["recent_outflow_at_fall"]]
        elim = [g["elim_step"] for g in group if g["eliminated"]]
        rivals = [g["rivals_final"] for g in group]
        oap = sum(g["out_after_peak"] for g in group)
        ols = sum(g["out_lost_soon"] for g in group)
        print(f"\n== {tag} (n={n}) ==")
        print(f"  eliminated: {sum(g['eliminated'] for g in group)}/{n}"
              f"  median elim step: {median(elim):.0f}" if elim else
              f"  eliminated: 0/{n}")
        print(f"  captures of our planets: {caps} total ({caps/n:.1f}/game)")
        print(f"  ...where we had launched >=5 ships OUT within {DRAIN_WINDOW} "
              f"steps before the fall: {drained} ({100*drained/max(caps,1):.0f}%)")
        print(f"  median garrison the step before a fall: {median(garr):.1f}")
        print(f"  median our-outflow in the {DRAIN_WINDOW} steps before a fall: "
              f"{median(outf):.1f}")
        print(f"  distinct rivals capturing us in final {FINAL_PHASE} steps: "
              f"median {median(rivals):.0f}  (>=2 in "
              f"{sum(1 for r in rivals if r >= 2)}/{n})")
        print(f"  ships launched out after our peak: {oap:.0f}; of those, "
              f"launched from planets that fell within {DRAIN_WINDOW} steps: "
              f"{ols:.0f} ({100*ols/max(oap,1):.0f}%)")

    report("4P LOSSES", losses)
    report("4P WINS", wins)


if __name__ == "__main__":
    main()
