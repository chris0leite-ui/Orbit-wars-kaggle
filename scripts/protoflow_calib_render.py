"""protoflow_calib_render — draw the synthetic calibration scenarios as board maps.

Each calibration "test" is a single hand-built board where we read the agent's ONE
decision (which fleets it launches). This renders that board the way you'd see it on the
map -- planets sized by production and coloured by owner (blue=us, red=enemy, grey=neutral)
-- with an arrow for every fleet the agent launches, labelled with ship count and kind.
Single-turn snapshots, not game replays.

Run:  python scripts/protoflow_calib_render.py   ->  writes PNGs to /tmp/calib_*.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

import agents.protoflow.main as proto


def radius(prod):
    return 1.0 + math.log(prod)


OWNER_COLOR = {0: "#2c6fbb", -1: "#b0b0b0"}  # us=blue, neutral=grey; enemies=red below
KIND_COLOR = {"wave": "#1a9850", "flip": "#f1a340", "def": "#7b3294",
              "regroup": "#999999", "default": "#222222"}


def make_obs(planets, fleets=None, player=0, step=20, omega=0.0):
    for p in planets:  # no planet may sit inside the sun (r=10 @ (50,50))
        if math.hypot(float(p[2]) - 50.0, float(p[3]) - 50.0) < 10.0:
            raise ValueError(f"planet {p[0]} at ({p[2]},{p[3]}) is inside the sun")
    return {"player": player, "planets": [list(p) for p in planets],
            "fleets": [list(f) for f in (fleets or [])], "angular_velocity": omega,
            "comet_planet_ids": [], "comets": [], "step": step, "remainingOverageTime": 60.0}


def render(name, planets, note, out, fleets=None, flags=None):
    # apply any flag overrides for this scenario, run the agent, capture launches
    saved = {}
    for k, v in (flags or {}).items():
        saved[k] = getattr(proto, k)
        setattr(proto, k, v)
    proto.reset_trace()
    proto.agent(make_obs(planets, fleets=fleets))
    launches = proto.get_trace()[-1]["launches"]
    for k, v in saved.items():
        setattr(proto, k, v)

    pby = {int(p[0]): p for p in planets}
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Circle((50, 50), 10, color="#ffcc66", alpha=0.55, zorder=0))  # sun
    ax.text(50, 50, "sun", ha="center", va="center", fontsize=8, color="#aa7700", zorder=1)

    for p in planets:
        pid, owner, x, y, _r, ships, prod = int(p[0]), int(p[1]), float(p[2]), float(p[3]), p[4], int(p[5]), int(p[6])
        color = OWNER_COLOR.get(owner, "#d6311f")  # any positive owner = enemy red
        rad = 1.5 + prod  # size by production (schematic, not to scale)
        ax.add_patch(Circle((x, y), rad, facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.9, zorder=2))
        ax.text(x, y - rad - 1.6, f"#{pid}  p{prod} s{ships}", ha="center", va="top", fontsize=7.5, zorder=3)

    for lc in launches:
        s, t = pby[int(lc["src"])], pby[int(lc["tgt"])]
        col = KIND_COLOR.get(lc["kind"], KIND_COLOR["default"])
        ax.add_patch(FancyArrowPatch((float(s[2]), float(s[3])), (float(t[2]), float(t[3])),
                     arrowstyle="-|>", mutation_scale=18, lw=2.2, color=col,
                     shrinkA=10, shrinkB=12, zorder=4, alpha=0.9))
        mx, my = (float(s[2]) + float(t[2])) / 2, (float(s[3]) + float(t[3])) / 2
        ax.text(mx, my, f"{lc['ships']} ({lc['kind']})", color=col, fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.85))

    summary = "HOLD (no launch)" if not launches else \
        "  ".join(f"#{l['src']}->#{l['tgt']}: {l['ships']} {l['kind']}" for l in launches)
    ax.set_title(f"{name}\n{note}\n=> {summary}", fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.invert_yaxis()  # origin top-left, as Kaggle
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}  ({summary})")


def main():
    # --- S14: springboard -- cluster-adjacent beats isolated ---
    render("S14 springboard (cluster vs isolated)",
           [[0,0,15,15,radius(3),14,3],[1,-1,15,40,radius(3),6,3],[2,-1,40,15,radius(3),6,3],
            [3,-1,58,15,radius(3),6,3],[4,-1,50,6,radius(3),6,3],[5,-1,40,4,radius(3),6,3]],
           "we fire at a cluster planet (springboard), not the isolated #1",
           "/tmp/calib_S14_springboard.png")

    # --- S16: production leads -- big isolated vs small-in-cluster, sum vs bounded ---
    b16 = [[0,0,8,50,radius(3),40,3],[1,-1,33,50,radius(5),6,5],[2,-1,8,78,radius(1),6,1],
           [3,-1,20,80,radius(1),6,1],[4,-1,2,88,radius(1),6,1],[5,-1,18,70,radius(1),6,1],
           [6,-1,2,68,radius(1),6,1],[7,-1,22,72,radius(1),6,1],[8,-1,12,90,radius(1),6,1],
           [9,-1,28,84,radius(1),6,1]]
    render("S16 OLD (springboard summed)", b16,
           "saturated springboard -> chase the small clustered #2 over the big #1",
           "/tmp/calib_S16a_sum.png", flags={"SPRINGBOARD_TOPK": 0})
    render("S16 NEW (springboard bounded)", b16,
           "bounded springboard -> production leads, fire the big planet #1",
           "/tmp/calib_S16b_bounded.png", flags={"SPRINGBOARD_TOPK": 2})

    # --- S16b: stepping-stone to big still rewarded ---
    render("S16b stepping-stone",
           [[0,0,15,50,radius(3),14,3],[1,-1,30,30,radius(1),6,1],[2,-1,42,22,radius(5),6,5],
            [3,-1,30,70,radius(1),6,1],[4,-1,42,78,radius(1),6,1]],
           "#1 (opens big #2) is preferred over #3 (opens only small #4)",
           "/tmp/calib_S16b_stepping.png")

    # --- S17: offensive pressure -- hub vs outpost, offense off vs on ---
    b17 = [[0,0,15,15,radius(3),60,3],[1,1,15,45,radius(3),10,3],[2,1,45,15,radius(3),10,3],
           [3,0,5,55,radius(3),20,3],[4,0,8,33,radius(3),20,3],[5,0,26,52,radius(3),20,3]]
    render("S17 offense OFF", b17,
           "cluster #3/#4 already coalition-attack the hub #1; hub value ~46",
           "/tmp/calib_S17a_off.png", flags={"OFFENSIVE_PRESSURE": False})
    render("S17 offense ON (collapse the opponent's reach)", b17,
           "same attack, but hub value lifted to ~56 (denying the opponent's region into us)",
           "/tmp/calib_S17b_on.png", flags={"OFFENSIVE_PRESSURE": True})

    # --- S2: convergence -- two sources form a coalition on one target ---
    render("S2 coalition (two sources, one arrival)",
           [[0,0,15,30,radius(3),18,3],[1,0,15,8,radius(3),18,3],[2,-1,40,19,radius(4),26,4],
            [3,1,88,80,radius(3),20,3]],
           "neither home can solo #2; the two converge on the same arrival turn",
           "/tmp/calib_S2_coalition.png")

    # --- S7a: holdable up-size next to a strong enemy ---
    render("S7a holdable up-size",
           [[0,0,15,30,radius(3),90,3],[1,-1,33,30,radius(2),6,3],[2,1,41,30,radius(1),40,1]],
           "capture #1 sized ABOVE the flip floor to survive the enemy #2 counter",
           "/tmp/calib_S7a_holdable.png")


if __name__ == "__main__":
    main()
