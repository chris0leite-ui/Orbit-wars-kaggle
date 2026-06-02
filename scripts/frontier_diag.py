"""scripts/frontier_diag.py — frontier-push loss diagnosis.

Tests the PI's hypothesis (2026-06-02): we lose corner-start games because we
grab the big nearby planet, then UNDER-EXPAND — our ships sit idle on owned
planets instead of advancing to the contested frontier, so the opponent
snowballs by pushing its fleets to the frontline.

Operationalised per game, over the mid phase (default steps 30..90):
  - advance: how far a side's SHIP MASS (garrisons + in-flight fleets,
    ship-weighted centroid) has moved from its own home toward the enemy
    home, normalised so 0 = hugging home, 1 = sitting on the enemy home.
  - inflight_frac: fraction of a side's ships that are in fleets (moving)
    rather than garrisoned (idle).

Run on a clean_ab --save-replays dir. Splits games by whether the
state_driven champion ("us") won or lost and reports the wins-vs-losses gap.
If the hypothesis holds, losses show our advance << opp advance and a lower
our inflight_frac.

    python scripts/frontier_diag.py /tmp/rep_dir [--lo 30] [--hi 90]
"""
import sys, glob, os, json, math

# Planet row:  [id, owner, x, y, _, ships, prod]
# Fleet  row:  [id, owner, x, y, angle, from_planet_id, ships]
P_OWNER, P_X, P_Y, P_SHIPS = 1, 2, 3, 5
F_OWNER, F_X, F_Y, F_SHIPS = 1, 2, 3, 6


def global_obs(steps, t):
    for seat in range(len(steps[t])):
        if steps[t][seat].get("status") == "ACTIVE":
            return steps[t][seat]["observation"]
    return steps[t][0]["observation"]


def home_centroid(obs, pid):
    pts = [(p[P_X], p[P_Y], max(1.0, p[P_SHIPS])) for p in obs.get("planets", []) if p[P_OWNER] == pid]
    if not pts:
        return None
    w = sum(p[2] for p in pts)
    return (sum(p[0] * p[2] for p in pts) / w, sum(p[1] * p[2] for p in pts) / w)


def ship_mass(obs, pid):
    """Return (centroid_xy, garrison_ships, inflight_ships) for a side."""
    gx = gy = g = inf = 0.0
    cx = cy = w = 0.0
    for p in obs.get("planets", []):
        if p[P_OWNER] == pid:
            s = p[P_SHIPS]
            g += s
            cx += p[P_X] * s; cy += p[P_Y] * s; w += s
    for f in obs.get("fleets", []):
        if f[F_OWNER] == pid:
            s = f[F_SHIPS]
            inf += s
            cx += f[F_X] * s; cy += f[F_Y] * s; w += s
    centroid = (cx / w, cy / w) if w > 0 else None
    return centroid, g, inf


def advance(centroid, home, opp_home):
    """Projection of (centroid-home) onto the home->opp_home axis, normalised."""
    if centroid is None or home is None or opp_home is None:
        return None
    ax, ay = opp_home[0] - home[0], opp_home[1] - home[1]
    L = math.hypot(ax, ay)
    if L < 1e-6:
        return None
    vx, vy = centroid[0] - home[0], centroid[1] - home[1]
    return (vx * ax + vy * ay) / (L * L)


def analyze_game(r, lo, hi):
    steps = r["steps"]
    n = len(steps)
    names = r["info"]["TeamNames"]
    our_seat = 0 if "state_driven" in names[0] else 1
    our_pid, opp_pid = our_seat, 1 - our_seat
    rew = r["rewards"]
    if rew[our_seat] is None or rew[opp_seat := 1 - our_seat] is None:
        return None
    won = rew[our_seat] > rew[opp_pid]

    o0 = global_obs(steps, 0)
    our_home = home_centroid(o0, our_pid)
    opp_home = home_centroid(o0, opp_pid)

    our_adv, opp_adv, our_inf, opp_inf = [], [], [], []
    for t in range(lo, min(hi, n - 1) + 1, 5):
        ob = global_obs(steps, t)
        oc, og, oinf = ship_mass(ob, our_pid)
        pc, pg, pinf = ship_mass(ob, opp_pid)
        a_us = advance(oc, our_home, opp_home)
        a_op = advance(pc, opp_home, our_home)
        if a_us is not None: our_adv.append(a_us)
        if a_op is not None: opp_adv.append(a_op)
        if og + oinf > 0: our_inf.append(oinf / (og + oinf))
        if pg + pinf > 0: opp_inf.append(pinf / (pg + pinf))

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "won": won, "n_steps": n,
        "our_adv": mean(our_adv), "opp_adv": mean(opp_adv),
        "our_inf": mean(our_inf), "opp_inf": mean(opp_inf),
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rep_dir"
    lo = int(sys.argv[sys.argv.index("--lo") + 1]) if "--lo" in sys.argv else 30
    hi = int(sys.argv[sys.argv.index("--hi") + 1]) if "--hi" in sys.argv else 90
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    rows = [g for f in files if (g := analyze_game(json.load(open(f)), lo, hi))]
    wins = [g for g in rows if g["won"]]
    losses = [g for g in rows if not g["won"]]

    def agg(label, gs):
        if not gs:
            print(f"   {label:8} (n=0)"); return
        m = lambda k: sum(g[k] for g in gs) / len(gs)
        print(f"   {label:8} (n={len(gs):2d})  our_adv={m('our_adv'):+.3f}  opp_adv={m('opp_adv'):+.3f}  "
              f"adv_gap(us-opp)={m('our_adv')-m('opp_adv'):+.3f}   "
              f"our_inflight={m('our_inf'):.2f}  opp_inflight={m('opp_inf'):.2f}")

    print(f"\n==== frontier_diag {os.path.basename(d)}  (champion='us', steps {lo}..{hi}) ====")
    print(f"   games={len(rows)}  wins={len(wins)}  losses={len(losses)}")
    print("   advance: 0=hugging home, 0.5=at frontier, 1=on enemy home")
    agg("WINS", wins)
    agg("LOSSES", losses)
    print("   hypothesis: in LOSSES our_adv << opp_adv and our_inflight is lower\n")


if __name__ == "__main__":
    main()
