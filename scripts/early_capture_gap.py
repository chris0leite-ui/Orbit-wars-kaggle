"""Early-capture-opportunity gate (2026-06-02) — decides the proactive-garrison axis.

The PI hypothesis: in the opening, idle garrison ships are wasted tempo that
should be expanding. The counter-hypothesis (H1 audit): early holding is RATIONAL
stockpiling — neutrals cost ~43-45 ships, so a single planet must accumulate
before it can afford the next capture; a too-small fleet just bounces.

This script decides between them on EXISTING replays. For early steps it counts
DECLINED AFFORDABLE CAPTURES: an uncaptured neutral that (a) a single friendly
source already has enough idle ships to take (idle >= cost), (b) is reachable
within HORIZON turns, and (c) is NOT already covered by an inbound friendly fleet
-- yet we left the ships idle instead of launching.

  declined ~ 0  -> early hold is correct stockpiling/no-affordable-target -> AXIS MOOT
  declined > 0  -> genuine early under-expansion -> the PI lever has a real target

'us' = state_driven champion (same convention as early_garrison / analyze_local_losses).
Schema: planet [id,owner,x,y,angle,ships,prod]; fleet [id,owner,x,y,angle,from,ships].
Usage: python scripts/early_capture_gap.py [replay_dir ...]
"""
import sys, glob, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.fleet import eta_turns

P_OWNER, P_X, P_Y, P_SHIPS = 1, 2, 3, 5
F_OWNER, F_X, F_Y, F_SHIPS = 1, 2, 3, 6
NEUTRAL = -1
HORIZON = 25          # early reach window (turns)
COVER_FRAC = 0.5      # an inbound friendly fleet >= COVER_FRAC*cost nearest a neutral "covers" it
EARLY = list(range(1, 21))


def aobs(steps, t):
    for s in steps[t]:
        if s.get("status") == "ACTIVE":
            return s["observation"]
    return steps[t][0]["observation"]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def us_seat(names):
    return 0 if "state_driven" in names[0] else 1


def won(steps, us):
    """True if champion (us) won. Prefer reward; fall back to planet+ship count."""
    last = steps[-1]
    try:
        r = [last[i].get("reward") for i in range(len(last))]
        if r[us] is not None and r[1 - us] is not None and r[us] != r[1 - us]:
            return r[us] > r[1 - us]
    except Exception:
        pass
    ob = last[0]["observation"]
    mine = sum(1 for p in ob["planets"] if p[P_OWNER] == us)
    opp = sum(1 for p in ob["planets"] if p[P_OWNER] == 1 - us)
    return mine > opp


def analyze_game(steps, us):
    """Return per-game means over early steps: declined captures (covered-aware
    and raw), max single-source idle ships, nearest-neutral cost, idle total."""
    decl_cov, decl_raw, max_idle, near_cost, idle_tot, n = [], [], [], [], [], 0
    for t in EARLY:
        if t >= len(steps):
            break
        ob = aobs(steps, t)
        srcs = [(p[P_X], p[P_Y], p[P_SHIPS]) for p in ob["planets"]
                if p[P_OWNER] == us and p[P_SHIPS] > 0]
        neuts = [(p[P_X], p[P_Y], p[P_SHIPS] + 1) for p in ob["planets"] if p[P_OWNER] == NEUTRAL]
        if not srcs or not neuts:
            continue
        ff = [(f[F_X], f[F_Y], f[F_SHIPS]) for f in ob.get("fleets", []) if f[F_OWNER] == us]
        d_cov = d_raw = 0
        nearest_cost_this = min(
            (min(dist((sx, sy), (nx, ny)) for sx, sy, _ in srcs), c)
            for nx, ny, c in neuts
        )[1] if neuts else float("nan")
        for nx, ny, cost in neuts:
            # affordable from at least one source, reachable within HORIZON
            ok = False
            for sx, sy, sh in srcs:
                if sh >= cost and eta_turns((sx, sy), (nx, ny), sh) <= HORIZON:
                    ok = True
                    break
            if not ok:
                continue
            d_raw += 1
            # covered? an inbound friendly fleet of decent size nearest THIS neutral
            covered = False
            for fx, fy, fsh in ff:
                if fsh < COVER_FRAC * cost:
                    continue
                dn = dist((fx, fy), (nx, ny))
                # nearest planet to the fleet is this neutral -> treat as en route here
                others = min(dist((fx, fy), (px, py)) for px, py, _ in
                             [(p[P_X], p[P_Y], 0) for p in ob["planets"]])
                if dn <= others + 1e-6:
                    covered = True
                    break
            if not covered:
                d_cov += 1
        decl_cov.append(d_cov)
        decl_raw.append(d_raw)
        max_idle.append(max(sh for _, _, sh in srcs))
        idle_tot.append(sum(sh for _, _, sh in srcs))
        near_cost.append(nearest_cost_this)
        n += 1
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return dict(decl_cov=m(decl_cov), decl_raw=m(decl_raw), max_idle=m(max_idle),
                near_cost=m(near_cost), idle_tot=m(idle_tot), steps=n,
                any_decl=1.0 if any(x > 0 for x in decl_cov) else 0.0)


def main():
    dirs = sys.argv[1:] or ["/tmp/rep_opening", "/tmp/rep_poscore"]
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("no replays in", dirs); return
    wins, losses = [], []
    for f in files:
        r = json.load(open(f))
        steps = r["steps"]
        us = us_seat(r["info"]["TeamNames"])
        g = analyze_game(steps, us)
        (wins if won(steps, us) else losses).append(g)

    def report(label, gs):
        if not gs:
            print(f"  {label:8s}: (none)"); return
        m = lambda k: sum(g[k] for g in gs) / len(gs)
        print(f"  {label:8s} (n={len(gs):3d}): "
              f"declined/step(cover-aware)={m('decl_cov'):.2f}  raw={m('decl_raw'):.2f}  "
              f"|  max_idle_source={m('max_idle'):5.1f}  nearest_neutral_cost={m('near_cost'):4.1f}  "
              f"idle_total={m('idle_tot'):6.1f}  |  games_with_any_decline={m('any_decl')*100:.0f}%")

    print(f"\n==== early_capture_gap  (steps 1-20, HORIZON={HORIZON})  {len(files)} games ====")
    print("  'declined' = affordable+reachable+uncovered neutral we left idle ships against.")
    report("ALL", wins + losses)
    report("WINS", wins)
    report("LOSSES", losses)
    allg = wins + losses
    m_all = sum(g["decl_cov"] for g in allg) / len(allg)
    print(f"\n  GATE: mean declined/step (cover-aware) = {m_all:.2f}")
    if m_all < 0.25:
        print("  -> NEGATIVE: early hold is correct stockpiling / no affordable target. AXIS MOOT.")
        print("     (max_idle_source vs nearest_neutral_cost shows whether we simply can't afford captures yet.)")
    else:
        print("  -> POSITIVE: idle ships sit while affordable reachable neutrals go untaken. Under-expansion.")
    print()


if __name__ == "__main__":
    main()
