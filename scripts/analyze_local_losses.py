"""Analyze LOCAL loss replays vs a (weaker) opponent — selection-bias-free.
Reuses episode_postmortem.attribute_fleets to classify our launched fleets,
and adds the audit's phase planet-count timeline. Run on a --save-replays dir."""
import sys, glob, os, json
sys.path.insert(0, "scripts")
from episode_postmortem import attribute_fleets

def planets_owned(obs, pid):
    return sum(1 for p in obs.get("planets", []) if p[1] == pid)

def prod_owned(obs, pid):
    return sum(p[6] for p in obs.get("planets", []) if p[1] == pid)

def global_obs(steps, t):
    for seat in range(len(steps[t])):
        if steps[t][seat].get("status") == "ACTIVE":
            return steps[t][seat]["observation"]
    return steps[t][0]["observation"]

def our_launch_count(steps, our_seat):
    n = 0
    for t in range(len(steps)):
        a = steps[t][our_seat].get("action")
        if a:
            n += len([x for x in a if isinstance(x, list) and len(x) >= 3])
    return n

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lossrep_v4p"
files = sorted(glob.glob(os.path.join(d, "*.json")))
wins = losses = 0
loss_files = []
for f in files:
    r = json.load(open(f))
    names = r["info"]["TeamNames"]
    our_seat = 0 if "state_driven" in names[0] else 1
    rew = r["rewards"]
    our_r, opp_r = rew[our_seat], rew[1 - our_seat]
    if our_r is None or opp_r is None:
        continue
    if our_r > opp_r:
        wins += 1
    elif our_r < opp_r:
        losses += 1
        loss_files.append((f, our_seat))

print(f"\n==== {os.path.basename(d)}: {wins}W / {losses}L  ({len(files)} games) ====\n")

for f, our_seat in loss_files:
    r = json.load(open(f))
    steps = r["steps"]
    n = len(steps)
    opp_seat = 1 - our_seat
    o0 = global_obs(steps, 0)
    our_pid = o0.get("player", our_seat)
    # the owner id we appear as in planet/fleet rows == our_seat for 2P
    our_pid = our_seat
    # phase timeline of planet count + production
    marks = [0, min(50, n-1), min(100, n-1), min(150, n-1), n-1]
    print(f"--- {os.path.basename(f)[:46]}  (our_seat={our_seat}, {n} steps) ---")
    print("   step :  ourPlanets/oppPlanets   ourProd/oppProd")
    for t in marks:
        ob = global_obs(steps, t)
        print(f"   {t:4d} :  {planets_owned(ob,our_pid):2d}/{planets_owned(ob,opp_seat):<2d}"
              f"              {prod_owned(ob,our_pid):5.0f}/{prod_owned(ob,opp_seat):<5.0f}")
    # captures/losses by phase (owner transitions of any planet)
    def transitions(t0, t1):
        cap = lost = 0
        prev = {p[0]: p[1] for p in global_obs(steps, t0).get("planets", [])}
        for t in range(t0+1, t1+1):
            cur = {p[0]: p[1] for p in global_obs(steps, t).get("planets", [])}
            for pid, ow in cur.items():
                pv = prev.get(pid)
                if pv != ow:
                    if ow == our_pid and pv != our_pid: cap += 1
                    if pv == our_pid and ow != our_pid: lost += 1
            prev = cur
        return cap, lost
    for (lo, hi, lab) in [(0,min(50,n-1),"open 0-50"),(min(50,n-1),min(150,n-1),"mid 50-150"),(min(150,n-1),n-1,"late 150+")]:
        if hi > lo:
            c,l = transitions(lo,hi)
            print(f"     {lab:12} captured {c:2d}  lost {l:2d}  (net {c-l:+d})")
    # fleet outcome buckets
    fleets = attribute_fleets(r, our_seat, our_pid)
    from collections import Counter
    bk = Counter(fl["outcome"] for fl in fleets)
    launched = our_launch_count(steps, our_seat)
    print(f"   fleets launched(actions)={launched}  tracked={len(fleets)}  buckets={dict(bk)}")
    fob = global_obs(steps, n-1)
    print(f"   FINAL: ours {planets_owned(fob,our_pid)}pl/{prod_owned(fob,our_pid):.0f}prod  vs opp {planets_owned(fob,opp_seat)}pl/{prod_owned(fob,opp_seat):.0f}prod\n")
