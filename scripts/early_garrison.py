"""Early-game garrison observer — tests the PI hypothesis (2026-06-02):
in the first ~10-20 steps the opponent is too far to threaten us, so any
ships HELD on our planets (garrison) instead of launched are wasted tempo.

Reports, averaged over all games in a --save-replays dir, for early steps:
  g_frac = garrison_ships / (garrison + inflight)   (1.0 = all ships idle)
  idle   = mean garrison ships on our planets
  nearEnemy = mean nearest distance from any OUR planet to any enemy
              unit (planet or fleet); home0 = step-0 home-to-home distance.
'us' = state_driven champion (same convention as analyze_local_losses).
"""
import sys, glob, os, json, math
P_OWNER,P_X,P_Y,P_SHIPS = 1,2,3,5
F_OWNER,F_X,F_Y,F_SHIPS = 1,2,3,6

def gobs(steps,t):
    for s in range(len(steps[t])):
        if steps[t][s].get("status")=="ACTIVE": return steps[t][s]["observation"]
    return steps[t][0]["observation"]

def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

d = sys.argv[1] if len(sys.argv)>1 else "/tmp/rep_opening"
files = sorted(glob.glob(os.path.join(d,"*.json")))
STEPS=[1,2,3,5,8,12,16,20]
acc={t:{"g":[],"idle":[],"ne":[]} for t in STEPS}
home0=[]
for f in files:
    r=json.load(open(f)); steps=r["steps"]; n=len(steps)
    names=r["info"]["TeamNames"]; us=0 if "state_driven" in names[0] else 1; opp=1-us
    o0=gobs(steps,0)
    ours0=[(p[P_X],p[P_Y]) for p in o0["planets"] if p[P_OWNER]==us]
    opps0=[(p[P_X],p[P_Y]) for p in o0["planets"] if p[P_OWNER]==opp]
    if ours0 and opps0: home0.append(dist(ours0[0],opps0[0]))
    for t in STEPS:
        if t>=n: continue
        ob=gobs(steps,t)
        garr=sum(p[P_SHIPS] for p in ob["planets"] if p[P_OWNER]==us)
        inf=sum(f[F_SHIPS] for f in ob.get("fleets",[]) if f[F_OWNER]==us)
        ourpl=[(p[P_X],p[P_Y]) for p in ob["planets"] if p[P_OWNER]==us]
        enemy=[(p[P_X],p[P_Y]) for p in ob["planets"] if p[P_OWNER]==opp]+\
              [(fl[F_X],fl[F_Y]) for fl in ob.get("fleets",[]) if fl[F_OWNER]==opp]
        if garr+inf>0: acc[t]["g"].append(garr/(garr+inf))
        acc[t]["idle"].append(garr)
        if ourpl and enemy:
            acc[t]["ne"].append(min(dist(a,b) for a in ourpl for b in enemy))

m=lambda xs: sum(xs)/len(xs) if xs else float('nan')
print(f"\n==== early_garrison {os.path.basename(d)}  ({len(files)} games) ====")
print(f"   step :  garrison_frac   idle_ships   nearestEnemyDist   (home0={m(home0):.0f})")
for t in STEPS:
    print(f"   {t:4d} :     {m(acc[t]['g']):.2f}          {m(acc[t]['idle']):5.1f}          {m(acc[t]['ne']):6.1f}")
print("   hypothesis: g_frac stays HIGH while nearestEnemy ~ home0 (enemy unreachable) = wasted hold\n")
