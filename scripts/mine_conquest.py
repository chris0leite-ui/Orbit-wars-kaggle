"""Who do we capture from, and who does the winner capture from?

For each 4P episode, every time a planet changes hands to a HUMAN player we
record (capturer, prior_owner, prior_owner_shipRank_at_that_step, step).
shipRank 1 = strongest of the 4 players by total ships at that moment.

Question: in losses, are WE aimed at the strongest enemy (kingmaker tax)
while the eventual winner feeds on the weakest?
"""
import glob, json, statistics, sys, collections

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "audit/live-episodes/53588922"
TEAM = "ChrisLeiteScha"

def player_ships(obs):
    tot = [0.0]*4
    for p in obs.get("planets") or []:
        o=int(p[1])
        if 0<=o<4: tot[o]+=float(p[5])
    for f in obs.get("fleets") or []:
        o=int(f[1])
        if 0<=o<4: tot[o]+=float(f[6])
    return tot

def rank_of(tot, who):
    # rank 1 = most ships
    order = sorted(range(4), key=lambda i:-tot[i])
    return order.index(who)+1

def analyze(d):
    info=d.get("info",{}); teams=info.get("TeamNames") or []
    if len(d["steps"][0])!=4: return None
    seats=[i for i,t in enumerate(teams) if t==TEAM]
    if len(seats)!=1: return None
    me=seats[0]; rw=d.get("rewards") or []
    if not rw or any(r is None for r in rw): return None
    mx=max(rw); won = rw[me]==mx and rw.count(mx)==1
    winner = rw.index(mx) if rw.count(mx)==1 else None
    steps=d["steps"]; prev=None
    # capturer -> list of prior-owner ranks (enemy captures only)
    cap_ranks=collections.defaultdict(list)
    neut=collections.Counter()
    for t,s in enumerate(steps):
        obs=s[0]["observation"]
        tot=player_ships(obs)
        planets={int(p[0]):int(p[1]) for p in obs.get("planets") or []}
        if prev is not None:
            for pid,o in planets.items():
                if pid in prev and prev[pid]!=o and 0<=o<4:
                    po=prev[pid]
                    if po==-1:
                        neut[o]+=1
                    elif 0<=po<4:
                        cap_ranks[o].append(rank_of(tot,po))
        prev=planets
    return dict(me=me,winner=winner,won=won,cap_ranks=cap_ranks,neut=neut)

losses_me=[]; losses_win=[]; wins_me=[]; neut_me_L=[]; neut_win_L=[]
for f in sorted(glob.glob(CORPUS+"/episode-*-replay.json")):
    r=analyze(json.load(open(f)))
    if r is None: continue
    me,winner=r["me"],r["winner"]
    if r["won"]:
        wins_me+= r["cap_ranks"][me]
    else:
        losses_me += r["cap_ranks"][me]
        if winner is not None:
            losses_win += r["cap_ranks"][winner]
            neut_me_L.append(r["neut"][me]); neut_win_L.append(r["neut"][winner])

def dist(xs):
    if not xs: return "none"
    c=collections.Counter(xs)
    return f"n={len(xs)} mean_rank={statistics.mean(xs):.2f}  rank-counts(1=strongest..4): " + \
           " ".join(f"{k}:{c.get(k,0)}" for k in (1,2,3,4))

print(f"=== {CORPUS} ===")
print("ENEMY captures — in LOSSES, by US     :", dist(losses_me))
print("ENEMY captures — in LOSSES, by WINNER :", dist(losses_win))
print("ENEMY captures — in WINS,   by US     :", dist(wins_me))
print(f"NEUTRAL captures in losses: us median={statistics.median(neut_me_L):.0f}  winner median={statistics.median(neut_win_L):.0f}  (n={len(neut_me_L)} games)")
