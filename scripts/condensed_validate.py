import sys, time
import numpy as np
from kaggle_environments import make
sys.path.insert(0,"scripts")
from condensed_policy import CondensedPolicy
from lib import fast_sim
from lib.opp_model import lite_greedy_policy, top_tier_mirror_policy as TEACHER

def winner(env, ns):
    sc=[0.0]*ns; obs=env.state[0].observation
    for p in obs["planets"]:
        o=int(p[1])
        if 0<=o<ns: sc[o]+=float(p[5])
    for f in obs.get("fleets",[]):
        o=int(f[1])
        if 0<=o<ns: sc[o]+=float(f[6])
    return int(np.argmax(sc))

def play(pols, ns):
    env=make("orbit_wars",configuration={"episodeSteps":500}); env.reset(num_agents=ns)
    for _ in range(500):
        if env.done: break
        env.step([pols[i](env.state[i].observation) for i in range(ns)])
    return winner(env, ns)

def h2h(opp, label, ns, n):
    w=0
    for g in range(n):
        seat=g%ns
        pols=[opp]*ns; pols[seat]=CondensedPolicy(num_seats=ns)
        if play(pols,ns)==seat: w+=1
    need = "50%" if ns==2 else "25%"
    print(f"  condensed vs {label} {ns}P: {w}/{n} = {100*w/n:.0f}% (beat>{need})", flush=True)

def speed():
    ns=4; env=make("orbit_wars",configuration={"episodeSteps":500}); env.reset(num_agents=ns)
    for _ in range(40):
        if env.done: break
        env.step([lite_greedy_policy(env.state[i].observation) for i in range(ns)])
    pol=CondensedPolicy(num_seats=ns); obs0=env.state[0].observation; pol(obs0)
    t=time.perf_counter()
    for _ in range(2000): pol(obs0)
    pc=(time.perf_counter()-t)/2000
    snap=fast_sim.from_obs(obs0, env.configuration, episode_seed=0, num_seats=ns)
    pols=[CondensedPolicy(num_seats=ns) for _ in range(ns)]; fast_sim.rollout(snap,50,pols)
    t=time.perf_counter()
    for _ in range(15): fast_sim.rollout(snap,50,pols)
    pr=(time.perf_counter()-t)/15
    print(f"  per-call {pc*1e6:.1f}us  K=50 rollout {pr*1000:.1f}ms -> {0.6/pr:.0f}/600ms "
          f"(refs: lite 5ms/120, top_tier 474ms/1.3)")

n=int(sys.argv[1]) if len(sys.argv)>1 else 24
print("=== SPEED ==="); speed()
print("=== STRENGTH ===")
t0=time.perf_counter()
h2h(lite_greedy_policy,"lite_greedy",2,n)
h2h(lite_greedy_policy,"lite_greedy",4,n)
h2h(TEACHER,"TEACHER",2,n)
h2h(TEACHER,"TEACHER",4,n)
print(f"  ({time.perf_counter()-t0:.0f}s)")
