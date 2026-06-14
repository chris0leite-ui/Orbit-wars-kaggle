"""Keystone go/no-go: is the distilled policy (a) fast and (b) stronger than
lite_greedy? Both gates must pass or the search moonshot is infeasible."""
import sys, json, time
import numpy as np
from kaggle_environments import make
sys.path.insert(0, "scripts")
from distill_lib import DistillPolicy
from lib import fast_sim
from lib.opp_model import lite_greedy_policy


def final_winner(env, ns):
    """Highest final score (ships on planets + in fleets) wins."""
    sc = [0.0] * ns
    obs = env.state[0].observation
    for p in obs["planets"]:
        o = int(p[1])
        if 0 <= o < ns:
            sc[o] += float(p[5])
    for f in obs.get("fleets", []):
        o = int(f[1])
        if 0 <= o < ns:
            sc[o] += float(f[6])
    return int(np.argmax(sc)), sc


def play(policies, ns, max_steps=500):
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    env.reset(num_agents=ns)
    for _ in range(max_steps):
        if env.done:
            break
        acts = [policies[i](env.state[i].observation) for i in range(ns)]
        env.step(acts)
    w, sc = final_winner(env, ns)
    return w


def head_to_head(student, ns, n_games):
    """student in a rotating seat vs lite_greedy filling the rest."""
    wins = 0
    for g in range(n_games):
        seat = g % ns
        pols = [lite_greedy_policy] * ns
        pols[seat] = student_for_seat(student, seat, ns)
        w = play(pols, ns)
        if w == seat:
            wins += 1
    return wins


def student_for_seat(weights, seat, ns):
    pol = DistillPolicy(weights, num_seats=ns)
    return pol  # reads obs['player'] for its own seat


def bench_speed(weights):
    ns = 4
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    env.reset(num_agents=ns)
    for _ in range(40):
        if env.done: break
        env.step([lite_greedy_policy(env.state[i].observation) for i in range(ns)])
    pol = DistillPolicy(weights, num_seats=ns)
    obs0 = env.state[0].observation
    # per-call featurize+score
    pol(obs0)
    t = time.perf_counter()
    for _ in range(2000):
        pol(obs0)
    per_call = (time.perf_counter() - t) / 2000
    # K=50 rollout, all seats distilled
    snap = fast_sim.from_obs(obs0, env.configuration, episode_seed=0, num_seats=ns)
    pols = [DistillPolicy(weights, num_seats=ns) for _ in range(ns)]
    fast_sim.rollout(snap, 50, pols)
    t = time.perf_counter()
    for _ in range(15):
        fast_sim.rollout(snap, 50, pols)
    per_roll = (time.perf_counter() - t) / 15
    print(f"  per-call: {per_call*1e6:7.1f} us   "
          f"K=50 rollout: {per_roll*1000:6.1f} ms  -> {0.6/per_roll:5.0f} rollouts/600ms")
    print(f"  (refs: lite_greedy 5ms/120, top_tier 474ms/1.3)")


if __name__ == "__main__":
    wpath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/distill_weights.json"
    n2 = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n4 = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    weights = json.load(open(wpath))
    if len(sys.argv) > 4:
        weights["threshold"] = float(sys.argv[4])
    print(f"=== threshold={weights['threshold']} ===")
    print("=== SPEED gate ===")
    bench_speed(weights)
    print("=== STRENGTH gate: distilled vs lite_greedy ===")
    t = time.perf_counter()
    w2 = head_to_head(weights, 2, n2)
    print(f"  2P: distilled won {w2}/{n2} = {100*w2/n2:.0f}%  (beat=>50%)")
    w4 = head_to_head(weights, 4, n4)
    print(f"  4P: distilled 1st {w4}/{n4} = {100*w4/n4:.0f}%  (beat=>25%)  "
          f"({time.perf_counter()-t:.0f}s)")
