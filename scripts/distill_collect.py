"""Collect (obs, teacher_action) pairs from top_tier_mirror self-play for
distilling a fast rollout policy (the never-built 'Tier 2'). Raw obs +
action are logged; feature extraction happens downstream so the dataset
is reusable. Mixed 2P/4P, varied seeds for state diversity."""
import sys, json, time, random, pickle
from kaggle_environments import make
from lib.opp_model import top_tier_mirror_policy as TEACHER

def collect(n_games, max_steps, out_path, seed0=0):
    samples = []  # list of (player_count, obs_dict, action)
    t0 = time.perf_counter()
    for g in range(n_games):
        ns = random.choice([2, 2, 4])           # ~ladder 55/45 mix
        env = make("orbit_wars", configuration={"episodeSteps": 500})
        env.reset(num_agents=ns)
        for _ in range(max_steps):
            if env.done:
                break
            acts = []
            for i in range(ns):
                obs = env.state[i].observation
                a = TEACHER(obs)
                acts.append(a)
                # log only seat 0's (obs, action) to avoid 4x correlated dupes
                if i == 0:
                    samples.append((ns,
                                    {"planets": [list(p) for p in obs["planets"]],
                                     "fleets": [list(f) for f in obs.get("fleets", [])],
                                     "step": int(obs["step"]),
                                     "player": 0,
                                     "angular_velocity": float(obs.get("angular_velocity", 0.0))},
                                    [list(x) for x in a]))
            env.step(acts)
        if (g + 1) % 10 == 0:
            print(f"  game {g+1}/{n_games}  samples={len(samples)}  "
                  f"{(time.perf_counter()-t0):.0f}s", flush=True)
    with open(out_path, "wb") as fh:
        pickle.dump(samples, fh)
    print(f"DONE: {len(samples)} samples -> {out_path}  ({(time.perf_counter()-t0):.0f}s)")
    return samples

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/distill_smoke.pkl"
    random.seed(0)
    collect(n, ms, out)
