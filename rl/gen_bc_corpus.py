"""Generate a producer-vs-producer game corpus for behavior cloning.

Sequential (contention-safe), restartable: skips seeds that already
have a dump. Each game -> gzip JSON: per-step planets/fleets/actions
for both seats + final rewards.

Usage: python -m rl.gen_bc_corpus [n_seeds] [out_dir]
"""
import gzip
import json
import os
import sys


def _load_agent_callable(path):
    import importlib.util
    name = "bc_" + os.path.basename(path).replace(".py", "")
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def gen_game(seed: int, out_path: str, agent_path: str):
    from kaggle_environments import make
    ag0 = _load_agent_callable(agent_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([ag0, ag0])

    steps = []
    for i, st in enumerate(env.steps[:-1]):
        obs = st[0].observation
        nxt = env.steps[i + 1]
        # action recorded on the NEXT step's state objects in
        # kaggle_environments: state[s].action is what seat s submitted
        # for the transition into this step.
        steps.append({
            "planets": [list(p) for p in obs["planets"]],
            "fleets": [list(f) for f in (obs.get("fleets") or [])],
            "step": int(obs.get("step", i)),
            "comets": [
                {"planet_ids": list(g["planet_ids"]),
                 "path_index": int(g["path_index"]),
                 "paths": [[list(pt) for pt in p] for p in g["paths"]]}
                for g in (obs.get("comets") or [])
            ],
            "actions": [
                list(nxt[0].action or []),
                list(nxt[1].action or []),
            ],
        })
    head = env.steps[0][0].observation
    rec = {
        "seed": seed,
        "angular_velocity": float(head["angular_velocity"]),
        "initial_planets": [list(p) for p in head["initial_planets"]],
        "comet_planet_ids": list(head.get("comet_planet_ids") or []),
        "rewards": [s.reward for s in env.steps[-1]],
        "n_steps": len(env.steps),
        "steps": steps,
    }
    with gzip.open(out_path, "wt") as f:
        json.dump(rec, f)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/bc_corpus"
    agent_path = "agents/producer/producer_agent.py"
    os.makedirs(out_dir, exist_ok=True)
    base = 500000
    for i in range(n):
        seed = base + i
        path = os.path.join(out_dir, f"prod2p_{seed}.json.gz")
        if os.path.exists(path):
            continue
        try:
            gen_game(seed, path, agent_path)
            print(f"game {seed} done -> {path}", flush=True)
        except Exception as e:
            print(f"game {seed} FAILED: {type(e).__name__} {e}", flush=True)


if __name__ == "__main__":
    main()
