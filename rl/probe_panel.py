"""Small balanced probe: RL agent vs opponents that matter.

For each opponent: 2 seeds x 2 seat orders. Records outcome for the RL
seat plus behavioral counters (max fleets in flight, planets owned at
peak, final material) so a silent never-launches failure is visible.

Usage: python -m rl.probe_panel /tmp/rl_night1.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor


OPPONENTS = [
    ("producer_plus", "agents/producer_plus/producer_agent.py"),
    ("producer", "agents/producer/producer_agent.py"),
    ("ledger_v1_4", "submissions/ledger_v1_4.py"),
]
SEEDS = [101, 202]


def _load_agent_callable(path):
    """Exec a .py agent and return its `agent` callable. Registers the
    module and adds its directory to sys.path first so package-relative
    imports (producer's orbit_lite) resolve — bare file paths passed to
    kaggle_environments silently idle on those."""
    import importlib.util
    import os
    import sys
    name = "probe_" + os.path.basename(path).replace(".py", "") \
        + "_" + str(abs(hash(path)) % 10000)
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    # producer_plus has no orbit_lite of its own — it shares producer's.
    prod_dir = os.path.abspath("agents/producer")
    if "producer_plus" in path and prod_dir not in sys.path:
        sys.path.append(prod_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def run_game(args):
    rl_path, opp_path, seed, rl_seat = args
    from kaggle_environments import make
    rl_agent = _load_agent_callable(rl_path)
    opp_agent = _load_agent_callable(opp_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = ([rl_agent, opp_agent] if rl_seat == 0
              else [opp_agent, rl_agent])
    env.run(agents)

    max_fleets = 0
    max_planets = 0
    for st in env.steps:
        obs = st[0].observation
        fl = sum(1 for f in (obs.get("fleets") or []) if f[1] == rl_seat)
        pl = sum(1 for p in obs["planets"] if p[1] == rl_seat)
        max_fleets = max(max_fleets, fl)
        max_planets = max(max_planets, pl)
    final_obs = env.steps[-1][0].observation
    my_ships = sum(p[5] for p in final_obs["planets"] if p[1] == rl_seat)
    reward = env.steps[-1][rl_seat].reward
    status = env.steps[-1][rl_seat].status
    return {
        "seed": seed, "rl_seat": rl_seat, "reward": reward,
        "status": str(status), "n_steps": len(env.steps),
        "max_fleets": max_fleets, "max_planets": max_planets,
        "final_ships": my_ships,
    }


def main():
    rl_path = sys.argv[1]
    jobs = []
    for opp_name, opp_path in OPPONENTS:
        for seed in SEEDS:
            for rl_seat in (0, 1):
                jobs.append((opp_name, (rl_path, opp_path, seed, rl_seat)))

    # One game per child process: producer and producer_plus both
    # register an `orbit_lite` module with different submodule sets, so
    # they must never share an interpreter.
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as ex:
        results = list(ex.map(run_game, [j[1] for j in jobs]))

    by_opp = {}
    for (opp_name, _), r in zip(jobs, results):
        by_opp.setdefault(opp_name, []).append(r)

    for opp_name, rs in by_opp.items():
        wins = sum(1 for r in rs if r["reward"] == 1)
        print(f"\n=== vs {opp_name}: {wins}/{len(rs)} wins ===")
        for r in rs:
            print(f"  seed={r['seed']} seat={r['rl_seat']} "
                  f"reward={r['reward']} steps={r['n_steps']} "
                  f"max_fleets={r['max_fleets']} "
                  f"max_planets={r['max_planets']} "
                  f"final_ships={r['final_ships']} {r['status']}")


if __name__ == "__main__":
    main()
