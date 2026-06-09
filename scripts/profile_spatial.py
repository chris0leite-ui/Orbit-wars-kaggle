"""One-off profiler: where does per-turn time go with the spatial value head?

Runs a single 2P game (agents/baseline vs agents/baseline) with
BASELINE_VALUE_HEAD=hybrid_spatial so the spatial positioning term is active,
under cProfile, and writes a tottime- and cumtime-sorted report to a file.

Also micro-benchmarks _positional_ship_value in isolation on the real
observations seen during the game, to quantify the double-call waste.
"""
import os, sys, time, cProfile, pstats, io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable the spatial head on the source agent (2P -> spatial active).
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid_spatial")
# Mirror the champion config so the profile reflects the real submission.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("JOINT_TOP_K", "5")
os.environ.setdefault("JOINT_MAX_PAIRS", "60")
os.environ.setdefault("REINFORCE_EMIT", "1")
os.environ.setdefault("REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("NEUTRAL_BONUS", "2.0")
os.environ.setdefault("NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("ORBITAL_SAFETY", "1")
os.environ.setdefault("PV_ETA", "1")
os.environ.setdefault("LAUNCH_RULES", "1")
os.environ.setdefault("CAPTURE_HORIZON_K", "10")

import fast

SEED = int(os.environ.get("PROF_SEED", "7"))
OUT = os.environ.get("PROF_OUT", "/tmp/prof_out.txt")

def make_env_seeded(seed):
    import kaggle_environments as ke
    # Raise actTimeout so cProfile's ~2x overhead doesn't trip the 1000ms
    # per-turn cap and corrupt the game; we want a clean full-game profile.
    cfg = {"seed": seed, "actTimeout": 60, "runTimeout": 100000}
    try:
        return ke.make("orbit_wars", configuration=cfg, debug=False)
    except Exception:
        e = ke.make("orbit_wars", debug=False)
        for k, v in cfg.items():
            try:
                setattr(e.configuration, k, v)
            except Exception:
                pass
        return e

def main():
    _, focal_path = fast.resolve_agent_spec("agents/baseline")
    _, opp_path = fast.resolve_agent_spec("agents/baseline")
    focal = fast._load_callable(focal_path)
    opp = fast._load_callable(opp_path)
    env = make_env_seeded(SEED)

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    env.run([focal, opp])
    pr.disable()
    wall = time.time() - t0

    n_steps = len(env.steps)

    buf = io.StringIO()
    buf.write("== profile: 2P game, head=hybrid_spatial, seed=%d ==\n" % SEED)
    buf.write("n_steps=%d  wallclock=%.2fs\n\n" % (n_steps, wall))

    for sort_key in ("tottime", "cumulative"):
        st = pstats.Stats(pr, stream=buf)
        st.strip_dirs().sort_stats(sort_key)
        buf.write("\n----- TOP 30 by %s -----\n" % sort_key)
        st.print_stats(30)

    with open(OUT, "w") as f:
        f.write(buf.getvalue())
    print("wrote", OUT, "n_steps", n_steps, "wall", round(wall, 2))

if __name__ == "__main__":
    main()
