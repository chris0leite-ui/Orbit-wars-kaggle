"""Seat-balanced 2P A/B: FND vs rebuilt orbitfix consolidated.

n=8 games = 4 seeds × 2 seat assignments. Workers=2 (less CPU contention).
WALLCLOCK_BUDGET_MS=500 caps per-turn time so games complete in reasonable wall.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Parent's env — workers inherit. Clear any baseline / proposer overrides.
for k in list(os.environ):
    if k.startswith("BASELINE_") or k.startswith("PROPOSER_"):
        os.environ.pop(k, None)
os.environ["BASELINE_WALLCLOCK_MS"] = "500"
os.environ["ANALYTICAL_WALLCLOCK_MS"] = "500"


def play_one(args):
    seed, fnd_seat, fnd_path, opp_path = args
    from kaggle_environments import make
    agents = [opp_path, opp_path]
    agents[fnd_seat] = fnd_path
    t0 = time.time()
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 250})
    env.run(agents)
    dt = time.time() - t0
    rewards = [s.reward for s in env.state]
    statuses = [s.status for s in env.state]
    final_step = env.state[0].observation.step
    return {
        "seed": seed, "fnd_seat": fnd_seat, "rewards": rewards,
        "statuses": statuses, "final_step": final_step, "wall_s": dt,
    }


def main():
    fnd = str(Path("submissions/_phase4_step1_FND.py").resolve())
    opp = str(Path("submissions/baseline_joint_aggr_consolidated_orbitfix.py").resolve())
    seeds = [42, 1, 7, 13]
    jobs = []
    for s in seeds:
        for seat in (0, 1):
            jobs.append((s, seat, fnd, opp))

    results = []
    print(f"running n={len(jobs)} games (seeds={seeds} × seat=0,1) with workers=2", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(play_one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            r_fnd = r["rewards"][r["fnd_seat"]]
            r_opp = r["rewards"][1 - r["fnd_seat"]]
            if r_fnd is None or r_opp is None:
                outcome = "ERR"
            elif r_fnd > r_opp:
                outcome = "W"
            elif r_fnd < r_opp:
                outcome = "L"
            else:
                outcome = "D"
            print(f"  seed={r['seed']} fnd@seat{r['fnd_seat']} step={r['final_step']:>3} "
                  f"st={r['statuses']} rw={r['rewards']} -> {outcome} ({r['wall_s']:.1f}s)",
                  flush=True)
    dt = time.time() - t0

    wins = losses = draws = errs = 0
    for r in results:
        a, b = r["rewards"][r["fnd_seat"]], r["rewards"][1 - r["fnd_seat"]]
        if a is None or b is None:
            errs += 1
        elif a > b:
            wins += 1
        elif a < b:
            losses += 1
        else:
            draws += 1
    n = len(results)
    print(f"\n  FND  W={wins}  L={losses}  D={draws}  ERR={errs}  / n={n}  ({dt:.1f}s wall)",
          flush=True)

    from math import sqrt
    if n > 0:
        p = wins / n
        z = 1.96
        denom = 1.0 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        lo = (centre - margin) / denom
        hi = (centre + margin) / denom
        print(f"  Wilson 95% CI on FND win rate: [{lo:.3f}, {hi:.3f}]", flush=True)

    s0 = [r for r in results if r["fnd_seat"] == 0]
    s1 = [r for r in results if r["fnd_seat"] == 1]

    def tally(rs):
        w = sum(1 for r in rs
                if r["rewards"][r["fnd_seat"]] is not None
                and r["rewards"][1 - r["fnd_seat"]] is not None
                and r["rewards"][r["fnd_seat"]] > r["rewards"][1 - r["fnd_seat"]])
        return f"{w}/{len(rs)}"

    print(f"  FND at seat 0 (FND plays first): {tally(s0)}", flush=True)
    print(f"  FND at seat 1 (FND plays second): {tally(s1)}", flush=True)


if __name__ == "__main__":
    main()
