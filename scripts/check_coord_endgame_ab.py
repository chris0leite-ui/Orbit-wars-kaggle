"""n=16 seat-swapped A/B for the coord smooth-ΔW endgame bonus.

Runs TWO modes against orbitfix (the live champion at μ≈1174.9):
- mode A: COORD_DELTA_W=1 + COORD_LAMBDA_W=anchor (the new variant)
- mode B: COORD_DELTA_W=0 (the existing coord, no endgame term)

8 seeds × 2 seats (swap) = 16 games per mode. Reports per-mode win/loss
and the Wilson lower bound for "variant > baseline" inference.

Usage:
    python scripts/check_coord_endgame_ab.py [--seeds 0-7] [--lambda 0.002]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# orbitfix env vars must be set BEFORE importing baseline.
def _prime_orbitfix_env():
    os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
    os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
    os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
    os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
    os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
    os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
    os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
    os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
    os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")


def play_one(seed, coord_as_p0):
    """Play one game; return ('coord', 'orbitfix', or 'tie')."""
    from kaggle_environments import make
    from agents.baseline.main import agent as orbitfix_agent
    from agents.coord.main import agent as coord_agent

    env = make("orbit_wars", configuration={"seed": int(seed)})
    env.reset(num_agents=2)
    turns = 0
    while not env.done and turns < 500:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        if coord_as_p0:
            a0 = coord_agent(obs0)
            a1 = orbitfix_agent(obs1)
        else:
            a0 = orbitfix_agent(obs0)
            a1 = coord_agent(obs1)
        env.step([a0, a1])
        turns += 1
    r0 = env.state[0].reward
    r1 = env.state[1].reward
    coord_r = r0 if coord_as_p0 else r1
    other_r = r1 if coord_as_p0 else r0
    if coord_r > other_r:
        return "coord", turns
    if coord_r < other_r:
        return "orbitfix", turns
    return "tie", turns


def run_panel(seeds, lambda_w, mode):
    """mode='on'        → COORD_DELTA_W=1, DEFEND_BONUS=1
       mode='off'       → COORD_DELTA_W=0
       mode='attackonly'→ COORD_DELTA_W=1, DEFEND_BONUS=0
    """
    if mode == "off":
        os.environ["COORD_DELTA_W"] = "0"
        os.environ["COORD_DEFEND_BONUS"] = "1"
    elif mode == "attackonly":
        os.environ["COORD_DELTA_W"] = "1"
        os.environ["COORD_DEFEND_BONUS"] = "0"
    else:  # "on"
        os.environ["COORD_DELTA_W"] = "1"
        os.environ["COORD_DEFEND_BONUS"] = "1"
    os.environ["COORD_LAMBDA_W"] = f"{lambda_w}"
    results = []
    print(f"\n=== mode {mode} (DELTA_W={'1' if mode == 'on' else '0'}, "
          f"λ_W={lambda_w}) ===", flush=True)
    for s in seeds:
        for coord_as_p0 in (True, False):
            t0 = time.perf_counter()
            winner, turns = play_one(s, coord_as_p0)
            dt = time.perf_counter() - t0
            seat = "P0" if coord_as_p0 else "P1"
            print(f"  seed {s} coord@{seat}: {winner:>10s}  "
                  f"(turns={turns}, elapsed={dt:.1f}s)", flush=True)
            results.append({
                "mode": mode, "seed": s, "coord_as_p0": coord_as_p0,
                "winner": winner, "turns": turns, "elapsed_s": dt,
            })
    return results


def wilson_lower(wins, n, z=1.96):
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-7",
                        help="seed range (e.g. '0-7' or '0,1,2,3')")
    parser.add_argument("--lambda", dest="lambda_w", type=float, default=0.002)
    parser.add_argument("--out", type=str, default="audit/2026-05-22-coord-endgame-ab.json")
    parser.add_argument("--modes", default="on,off",
                        help="comma-separated subset of {on,off,attackonly}")
    args = parser.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    _prime_orbitfix_env()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for mode in modes:
        all_results.extend(run_panel(seeds, args.lambda_w, mode))

    def aggregate(rows):
        n = len(rows)
        w = sum(1 for r in rows if r["winner"] == "coord")
        l = sum(1 for r in rows if r["winner"] == "orbitfix")
        t = sum(1 for r in rows if r["winner"] == "tie")
        return n, w, l, t

    print("\n========================================")
    print("SUMMARY")
    print("========================================")
    summary = {}
    for mode in modes:
        rows = [r for r in all_results if r["mode"] == mode]
        n, w, l, t = aggregate(rows)
        wr = w / n if n else 0.0
        wlo = wilson_lower(w, n)
        print(f"mode {mode:>10s}: {w}W/{l}L/{t}T in n={n}, "
              f"win-rate={wr:.3f}, Wilson-lo={wlo:.3f}")
        summary[mode] = {
            "n": n, "wins": w, "losses": l, "ties": t,
            "win_rate": wr, "wilson_lo": wlo,
        }
    if "on" in summary and "off" in summary:
        print(f"  Δ win-rate (on − off): "
              f"{summary['on']['win_rate'] - summary['off']['win_rate']:+.3f}")
    if "attackonly" in summary and "off" in summary:
        print(f"  Δ win-rate (attackonly − off): "
              f"{summary['attackonly']['win_rate'] - summary['off']['win_rate']:+.3f}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seeds": seeds, "lambda_w": args.lambda_w, "modes": modes,
        "summary": summary,
        "results": all_results,
    }, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
