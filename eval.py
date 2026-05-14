"""eval.py — local A/B for an Orbit Wars agent.

Three modes:
    python eval.py --vs <opponent> [-n 24]    # head-to-head vs one opponent
    python eval.py --panel        [-n 24]     # vs the 3-opponent calibration panel
    python eval.py --smoke                    # 4 games vs random; should be ~100%
    python eval.py --4p           [-n 24]     # 4-player; my agent vs 3 copies of opponent

`<opponent>` can be:
    - a short name: random | starter | nearest | v7_0 | v4_planner
    - a path to a .py file with `def agent(obs)`

Output: per-opponent winrate, Wilson 95 % CI, mean game length.

Why a panel by default
----------------------
Two consecutive prior submissions (`v3.5.1`, `geo v3.1`) over-predicted
the live ladder because they passed local A/B vs a single opponent
(`v7_0`). Calibration is the bottleneck, not raw winrate — the panel
spans architecturally distinct opponents (one weak heuristic, two
strong drop-one and receding-horizon planners) so the local signal
correlates with the live ladder distribution.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Short-name → spec. Resolved to:
#   - kaggle_environments builtin string  ("random", "starter")
#   - path to a .py file (run by env.run via the kaggle harness)
BASELINES: dict[str, str] = {
    "random":     "random",
    "starter":    "starter",
    "nearest":    str(REPO / "baselines" / "nearest.py"),
    "v7_0":       str(REPO / "baselines" / "v7_0.py"),
    "v4_planner": str(REPO / "baselines" / "v4_planner.py"),
}

# 3-opponent calibration panel: 1 weak heuristic + 2 architecturally
# distinct strong baselines. Closes the `local-overpredict-2x` friction.
PANEL = ["nearest", "v7_0", "v4_planner"]

MY_AGENT = str(REPO / "main.py")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


def resolve(spec: str) -> str:
    """Map short-name → spec; pass-through for paths and kaggle builtins."""
    if spec in BASELINES:
        return BASELINES[spec]
    return spec


def _run_game(args):
    """Worker. Each subprocess imports kaggle_environments fresh."""
    my_spec, opp_spec, seed, my_slot, n_players = args
    from kaggle_environments import make

    agents = [opp_spec] * n_players
    agents[my_slot] = my_spec

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    t0 = time.perf_counter()
    env.run(agents)
    elapsed = time.perf_counter() - t0

    final = env.steps[-1]
    rewards = [s.reward if s.reward is not None else -1 for s in final]
    my_reward = rewards[my_slot]
    # Winner = unique-max-reward; ties counted as 0.5 for 2P, 0 for 4P.
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r]
    if n_players == 2:
        win = 1.0 if winners == [my_slot] else (0.5 if my_slot in winners else 0.0)
    else:
        win = 1.0 if winners == [my_slot] else 0.0
    return {
        "seed": seed,
        "my_slot": my_slot,
        "my_reward": my_reward,
        "rewards": rewards,
        "win": win,
        "turns": len(env.steps),
        "elapsed_s": elapsed,
    }


def head_to_head(
    my_spec: str,
    opp_spec: str,
    n: int,
    seed0: int = 1000,
    n_players: int = 2,
    workers: int = 4,
) -> dict:
    """Run n games, rotating my agent through every player slot to remove
    first-player bias."""
    tasks = []
    for i in range(n):
        slot = i % n_players
        tasks.append((my_spec, opp_spec, seed0 + i, slot, n_players))

    games = []
    if workers <= 1:
        for t in tasks:
            games.append(_run_game(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_run_game, t) for t in tasks]):
                games.append(fut.result())

    wins = sum(g["win"] for g in games)
    n_real = len(games)
    p = wins / n_real if n_real else 0.0
    lo, hi = wilson_ci(int(wins), n_real)
    mean_turns = statistics.fmean(g["turns"] for g in games) if games else 0.0
    mean_elapsed = statistics.fmean(g["elapsed_s"] for g in games) if games else 0.0
    return {
        "wins": wins,
        "n": n_real,
        "p": p,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_turns": mean_turns,
        "mean_elapsed_s": mean_elapsed,
        "games": games,
    }


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def print_row(label: str, r: dict, n_players: int) -> None:
    pct = 100 * r["p"]
    lo = 100 * r["wilson_lo"]
    hi = 100 * r["wilson_hi"]
    decisive = (
        "WIN"
        if lo > 50
        else ("LOSS" if hi < 50 else "tie")
    )
    print(
        f"  {label:<14} n={r['n']:>3}  win={pct:5.1f}%  "
        f"95%CI=[{lo:4.1f}, {hi:4.1f}]  "
        f"avg-turns={r['mean_turns']:5.1f}  "
        f"sec/game={r['mean_elapsed_s']:4.1f}  "
        f"verdict={decisive}"
    )


def cmd_smoke(my_spec: str, workers: int) -> int:
    print(f"smoke: {my_spec} vs random (n=4)")
    r = head_to_head(my_spec, "random", n=4, workers=workers)
    print_row("vs random", r, n_players=2)
    return 0 if r["p"] >= 0.75 else 1


def cmd_vs(my_spec: str, opp: str, n: int, workers: int) -> int:
    opp_spec = resolve(opp)
    print(f"head-to-head: {my_spec} vs {opp} (n={n}, 2P)")
    r = head_to_head(my_spec, opp_spec, n=n, workers=workers)
    print_row(f"vs {opp}", r, n_players=2)
    return 0


def cmd_panel(my_spec: str, n: int, workers: int) -> int:
    print(f"panel: {my_spec} vs {PANEL} (n={n} each, 2P)")
    overall_wins = 0
    overall_n = 0
    for opp in PANEL:
        opp_spec = resolve(opp)
        r = head_to_head(my_spec, opp_spec, n=n, workers=workers)
        print_row(f"vs {opp}", r, n_players=2)
        overall_wins += r["wins"]
        overall_n += r["n"]
    if overall_n:
        lo, hi = wilson_ci(int(overall_wins), overall_n)
        print(
            f"  {'OVERALL':<14} n={overall_n:>3}  "
            f"win={100 * overall_wins / overall_n:5.1f}%  "
            f"95%CI=[{100 * lo:4.1f}, {100 * hi:4.1f}]"
        )
    return 0


def cmd_4p(my_spec: str, opp: str, n: int, workers: int) -> int:
    opp_spec = resolve(opp)
    print(f"4-player: {my_spec} vs 3x {opp} (n={n})")
    r = head_to_head(my_spec, opp_spec, n=n, workers=workers, n_players=4)
    print_row(f"4P vs 3x {opp}", r, n_players=4)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=MY_AGENT, help="agent .py to test")
    ap.add_argument("-n", type=int, default=24, help="games per opponent")
    ap.add_argument("-w", "--workers", type=int, default=4, help="parallel workers")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="4 games vs random")
    g.add_argument("--vs", metavar="OPP", help="head-to-head vs one opponent")
    g.add_argument("--panel", action="store_true", help="3-opponent calibration panel")
    g.add_argument("--4p", dest="four_p", metavar="OPP", help="4P: me vs 3x OPP")
    args = ap.parse_args(argv)

    my_spec = args.agent
    if args.smoke:
        return cmd_smoke(my_spec, args.workers)
    if args.vs:
        return cmd_vs(my_spec, args.vs, args.n, args.workers)
    if args.panel:
        return cmd_panel(my_spec, args.n, args.workers)
    if args.four_p:
        return cmd_4p(my_spec, args.four_p, args.n, args.workers)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
