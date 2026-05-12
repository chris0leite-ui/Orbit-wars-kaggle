"""PSRO payoff matrix builder — runs all-pairs (upper triangle) tournament.

Computes the empirical payoff matrix P[i][j] = (wins_i − wins_j) / (n_games)
∈ [-1, 1] for a pool of policies. Uses upper-triangle + antisymmetry so
each unordered pair is played once.

Output: JSON with keys
  - policies: list[str]
  - P: list[list[float]]  (full symmetric matrix; antisymmetric: P[i,j] = -P[j,i])
  - n_games_per_pair: int
  - raw: list of per-game records

Usage:
  python -m scripts.psro_tournament \
    --policies v7_minimax v3_snipe precision roi \
    --seeds 4 \
    --episode-steps 500 \
    --out audit/tournaments/psro_payoff_v1.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]


POLICY_PATHS = {
    "v7_minimax": "agents/v7_minimax/main.py",
    "v3_snipe": "agents/v3_snipe/main.py",
    "v3_lookahead": "agents/v3_lookahead/main.py",
    "v4_endgame": "agents/v4_endgame/main.py",
    "v4_hybrid": "agents/v4_hybrid/main.py",
    "v4_mirror": "agents/v4_mirror/main.py",
    "v6_steady": "agents/v6_steady/main.py",
    "v2": "agents/v2/main.py",
    "v1_orbitfix": "agents/v1_orbitfix/main.py",
    "roi": "agents/simple/roi.py",
    "baseline": "data/main.py",
    "precision": "agents/precision/main.py",
}


def _load(name: str) -> callable:
    path = POLICY_PATHS.get(name)
    if path is None:
        raise ValueError(f"unknown policy: {name}")
    full = REPO / path
    spec = importlib.util.spec_from_file_location(f"_psro_{name}", full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _play(agent_a: callable, agent_b: callable, seed: int, episode_steps: int) -> int:
    """Run one game; return +1 if agent_a (P0) wins, -1 if agent_b (P1) wins, 0 draw."""
    env = make("orbit_wars", configuration={"episodeSteps": episode_steps, "seed": seed})
    rec = env.run([agent_a, agent_b])
    last = rec[-1]
    r0, r1 = last[0].reward, last[1].reward
    if r0 == r1:
        return 0
    return 1 if r0 > r1 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True,
                    help=f"Names from: {list(POLICY_PATHS)}")
    ap.add_argument("--seeds", type=int, default=4,
                    help="Number of seeds per ordered pair (both sides played)")
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pool = args.policies
    n = len(pool)
    print(f"PSRO tournament: {n} policies × {args.seeds} seeds × 2 sides per pair")

    # Load all policies once.
    agents = {p: _load(p) for p in pool}
    print(f"  loaded: {list(agents)}")

    # Upper-triangle pairs only; antisymmetric.
    # P[i][j] = (wins of i over j across all games where they meet) /
    #          (total games of i vs j)
    # Stored ∈ [-1, +1].
    wins = [[0] * n for _ in range(n)]   # wins_ij over j across all positions
    games = [[0] * n for _ in range(n)]
    raw = []
    t_all = time.time()
    n_pairs = n * (n - 1) // 2
    p_done = 0

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pool[i], pool[j]
            p_done += 1
            t_pair = time.time()
            print(f"[{p_done}/{n_pairs}] {pi} vs {pj}", flush=True)
            for s in range(args.seeds):
                # i as P0, j as P1
                t0 = time.time()
                outcome = _play(agents[pi], agents[pj], s, args.episode_steps)
                raw.append({
                    "i": pi, "j": pj, "side": "i=P0", "seed": s,
                    "outcome": outcome, "elapsed_s": round(time.time() - t0, 1),
                })
                if outcome == 1:
                    wins[i][j] += 1
                elif outcome == -1:
                    wins[j][i] += 1
                games[i][j] += 1
                games[j][i] += 1

                # j as P0, i as P1
                t0 = time.time()
                outcome = _play(agents[pj], agents[pi], s, args.episode_steps)
                raw.append({
                    "i": pi, "j": pj, "side": "i=P1", "seed": s,
                    "outcome": outcome, "elapsed_s": round(time.time() - t0, 1),
                })
                if outcome == 1:
                    wins[j][i] += 1
                elif outcome == -1:
                    wins[i][j] += 1
                games[i][j] += 1
                games[j][i] += 1
            print(f"    pair done in {time.time() - t_pair:.0f}s; "
                  f"score so far: {pi} {wins[i][j]} vs {pj} {wins[j][i]} (draws: {games[i][j] - wins[i][j] - wins[j][i]})",
                  flush=True)

    # Build antisymmetric payoff matrix.
    # P[i][j] = (wins[i][j] - wins[j][i]) / games[i][j]    where defined; else 0
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i][j] = 0.0
            elif games[i][j] > 0:
                P[i][j] = (wins[i][j] - wins[j][i]) / games[i][j]

    out = {
        "policies": pool,
        "P": P,
        "wins": wins,
        "games": games,
        "n_seeds": args.seeds,
        "episode_steps": args.episode_steps,
        "elapsed_total_s": round(time.time() - t_all, 0),
        "raw": raw,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes); "
          f"total {out['elapsed_total_s']:.0f}s")


if __name__ == "__main__":
    main()
