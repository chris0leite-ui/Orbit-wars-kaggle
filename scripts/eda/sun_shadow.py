"""Mine 5 — Sun-shadow exploitation audit.

For each replay, for each turn, count "shielded turns per owned planet"
of the winner. A planet is *shielded* on a turn when the nearest enemy
planet's straight-line shot to it would be blocked by the sun. Then
report (a) total shielded turns, (b) per-planet shielded-turn count
mean, (c) Spearman of shielded-turn-count vs survived-to-end.

CLI:
    python -m scripts.eda.sun_shadow --replay-dirs audit/external/replays --out audit/<date>-sun-shadow.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib.geometry import path_clears_sun  # noqa: E402


def _winner_seat(replay):
    rewards = replay.get("rewards") or []
    if rewards.count(1) == 1:
        return rewards.index(1)
    return None


def _spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    rank_x = [0] * len(xs)
    rank_y = [0] * len(ys)
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    mx = statistics.fmean(rank_x)
    my = statistics.fmean(rank_y)
    num = sum((rank_x[i] - mx) * (rank_y[i] - my) for i in range(len(xs)))
    dx = math.sqrt(sum((rank_x[i] - mx) ** 2 for i in range(len(xs))))
    dy = math.sqrt(sum((rank_y[i] - my) ** 2 for i in range(len(ys))))
    if dx * dy == 0:
        return None
    return num / (dx * dy)


def extract(replay_path: Path) -> dict:
    replay = json.load(open(replay_path))
    winner = _winner_seat(replay)
    if winner is None:
        return None
    steps = replay["steps"]
    n_steps = len(steps)

    # Track per-planet shielded turns
    shielded_turns = {}   # pid -> count of turns shielded by sun
    owned_turns = {}      # pid -> count of turns owned by winner

    for step_idx, step in enumerate(steps):
        obs = step[0]["observation"]
        planets = obs["planets"]
        owned = [p for p in planets if p[1] == winner]
        enemies = [p for p in planets if p[1] >= 0 and p[1] != winner]
        for op in owned:
            pid = op[0]
            owned_turns[pid] = owned_turns.get(pid, 0) + 1
            if enemies:
                # Find nearest enemy planet by Euclidean distance.
                nearest = min(enemies, key=lambda e: math.hypot(e[2] - op[2], e[3] - op[3]))
                shielded = not path_clears_sun(
                    (op[2], op[3]), (nearest[2], nearest[3])
                )
                if shielded:
                    shielded_turns[pid] = shielded_turns.get(pid, 0) + 1

    # End state — which planets still owned by winner?
    end_planets = {p[0]: p[1] for p in steps[-1][0]["observation"]["planets"]}

    # Per-planet rows
    rows = []
    for pid, n_owned in owned_turns.items():
        n_shield = shielded_turns.get(pid, 0)
        survived = end_planets.get(pid) == winner
        rows.append({
            "pid": pid,
            "owned_turns": n_owned,
            "shielded_turns": n_shield,
            "shielded_frac": n_shield / n_owned if n_owned else 0,
            "survived": survived,
        })

    # Spearman: shielded-fraction vs survived
    if len(rows) >= 3:
        xs = [r["shielded_frac"] for r in rows]
        ys = [1.0 if r["survived"] else 0.0 for r in rows]
        rho = _spearman(xs, ys)
    else:
        rho = None

    return {
        "replay": replay_path.name,
        "winner_seat": winner,
        "n_steps": n_steps,
        "n_owned_planets": len(rows),
        "total_shielded_turns": sum(r["shielded_turns"] for r in rows),
        "mean_shielded_frac": statistics.fmean(r["shielded_frac"] for r in rows) if rows else 0,
        "survival_rate": statistics.fmean(1.0 if r["survived"] else 0.0 for r in rows) if rows else 0,
        "spearman_shielded_vs_survived": rho,
        "planet_rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args(argv)

    games = []
    for d in args.replay_dirs:
        for fp in sorted(Path(d).glob(args.glob)):
            try:
                g = extract(fp)
                if g:
                    games.append(g)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_games": len(games), "games": games}))
    print(f"wrote {len(games)} games -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
