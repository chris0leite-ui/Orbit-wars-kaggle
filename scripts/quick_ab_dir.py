"""Subprocess-per-game A/B that loads agents by dir name (no bundling).

Each game runs in a fresh subprocess → no env-var pollution. Loads
agents via `from agents.<name>.main import agent` so we can A/B
production agent dirs directly. Supports 2P or 4P.

Usage:
    python scripts/quick_ab_dir.py focal opp --seeds 2 --players 4 \
        --background opp,opp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _make_runner_code(seed: int, focal_seat: int, focal_name: str,
                     opp_names: list[str], n_players: int,
                     episode_steps: int) -> str:
    return f"""
import json, sys, time, importlib
sys.path.insert(0, {str(REPO)!r})
from kaggle_environments import make

def _load(name):
    mod = importlib.import_module(f'agents.{{name}}.main')
    return mod.agent

n_players = {n_players}
focal_seat = {focal_seat}
focal_name = {focal_name!r}
opp_names = {opp_names!r}
agents = []
opp_iter = iter(opp_names)
for i in range(n_players):
    if i == focal_seat:
        agents.append(_load(focal_name))
    else:
        agents.append(_load(next(opp_iter)))

env = make('orbit_wars', configuration={{'seed': {seed}, 'episodeSteps': {episode_steps}}}, debug=False)
t0 = time.perf_counter()
env.run(agents)
wall = time.perf_counter() - t0
rewards = [s.reward for s in env.state]
final_step = env.state[0].observation.step
print(json.dumps({{'rewards': rewards, 'final_step': final_step, 'wall': wall, 'focal_seat': focal_seat}}))
"""


def _worker_play(args):
    seed, focal_seat, focal_name, opp_names, n_players, episode_steps = args
    code = _make_runner_code(seed, focal_seat, focal_name, opp_names,
                             n_players, episode_steps)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired as e:
        return {"seed": seed, "outcome": "timeout", "stderr": f"timeout {e.timeout}s"}
    lines = (proc.stdout or "").strip().splitlines()
    line = next((l for l in reversed(lines) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "outcome": "error",
                "stderr": (proc.stderr or "")[:600]}
    data = json.loads(line)
    rewards = data["rewards"]
    fs = data["focal_seat"]
    fr = rewards[fs]
    if fr is None:
        outcome = "error"
    elif n_players == 2:
        other = rewards[1 - fs]
        outcome = "win" if fr > other else ("draw" if fr == other else "loss")
    else:
        # 4P: focal wins if it has top reward (unique max)
        max_r = max(r for r in rewards if r is not None)
        n_max = sum(1 for r in rewards if r == max_r)
        if fr == max_r and n_max == 1:
            outcome = "win"
        elif fr == max_r:
            outcome = "tie_top"
        else:
            outcome = "loss"
    return {"seed": seed, "focal_seat": fs, "outcome": outcome,
            "rewards": rewards, "final_step": data["final_step"],
            "wall": data["wall"]}


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("focal", help="agent dir name under agents/")
    ap.add_argument("opp", help="opponent dir name under agents/ (default fill for background)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--players", type=int, choices=(2, 4), default=4)
    ap.add_argument("--background", default=None,
                    help="comma-separated opp names for 4P background seats (defaults to opp,opp,opp)")
    ap.add_argument("--seat-rotate", action="store_true",
                    help="rotate focal seat across (0..n_players-1) for each seed (default: all seats)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--episode-steps", type=int, default=250)
    args = ap.parse_args()

    if args.players == 2:
        opp_names_pool = [args.opp]
        seats = [0, 1]
    else:
        if args.background:
            bg = args.background.split(",")
            assert len(bg) == args.players - 1, f"--background must list {args.players-1} agents"
            opp_names_pool = bg
        else:
            opp_names_pool = [args.opp] * (args.players - 1)
        seats = list(range(args.players)) if args.seat_rotate else [0, 1]

    tasks = []
    for seed in range(args.seeds):
        for seat in seats:
            tasks.append((seed, seat, args.focal, opp_names_pool, args.players, args.episode_steps))

    print(f"== quick_ab_dir focal={args.focal} opp={args.opp} players={args.players} "
          f"seeds={args.seeds} seats={seats} → {len(tasks)} games workers={args.workers} ==")

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error", "stderr": f"{type(e).__name__}: {e}"[:300]}
            results.append(r)
            print(f"   seed={r.get('seed','?'):>3} seat={r.get('focal_seat','-')} "
                  f"{r.get('outcome','?'):>8} rewards={r.get('rewards','-')} "
                  f"steps={r.get('final_step','-')} wall={r.get('wall',0):.1f}s "
                  f"{'err: ' + r.get('stderr','')[:200] if r.get('stderr') else ''}")
    wins = sum(1 for r in results if r.get("outcome") == "win")
    ties = sum(1 for r in results if r.get("outcome") == "tie_top")
    losses = sum(1 for r in results if r.get("outcome") == "loss")
    draws = sum(1 for r in results if r.get("outcome") == "draw")
    errs = sum(1 for r in results if r.get("outcome") in ("error", "timeout"))
    n = wins + ties + losses + draws
    elapsed = time.perf_counter() - t0
    print(f"\n   focal_wins={wins} ties={ties} losses={losses} draws={draws} errs={errs} "
          f"(n_valid={n}) elapsed={elapsed:.0f}s")
    if n > 0:
        print(f"   win_rate={wins/n:.3f}  (ties_top counted separately)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
