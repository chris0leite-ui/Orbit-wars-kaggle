"""Current agent vs REAL public competitor agents (pulled from Kaggle kernels).

Background (the 3 strongest public agents that actually play, confirmed by
launch-count): konbu17 ML-hybrid (~85% panel), Roman Tamrazov LB-1224 rule-base,
ykhnkf distance-prioritized LB-1100. Focal = least_resistance (all levers OFF
= the shipped agent). 4P, 8 seeds, balanced seats, 250-step cap. fair share 25%.

External agent code lives under audit/external/ (gitignored, not committed).
Appends each game to a committed log and pushes every 2 games (restart-safe).

    python scripts/verify_public.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))  # exposes orbit_lite if needed
from kaggle_environments import make  # noqa: E402

BRANCH = "claude/dreamy-fermi-8unqi5"
PLAYERS = 4
STEPS = 250
SEEDS = [76670184, 1492346051, 768065184, 641308308,
         305419896, 12648430, 20240617, 88888883]
EXT = REPO / "audit" / "external" / "agents"
FOCAL = ("least_resistance", str(REPO / "agents" / "least_resistance" / "main.py"))
BACKGROUND = [
    ("konbu17",   str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py")),
    ("roman1224", str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py")),
    ("ykhnkf1100", str(EXT / "ykhnkf_distance-prioritized-agent-lb-max-score-1100" / "main.py")),
]
LOG = REPO / "audit" / ("public-verify-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "_pub_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "agent", None) or getattr(mod, "act")


def _current_agent_env():
    for k in ("LR_LEADER_RELATIVE_4P", "LR_VALUE_COMMIT", "LR_ANYTIME"):
        os.environ[k] = "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ROLLOUT_DEPTH"] = "0"


def _run_game(seed, seat):
    _current_agent_env()
    focal = _load(FOCAL[1])
    bgs = [_load(p) for _, p in BACKGROUND]
    ts = []

    def timed(o, c=None):
        t = time.perf_counter()
        try:
            return focal(o, c)
        finally:
            ts.append((time.perf_counter() - t) * 1000.0)

    seats = [None] * PLAYERS
    seats[seat] = timed
    j = 0
    for i in range(PLAYERS):
        if i == seat:
            continue
        seats[i] = bgs[j]
        j += 1
    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": STEPS}, debug=False)
    env.run(seats)
    final = env.steps[-1]
    rew = [s.reward for s in final]
    fr = rew[seat]
    valid = [r for r in rew if r is not None]
    win = fr is not None and bool(valid) and fr == max(valid)
    return win, (max(ts) if ts else 0.0), len(env.steps), rew


def _log(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def _checkpoint(msg):
    for cmd in (["git", "add", str(LOG)],
                ["git", "commit", "-q", "-m", msg],
                ["git", "push", "-q", "origin", BRANCH]):
        subprocess.run(cmd, cwd=str(REPO), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    bg = "+".join(n for n, _ in BACKGROUND)
    _log("# PUBLIC panel | %dP steps<=%d | focal=%s | background={%s} | fair=%.0f%% | seeds=%s"
         % (PLAYERS, STEPS, FOCAL[0], bg, 100.0 / PLAYERS, SEEDS))
    wins = []
    for i, seed in enumerate(SEEDS):
        seat = i % PLAYERS
        win, mx, nsteps, rew = _run_game(seed, seat)
        wins.append(win)
        _log("%-16s seed=%-11d seat=%d %s steps=%d max_ms=%4.0f rew=%s | %d/%d"
             % (FOCAL[0], seed, seat, "WIN " if win else "loss", nsteps, mx, rew,
                sum(wins), len(wins)))
        if len(wins) % 2 == 0:
            _checkpoint("public-verify checkpoint: %d/%d (%d first-place)"
                        % (len(wins), len(SEEDS), sum(wins)))
    _log("# SUMMARY  %s vs {%s}: %d/%d first-place (fair share %.0f%%)"
         % (FOCAL[0], bg, sum(wins), len(wins), 100.0 / PLAYERS))
    _checkpoint("public-verify complete: %d/%d" % (sum(wins), len(wins)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
