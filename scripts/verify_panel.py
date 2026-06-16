"""Panel verification vs a varied strong field (public-archetype proxies).

The literal external public kernels are gitignored (absent from a fresh
container). Our closest stand-ins, per the top-performer study:
  - agents/baseline   = the Roman-Tamrazov mu1224 rule-base archetype
  - agents/v3_snipe   = converges to the same Roman archetype
plus a varied zoo (geo, v7_minimax).

Two focals, each vs the SAME 3-agent background, so we read both:
  (a) does the CURRENT agent (least_resistance, all levers OFF) beat a varied
      strong field, and
  (b) how does it stack up against `baseline` (the public-strongest proxy) on
      the identical field.

4P, 8 seeds, balanced seats, 250-step cap. Appends each game to a log and
commits+pushes every 2 games (container-restart safe).

    python scripts/verify_panel.py
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
from kaggle_environments import make  # noqa: E402

BRANCH = "claude/dreamy-fermi-8unqi5"
PLAYERS = 4
STEPS = 250
SEEDS = [76670184, 1492346051, 768065184, 641308308,
         305419896, 12648430, 20240617, 88888883]
FOCALS = [
    ("least_resistance", str(REPO / "agents" / "least_resistance" / "main.py")),
    ("baseline(Roman)",  str(REPO / "agents" / "baseline" / "main.py")),
]
BACKGROUND = [
    ("v3_snipe",   str(REPO / "agents" / "v3_snipe" / "main.py")),
    ("geo",        str(REPO / "agents" / "geo" / "main.py")),
    ("v7_minimax", str(REPO / "agents" / "v7_minimax" / "main.py")),
]
LOG = REPO / "audit" / ("panel-verify-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "_p_%s_%d" % (Path(path).stem, id(object())), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _current_agent_env():
    # All experimental levers OFF -> the shipped agent.
    for k in ("LR_LEADER_RELATIVE_4P", "LR_VALUE_COMMIT", "LR_ANYTIME"):
        os.environ[k] = "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ROLLOUT_DEPTH"] = "0"


def _run_game(focal_path, seed, seat):
    _current_agent_env()
    focal = _load(focal_path)
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
    _log("# panel verify | %dP steps<=%d | background={%s} | fair share=%.0f%% | seeds=%s"
         % (PLAYERS, STEPS, bg, 100.0 / PLAYERS, SEEDS))
    n = 0
    for fname, fpath in FOCALS:
        wins = []
        for i, seed in enumerate(SEEDS):
            seat = i % PLAYERS
            win, mx, nsteps, rew = _run_game(fpath, seed, seat)
            wins.append(win)
            n += 1
            _log("%-16s seed=%-11d seat=%d %s steps=%d max_ms=%4.0f rew=%s | %s=%d/%d"
                 % (fname, seed, seat, "WIN " if win else "loss", nsteps, mx, rew,
                    fname, sum(wins), len(wins)))
            if n % 2 == 0:
                _checkpoint("panel-verify checkpoint: %d games (%s %d/%d)"
                            % (n, fname, sum(wins), len(wins)))
        _log("# %s vs {%s}: %d/%d first-place (fair share %.0f%%)"
             % (fname, bg, sum(wins), len(wins), 100.0 / PLAYERS))
    _checkpoint("panel-verify complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
