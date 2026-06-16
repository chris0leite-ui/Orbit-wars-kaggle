"""Deep-search-ISOLATED verification vs the producer (fast feedback).

Context: the 16-game producer A/B showed the leader-relative + value-commit
levers REGRESS 4P (12/16 -> 6/16), and the current agent already crushes the
producer (12/16 = 75%, fair share 25%). So deep search must be tested ALONE on
the strong current base, not stacked on those levers.

  ON  = current base + LR_ROLLOUT_DEPTH (deeper K-turn producer rollout only).
  OFF = current shipped agent (all levers off).

8 distinct seeds, seats balanced (seat = i % players), capped at 250 steps,
vs producer x3, interleaved OFF/ON per seed. Appends each game to a log and
commits+pushes it every 2 games so partial results survive a container restart.

    python scripts/verify_deep_search.py
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
DEPTH = int(os.environ.get("VERIFY_DEPTH", "4"))
SEEDS = [76670184, 1492346051, 768065184, 641308308,
         305419896, 12648430, 20240617, 88888883]
LR = str(REPO / "agents" / "least_resistance" / "main.py")
PRODUCER = str(REPO / "agents" / "producer" / "main.py")
LOG = REPO / "audit" / ("deep-verify-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "_v_%s_%d" % (Path(path).stem, id(object())), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _variant_env(on):
    # The regressing levers stay OFF in BOTH arms; only deep rollout toggles.
    os.environ["LR_LEADER_RELATIVE_4P"] = "0"
    os.environ["LR_VALUE_COMMIT"] = "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ANYTIME"] = "0"
    os.environ["LR_ROLLOUT_DEPTH"] = str(DEPTH) if on else "0"


def _run_game(seed, seat):
    focal = _load(LR)
    bgs = [_load(PRODUCER) for _ in range(PLAYERS - 1)]
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
    _log("# deep-search ISOLATED vs producer | %dP depth=%d steps<=%d seeds=%s"
         % (PLAYERS, DEPTH, STEPS, SEEDS))
    off, on, n = [], [], 0
    for i, seed in enumerate(SEEDS):
        seat = i % PLAYERS
        for on_flag, name, acc in ((False, "OFF", off), (True, "ON ", on)):
            _variant_env(on_flag)
            win, mx, nsteps, rew = _run_game(seed, seat)
            acc.append(win)
            n += 1
            _log("%s seed=%-11d seat=%d %s steps=%d max_ms=%4.0f rew=%s "
                 "| OFF=%d/%d ON=%d/%d"
                 % (name, seed, seat, "WIN " if win else "loss", nsteps, mx, rew,
                    sum(off), len(off), sum(on), len(on)))
            if n % 2 == 0:
                _checkpoint("deep-verify checkpoint: %d games (OFF %d/%d, ON %d/%d)"
                            % (n, sum(off), len(off), sum(on), len(on)))
    _log("# SUMMARY  OFF=%d/%d   ON(deep d=%d)=%d/%d   vs producer %dP"
         % (sum(off), len(off), DEPTH, sum(on), len(on), PLAYERS))
    _checkpoint("deep-verify complete: OFF %d/%d ON %d/%d"
                % (sum(off), len(off), sum(on), len(on)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
