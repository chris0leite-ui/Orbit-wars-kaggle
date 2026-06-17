"""A/B for the hold-sizing + reinforce change, focused on Producer V2.

OFF = current shipped agent. ON = LR_HOLD_MARGIN + LR_DEFEND (take-and-hold:
concentrated, hold-sized strikes + reinforce of threatened own planets).

Jobs (2P, both seats, idle-checked, timed):
  1. OFF vs Producer V2   (baseline; was 3/6)
  2. ON  vs Producer V2   (does take-and-hold beat our peer?)
  3. ON  vs Roman-1224    (regression: must still crush)
  4. ON  vs konbu17       (regression: must still crush)

All on our orbit_lite (V2 verified to run on it). Checkpointed every 2 games.

    python scripts/verify_hold_v2.py
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
sys.path.insert(0, str(REPO / "agents" / "producer"))
from kaggle_environments import make  # noqa: E402

BRANCH = "claude/dreamy-fermi-8unqi5"
EXT = REPO / "audit" / "external" / "agents"
LR = str(REPO / "agents" / "least_resistance" / "main.py")
V2 = str(EXT / "slawekbiel_the-producer-v2" / "main.py")
ROMAN = str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py")
KONBU = str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py")
SEEDS = [768065184, 76670184, 305419896, 12648430]
SEEDS2 = [768065184, 305419896]
JOBS = [
    ("OFF", "producerV2", V2, SEEDS),
    ("ON ", "producerV2", V2, SEEDS),
    ("ON ", "roman1224",  ROMAN, SEEDS2),
    ("ON ", "konbu17",    KONBU, SEEDS2),
]
LOG = REPO / "audit" / ("hold-v2-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location("_hv_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "agent", None) or getattr(mod, "act")


def _setenv(on):
    for k in ("LR_LEADER_RELATIVE_4P", "LR_VALUE_COMMIT", "LR_ANYTIME"):
        os.environ[k] = "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ROLLOUT_DEPTH"] = "0"
    os.environ["LR_HOLD_MARGIN"] = "0.5" if on else "0.0"
    os.environ["LR_DEFEND"] = "1" if on else "0"


def _arity(fn):
    argc = fn.__code__.co_argcount if hasattr(fn, "__code__") else 2
    return (lambda o: fn(o)) if argc == 1 else (lambda o, c=None: fn(o, c))


def _run(opp_path, seed, focal_seat, on):
    _setenv(on)
    focal = _load(LR)
    opp = _load(opp_path)
    ts = []

    def timed(o, c=None):
        t = time.perf_counter()
        try:
            return focal(o, c)
        finally:
            ts.append((time.perf_counter() - t) * 1000.0)

    seats = [None, None]
    seats[focal_seat] = timed
    seats[1 - focal_seat] = _arity(opp)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(seats)
    rew = [s.reward for s in env.steps[-1]]
    fr = rew[focal_seat]
    valid = [r for r in rew if r is not None]
    win = fr is not None and bool(valid) and fr == max(valid)
    fl = sum(len(st[focal_seat].action) for st in env.steps if st[focal_seat].action)
    ol = sum(len(st[1 - focal_seat].action) for st in env.steps if st[1 - focal_seat].action)
    return win, fl, ol, len(env.steps), (max(ts) if ts else 0.0), rew


def _log(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def _checkpoint(msg):
    for cmd in (["git", "add", str(LOG)], ["git", "commit", "-q", "-m", msg],
                ["git", "push", "-q", "origin", BRANCH]):
        subprocess.run(cmd, cwd=str(REPO), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    _log("# hold-sizing+reinforce A/B | 2P | ON=LR_HOLD_MARGIN=0.5,LR_DEFEND=1")
    n = 0
    for variant, oname, opath, seeds in JOBS:
        on = variant.strip() == "ON"
        wins = invalid = games = 0
        mxall = 0.0
        for seed in seeds:
            for seat in (0, 1):
                win, fl, ol, steps, mx, rew = _run(opath, seed, seat, on)
                games += 1
                n += 1
                mxall = max(mxall, mx)
                idle = (fl == 0) or (ol == 0)
                if idle:
                    invalid += 1
                else:
                    wins += int(win)
                tag = "INVALID-IDLE" if idle else ("WIN " if win else "loss")
                _log("%s vs %-11s seed=%-11d seat=%d %s steps=%d "
                     "focal_launch=%d opp_launch=%d max_ms=%4.0f rew=%s"
                     % (variant, oname, seed, seat, tag, steps, fl, ol, mx, rew))
                if n % 2 == 0:
                    _checkpoint("hold-v2 checkpoint: %d games" % n)
        _log("# %s vs %s: %d/%d wins on valid games (%d idle); focal max_ms=%.0f"
             % (variant, oname, wins, games - invalid, invalid, mxall))
    _checkpoint("hold-v2 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
