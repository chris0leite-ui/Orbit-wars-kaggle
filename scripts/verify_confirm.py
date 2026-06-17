"""n>=32 confirmation of take-and-hold (LR_HOLD_MARGIN + LR_DEFEND) vs Producer V2.

OFF = shipped agent.  ON = hold-sizing + reinforce.
Interleaves OFF/ON per (seed, seat) for a paired read; idle-checked; timed.

  2P block: focal vs Producer V2          | 16 seeds x 2 seats = 32 per variant.
  4P block: focal vs {V2, Roman1224, konbu17} | 8 seeds x 4 seats = 32 per variant.

The 4P block is the must-pass regression gate (these levers are NOT mode-gated).
Checkpointed every 4 games (one seed/seat cell). Runs 2P first.

    python scripts/verify_confirm.py
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
SEEDS = [76670184, 1492346051, 768065184, 641308308, 305419896, 12648430,
         20240617, 88888883, 13, 42, 777, 2024, 555555, 31337, 9001, 123456789]
LOG = REPO / "audit" / ("confirm-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location("_c_%d" % abs(hash(path)), path)
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


def _run(opp_paths, focal_seat, seed, on):
    _setenv(on)
    players = len(opp_paths) + 1
    focal = _load(LR)
    opps = [_load(p) for p in opp_paths]
    ts = []

    def timed(o, c=None):
        t = time.perf_counter()
        try:
            return focal(o, c)
        finally:
            ts.append((time.perf_counter() - t) * 1000.0)

    seats = [None] * players
    seats[focal_seat] = timed
    j = 0
    for i in range(players):
        if i == focal_seat:
            continue
        seats[i] = _arity(opps[j])
        j += 1
    try:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(seats)
    except Exception as e:
        return None, 0, 0, 0, 0.0, repr(e)[:60]
    rew = [s.reward for s in env.steps[-1]]
    fr = rew[focal_seat]
    valid = [r for r in rew if r is not None]
    win = fr is not None and bool(valid) and fr == max(valid)
    fl = sum(len(st[focal_seat].action) for st in env.steps if st[focal_seat].action)
    ol = sum(len(st[i].action or []) for st in env.steps
             for i in range(players) if i != focal_seat)
    return win, fl, ol, len(env.steps), (max(ts) if ts else 0.0), None


def _log(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def _ck(msg):
    for cmd in (["git", "add", str(LOG)], ["git", "commit", "-q", "-m", msg],
                ["git", "push", "-q", "origin", BRANCH]):
        subprocess.run(cmd, cwd=str(REPO), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _block(name, opp_paths, seeds, n):
    players = len(opp_paths) + 1
    tally = {"OFF": [0, 0, 0], "ON": [0, 0, 0]}   # wins, valid, idle
    mxall = {"OFF": 0.0, "ON": 0.0}
    games = 0
    _log("## %s block: focal vs %d opp(s), %d seeds x %d seats = %d/variant"
         % (name, len(opp_paths), len(seeds), players, len(seeds) * players))
    for seed in seeds:
        for seat in range(players):
            for on, key in ((False, "OFF"), (True, "ON")):
                win, fl, ol, steps, mx, err = _run(opp_paths, seat, seed, on)
                games += 1
                if err is not None:
                    _log("%s %s seed=%-11d seat=%d ERROR %s" % (name, key, seed, seat, err))
                    continue
                mxall[key] = max(mxall[key], mx)
                idle = (fl == 0) or (ol == 0)
                tally[key][1 if not idle else 2] += 1
                if not idle and win:
                    tally[key][0] += 1
                tag = "IDLE" if idle else ("WIN " if win else "loss")
                _log("%s %s seed=%-11d seat=%d %s steps=%d fl=%d ol=%d max_ms=%4.0f"
                     % (name, key, seed, seat, tag, steps, fl, ol, mx))
            if games % 4 == 0:
                _ck("%s checkpoint: OFF %d/%d, ON %d/%d"
                    % (name, tally["OFF"][0], tally["OFF"][1],
                       tally["ON"][0], tally["ON"][1]))
    _log("# %s RESULT  OFF %d/%d  ON %d/%d  (idle OFF=%d ON=%d; max_ms OFF=%.0f ON=%.0f)"
         % (name, tally["OFF"][0], tally["OFF"][1], tally["ON"][0], tally["ON"][1],
            tally["OFF"][2], tally["ON"][2], mxall["OFF"], mxall["ON"]))


def main():
    _log("# n>=32 confirmation | ON=LR_HOLD_MARGIN=0.5,LR_DEFEND=1 | %s"
         % datetime.now(timezone.utc).isoformat())
    _block("2P", [V2], SEEDS[:16], 32)
    _block("4P", [V2, ROMAN, KONBU], SEEDS[:8], 32)
    _ck("confirm complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
